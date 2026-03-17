import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union
from anyio import fail_after, get_cancelled_exc_class
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from pydantic import BaseModel

# 配置与导入
from config.settings import ALLOWED_ORIGINS, SSE_MAX_INFLIGHT, SSE_ACQUIRE_TIMEOUT, PII_ENABLED, PRIMARY_MODEL_NAME, PRIMARY_DEFAULT_MAX_TOKENS
from services.api_service import run_usage_pipeline, check_input_safety, generate_safe_response
from workflows import scanner, qwen_guard
from utils.sse_stream import gated_streaming_generator

# PII 脱敏：按需加载，presidio 未安装时静默降级
try:
    from guards.pii_redactor import redact as _pii_redact, available as _pii_available
    _pii_redact_fn = _pii_redact if (PII_ENABLED and _pii_available) else None
except Exception:
    _pii_redact_fn = None

# 初始化
# 不调用 basicConfig——uvicorn 启动时会重置 root logger 的 handler，导致 basicConfig 失效。
# 正确做法：复用 uvicorn 已配置好的 handler，把业务 logger 的日志传播到 uvicorn logger。
logging.getLogger("uvicorn.access").setLevel(logging.INFO)
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT", "50"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
CancelledError = get_cancelled_exc_class()

_inflight: Optional[asyncio.Semaphore] = None
_sse_inflight: Optional[asyncio.Semaphore] = None
ACTIVE_TASKS: Dict[str, asyncio.Task] = {}
ACTIVE_TASKS_LOCK: Optional[asyncio.Lock] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _inflight, ACTIVE_TASKS_LOCK, _sse_inflight
    _inflight = asyncio.Semaphore(MAX_INFLIGHT) if MAX_INFLIGHT > 0 else None
    _sse_inflight = asyncio.Semaphore(SSE_MAX_INFLIGHT) if SSE_MAX_INFLIGHT > 0 else None
    ACTIVE_TASKS_LOCK = asyncio.Lock()
    yield
    from utils.http_client import aclose_all_http_clients
    await aclose_all_http_clients()


app = FastAPI(title="LLM Safety Guard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModerateRequest(BaseModel):
    prompt: str
    history: Optional[list] = None
    stream: Optional[bool] = False
    primary_model_name: Optional[str] = None
    provider: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = None
    primary_model_url: Optional[str] = None
    primary_max_tokens: Optional[int] = None
    protection_enabled: Optional[bool] = True
    stream_chunk_size: Optional[int] = None  # 预留字段，暂不接入流程


class ModerateResponse(BaseModel):
    prompt: str
    history: Optional[list] = None
    route: Optional[str] = None
    prompt_moderation: Optional[Dict[str, Any]] = None
    response: Optional[str] = None
    reasoning_content: Optional[str] = None
    response_moderation: Optional[Dict[str, Any]] = None
    keyword_flagged: bool = False
    protection_enabled: bool = True
    logs: list = []
    metadata: Dict[str, Any] = {}


class ReloadResponse(BaseModel):
    ok: bool
    word_count: int


@app.middleware("http")
async def add_req_id(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.req_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


@app.post("/moderate", response_model=ModerateResponse)
async def moderate_endpoint(
    request: Request,
    payload: Union[ModerateRequest, Dict[str, Any], str] = Body(...),
):
    req_id = request.state.req_id
    t_start = time.monotonic()

    try:
        if isinstance(payload, ModerateRequest):
            # 逐字段显式提取，避免 model_dump() 在 Pydantic v2 Union 类型下丢失 Optional 字段值
            data = {
                "prompt": payload.prompt,
                "history": payload.history,
                "stream": payload.stream,
                "primary_model_name": payload.primary_model_name,
                "provider": payload.provider,
                "model_params": payload.model_params,
                "primary_model_url": payload.primary_model_url,
                "primary_max_tokens": payload.primary_max_tokens,
                "protection_enabled": payload.protection_enabled,
                "stream_chunk_size": payload.stream_chunk_size,
            }
        elif isinstance(payload, str):
            data = json.loads(payload)
        else:
            data = dict(payload)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    prompt = data.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(status_code=400, detail="字段 'prompt' 不能为空")

    provider_name = data.get("provider") or "openai"
    model_name = data.get("primary_model_name") or PRIMARY_MODEL_NAME
    want_stream = bool(data.get("stream", False))
    protection_enabled = bool(data.get("protection_enabled", True))

    # 确保 data 中的 primary_model_name 被设置（用于传给 run_usage_pipeline）
    if not data.get("primary_model_name"):
        data["primary_model_name"] = model_name

    logger.info(
        "REQ_IN  req_id=%s provider=%s model=%s stream=%s protection=%s prompt=%.100s",
        req_id, provider_name, model_name, want_stream, protection_enabled, prompt,
    )

    # 规范化便捷字段：primary_model_url → provider_config.base_url
    if data.get("primary_model_url"):
        if not data.get("provider_config"):
            data["provider_config"] = {}
        if not data["provider_config"].get("base_url"):
            data["provider_config"]["base_url"] = data["primary_model_url"]

    if data.get("primary_max_tokens"):
        if not data.get("model_params"):
            data["model_params"] = {}
        if not data["model_params"].get("max_tokens"):
            data["model_params"]["max_tokens"] = data["primary_max_tokens"]

    if want_stream:
        from workflows.usage_workflow import _get_provider, async_client_oyster, MODEL_OYSTER
        t_provider = time.monotonic()
        provider = _get_provider(data)
        logger.info("PROVIDER_INIT req_id=%s provider=%s elapsed=%.3fs",
                    req_id, provider_name, time.monotonic() - t_provider)

        history = data.get("history") or []
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": prompt})
        model_params = data.get("model_params") or {}
        # 若调用方未指定 max_tokens，使用配置默认值，避免模型服务端默认值截断长推理
        if "max_tokens" not in model_params:
            model_params = {**model_params, "max_tokens": PRIMARY_DEFAULT_MAX_TOKENS}

        async def primary_stream_factory():
            t0 = time.monotonic()
            logger.info("STREAM_START req_id=%s model=%s", req_id, model_name)
            result = await provider.async_chat_completion(
                messages=messages,
                model=model_name,
                stream=True,
                **model_params,
            )
            logger.info("STREAM_CONNECTED req_id=%s elapsed=%.3fs", req_id, time.monotonic() - t0)
            return result

        async def oyster_stream_factory(original_prompt: str):
            response = await async_client_oyster.chat.completions.create(
                model=MODEL_OYSTER,
                messages=[{"role": "user", "content": original_prompt[-2000:]}],
                stream=True,
            )
            async def _iter():
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            return _iter()

        async def final_audit_callback(full_text: str, route: str, audit_result: dict):
            elapsed = time.monotonic() - t_start
            logger.info(
                "STREAM_DONE req_id=%s route=%s safe=%s total_elapsed=%.3fs resp_chars=%d",
                req_id, route, audit_result.get("safe"), elapsed, len(full_text),
            )

        if _sse_inflight is not None:
            try:
                await asyncio.wait_for(_sse_inflight.acquire(), timeout=SSE_ACQUIRE_TIMEOUT)
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=503,
                    detail=f"流式请求并发已达上限（{SSE_MAX_INFLIGHT}），请稍后重试",
                )

            async def guarded_gen():
                try:
                    async for chunk in gated_streaming_generator(
                        primary_stream_factory=primary_stream_factory,
                        model_name=model_name,
                        request_id=req_id,
                        prompt=prompt,
                        scanner_fn=scanner.scan,
                        hybrid_guard_fn=qwen_guard.async_check,
                        oyster_stream_factory=oyster_stream_factory,
                        final_audit_callback=final_audit_callback,
                        pii_redact_fn=_pii_redact_fn,
                    ):
                        yield chunk
                except CancelledError:
                    logger.info("[stream:%s] 客户端断开连接，释放流式槽位", req_id)
                    raise
                except Exception as e:
                    logger.error("[stream:%s] 流式生成异常: %s，释放流式槽位", req_id, e, exc_info=True)
                    raise
                finally:
                    # 确保无论如何都释放槽位（包括 CancelledError、异常、正常结束）
                    if _sse_inflight is not None:
                        _sse_inflight.release()

            return StreamingResponse(guarded_gen(), media_type="text/event-stream")

        return StreamingResponse(
            gated_streaming_generator(
                primary_stream_factory=primary_stream_factory,
                model_name=model_name,
                request_id=req_id,
                prompt=prompt,
                scanner_fn=scanner.scan,
                hybrid_guard_fn=qwen_guard.async_check,
                oyster_stream_factory=oyster_stream_factory,
                final_audit_callback=final_audit_callback,
                pii_redact_fn=_pii_redact_fn,
            ),
            media_type="text/event-stream",
        )

    # 非流式：走完整 pipeline
    async def execute_task():
        t0 = time.monotonic()
        hits = scanner.scan(prompt)
        logger.info("SCAN req_id=%s hits=%d elapsed=%.3fs", req_id, len(hits), time.monotonic() - t0)
        if hits:
            names = ",".join(set([
                h["sources"][0].replace(".txt", "") for h in hits if h.get("sources")
            ]))
            logger.info("BLOCKED req_id=%s reason=keyword names=%s", req_id, names)
            return {
                "prompt": prompt,
                "response": f"涉及{names}内容，拒绝回答",
                "keyword_flagged": True,
                "protection_enabled": protection_enabled,
                "route": "blocked",
                "logs": [],
                "metadata": {"req_id": req_id},
            }

        t1 = time.monotonic()
        result = await run_usage_pipeline(data)
        pipeline_elapsed = time.monotonic() - t1

        if _pii_redact_fn and result.get("response"):
            result["response"] = _pii_redact_fn(result["response"])

        resp_text = result.get("response") or ""
        logger.info(
            "PIPELINE_DONE req_id=%s route=%s prompt_safe=%s resp_safe=%s "
            "pipeline_elapsed=%.3fs total_elapsed=%.3fs resp_chars=%d resp_preview=%.80s",
            req_id,
            result.get("route"),
            (result.get("prompt_moderation") or {}).get("safe"),
            (result.get("response_moderation") or {}).get("safe"),
            pipeline_elapsed,
            time.monotonic() - t_start,
            len(resp_text),
            resp_text,
        )

        result.setdefault("protection_enabled", protection_enabled)
        result.setdefault("metadata", {"req_id": req_id})
        return result

    try:
        async def _run_with_tracking():
            task = asyncio.create_task(execute_task())
            async with ACTIVE_TASKS_LOCK:
                ACTIVE_TASKS[req_id] = task
            try:
                return await task
            finally:
                async with ACTIVE_TASKS_LOCK:
                    ACTIVE_TASKS.pop(req_id, None)

        with fail_after(REQUEST_TIMEOUT):
            if _inflight is not None:
                async with _inflight:
                    return await _run_with_tracking()
            else:
                return await _run_with_tracking()

    except CancelledError:
        raise HTTPException(status_code=499, detail="Request cancelled")
    except Exception:
        logger.exception("INTERNAL_ERROR req_id=%s elapsed=%.3fs", req_id, time.monotonic() - t_start)
        raise HTTPException(status_code=500, detail="An internal server error occurred. Please contact admin.")


@app.post("/v1/stop")
async def stop_endpoint(payload: Dict[str, str] = Body(...)):
    rid = payload.get("request_id")
    async with ACTIVE_TASKS_LOCK:
        t = ACTIVE_TASKS.get(rid)
        if t and not t.done():
            t.cancel()
            return {"status": "ok"}
    return {"status": "not_found", "message": "Task not found or already finished"}
