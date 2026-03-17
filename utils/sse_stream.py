"""
SSE 门控流式生成器
完整漏斗审核版：关键词 → Qwen Guard → SafetyBERT
支持 Oyster 流式代答 + 流式结束后全量审核回调
"""
import json
import time
import uuid
import logging
import re
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from utils.tokenizer import get_tokenizer

logger = logging.getLogger("safeguard_system")

# 初始化 tokenizer
tok = get_tokenizer()

# 句末标点触发审核
_AUDIT_TRIGGER = re.compile(r"[。？！\n]")
# buffer 长度上限（未遇到标点时强制触发）
_BUFFER_MAX = 30

# <think> 标签相关常量
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_TAG_MAXLEN = max(len(_THINK_OPEN), len(_THINK_CLOSE))  # 8


async def _strip_think_blocks(source: AsyncIterator[str]) -> AsyncIterator[str]:
    """
    从流式 token 源中剥离 <think>...</think> 块，正确处理标签被分割在两个 chunk 之间的情况。

    算法：维护一个扫描缓冲区 buf，每次收到新 token 就拼入缓冲区，然后循环：
      - 在 think 块内：寻找 </think>；找到则退出块，继续扫描剩余；
        否则保留末尾 _THINK_TAG_MAXLEN-1 个字节（可能是部分标签），丢弃其余。
      - 在 think 块外：寻找 <think>；找到则 yield 标签前的安全文本，进入块，继续；
        否则 yield 安全前缀（末尾保留 _THINK_TAG_MAXLEN-1 个字节作为 carry），等待下一 token。
    """
    buf = ""
    in_think = False

    async for token in source:
        buf += token

        while True:
            if in_think:
                idx = buf.find(_THINK_CLOSE)
                if idx >= 0:
                    # 找到关闭标签，丢弃直到并包含 </think>，继续扫描
                    in_think = False
                    buf = buf[idx + len(_THINK_CLOSE):]
                else:
                    # 未找到，保留末尾可能的部分 </think>，丢弃其余（都在 think 块内）
                    keep = min(len(buf), _THINK_TAG_MAXLEN - 1)
                    buf = buf[-keep:] if keep else ""
                    break
            else:
                idx = buf.find(_THINK_OPEN)
                if idx >= 0:
                    # 找到开始标签，yield 标签前的安全文本，进入 think 块
                    if idx > 0:
                        yield buf[:idx]
                    in_think = True
                    buf = buf[idx + len(_THINK_OPEN):]
                    # 继续扫描剩余（可能紧跟 </think>）
                else:
                    # 无开始标签，yield 安全前缀，保留末尾 carry 防止跨 chunk 分割
                    keep = min(len(buf), _THINK_TAG_MAXLEN - 1)
                    safe = buf[:-keep] if keep else buf
                    buf = buf[-keep:] if keep else ""
                    if safe:
                        yield safe
                    break

    # 流结束：flush 剩余非 think 内容
    if not in_think and buf:
        yield buf


def generate_chunk_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:32]}"


def create_sse_chunk(
    chunk_id: str,
    model: str,
    content: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> str:
    delta = {"content": content} if content is not None else {}
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


async def _run_funnel(
    text: str,
    prompt: str,
    role: str,
    scanner_fn: Callable[[str], List[Dict[str, Any]]],
    hybrid_guard_fn: Callable[[List[Dict[str, str]]], Awaitable[Dict[str, Any]]],
    tok=None,
    window_size: int = 512,
    overlap: int = 128,
) -> tuple[bool, str]:
    """完整漏斗审核（修复超长 400 错误与 Token 乱码问题）"""
    hits = scanner_fn(text)
    if hits:
        return True, "敏感词"

    # 【关键修复 1】：把 safe_prompt 的生成提到最上面，让所有分支都能用到！
    # 限制 prompt 最长不超过 2000 字符，防止在任何情况下撑爆模型上下文
    safe_prompt = prompt[-2000:] if len(prompt) > 2000 else prompt

    # 如果没有 tokenizer，降级到简单截断
    if tok is None:
        safe_text = text[-2000:] if len(text) > 2000 else text
        if role == "assistant":
            messages = [
                {"role": "user", "content": safe_prompt},
                {"role": "assistant", "content": safe_text},
            ]
        else:
            messages = [{"role": "user", "content": safe_text}]
        
        res = await hybrid_guard_fn(messages)

        if res.get("status") == "error":
            logger.warning("Guard engine error: %s, treating as safe to avoid false positives", res.get("raw"))
            return False, ""

        if not res.get("safe", True):
            cats_list = res.get("categories") or ["违规"]
            return True, ",".join(cats_list)

        return False, ""

    # 使用滑动窗口扫描长文本
    tokens = tok.encode(text)
    if len(tokens) <= window_size:
        # 文本短于窗口，直接审核
        if role == "assistant":
            messages = [
                {"role": "user", "content": safe_prompt}, # 使用 safe_prompt
                {"role": "assistant", "content": text},
            ]
        else:
            messages = [{"role": "user", "content": text}]
            
        res = await hybrid_guard_fn(messages)

        if res.get("status") == "error":
            logger.warning("Guard engine error: %s, treating as safe to avoid false positives", res.get("raw"))
            return False, ""

        if not res.get("safe", True):
            cats_list = res.get("categories") or ["违规"]
            return True, ",".join(cats_list)

        return False, ""

    # 滑动窗口审核
    step = window_size - overlap
    for start_idx in range(0, len(tokens), step):
        end_idx = min(start_idx + window_size, len(tokens))
        window_tokens = tokens[start_idx:end_idx]
        
        # 【关键修复 2】：使用 errors="ignore" 防止半个中文字符解码成  引发安全模型误判
        window_text = tok.decode(window_tokens, errors="ignore") 

        if role == "assistant":
            messages = [
                {"role": "user", "content": safe_prompt}, # 修复原本使用超长 prompt 的 Bug
                {"role": "assistant", "content": window_text},
            ]
        else:
            messages = [{"role": "user", "content": window_text}]

        res = await hybrid_guard_fn(messages)

        if res.get("status") == "error":
            logger.warning("Guard engine error in sliding window: %s", res.get("raw"))
            continue

        if not res.get("safe", True):
            cats_list = res.get("categories") or ["违规"]
            return True, ",".join(cats_list)

    return False, ""


async def gated_streaming_generator(
    primary_stream_factory: Callable[[], Awaitable[AsyncIterator[str]]],
    model_name: str,
    request_id: str,
    prompt: str,
    scanner_fn: Callable[[str], List[Dict[str, Any]]],
    hybrid_guard_fn: Callable[[List[Dict[str, str]]], Awaitable[Dict[str, Any]]],
    oyster_stream_factory: Callable[[str], Awaitable[AsyncIterator[str]]],
    final_audit_callback: Optional[Callable[[str, str, Dict[str, Any]], Awaitable[None]]] = None,
    pii_redact_fn: Optional[Callable[[str], str]] = None,
) -> AsyncIterator[str]:
    """
    完整门控流式生成器。

    流程：
    1. prompt 漏斗审核（关键词 → HybridGuard）
       - 安全 → 调 primary 模型流式生成
       - 不安全 → 直接走 Oyster 流式代答
    2. 每个 buffer（句末标点或 ≥100 字符）做完整漏斗审核
       - 安全 → 释放 buffer 流式输出给客户端
       - 不安全 → 插入提示，切换 Oyster 流式代答；Oyster 也做漏斗审核
                   Oyster 若也不安全 → 直接终止
    3. 流式结束后，对完整拼接文本做一次全量审核，结果通过 final_audit_callback 回传

    Args:
        primary_stream_factory: 无参异步工厂，返回 primary 模型的 token 流
        model_name: 模型名（用于 SSE chunk 的 model 字段）
        request_id: 请求 ID（用于日志）
        prompt: 原始用户 prompt（用于审核上下文 + Oyster 代答输入）
        scanner_fn: 关键词扫描函数
        hybrid_guard_fn: HybridGuard 完整漏斗审核（Qwen+SafetyBERT）
        oyster_stream_factory: 接受 prompt 字符串，返回 Oyster token 流
        final_audit_callback: 流结束后的全量审核回调，签名：
            async (full_text: str, route: str, audit_result: dict) -> None
        pii_redact_fn: 可选的 PII 脱敏函数，签名：(text: str) -> str。
            在每个 buffer 进入安全漏斗之前调用，确保客户端收到的内容已脱敏。
    """
    chunk_id = generate_chunk_id()

    # ── 1. Prompt 漏斗审核 ─────────────────────────────────────────────
    prompt_blocked, prompt_reason = await _run_funnel(
        text=prompt,
        prompt=prompt,
        role="user",
        scanner_fn=scanner_fn,
        hybrid_guard_fn=hybrid_guard_fn,
        tok=tok,
    )

    if prompt_blocked:
        logger.info("[stream:%s] prompt 被拦截(%s)，切换 Oyster", request_id, prompt_reason)
        async for chunk in _oyster_stream(
            chunk_id, model_name, request_id, prompt,
            oyster_stream_factory, scanner_fn, hybrid_guard_fn,
            final_audit_callback, route="oyster_prompt_block",
            pii_redact_fn=pii_redact_fn,
        ):
            yield chunk
        return

    # ── 2. Primary 流式生成 + 滚动审核 ────────────────────────────────
    buffer = ""
    full_released = ""   # 已通过审核并输出给客户端的文本

    try:
        primary_stream = await primary_stream_factory()
        async for token in primary_stream:
            # primary 模型的 <think> 块原样透传给客户端（reasoning_content 已由 provider 包装）

            buffer += token

            is_punctuation = bool(_AUDIT_TRIGGER.search(token))
            is_buffer_full = len(buffer) >= _BUFFER_MAX

            # 【核心逻辑】：判断当前 buffer 最后一位是否是字母或数字
            # 如果是，说明可能正好从手机号、身份证号或英文单词的正中间被切断了！
            in_word_boundary = bool(buffer and buffer[-1].isalnum())

            # 如果没遇到句末标点，且（buffer没满，或者虽然满了但卡在敏感词中间），就继续攒字，不触发处理！
            if not is_punctuation and (not is_buffer_full or in_word_boundary):
                # 设立绝对上限防止撑爆内存
                if len(buffer) < 80:
                    continue

            # --- 下方只有当完整度得到保证时，才会触发 ---
            if pii_redact_fn:
                buffer = pii_redact_fn(buffer)

            # 【滑动窗口审核】携带最近 1500 字上下文，避免文本线性增长
            blocked, reason = await _run_funnel(
                text=(full_released + buffer)[-1500:],
                prompt=prompt,
                role="assistant",
                scanner_fn=scanner_fn,
                hybrid_guard_fn=hybrid_guard_fn,
                tok=tok,
            )

            if blocked:
                logger.info("[stream:%s] response buffer 被拦截(%s)，切换 Oyster", request_id, reason)
                # 插入切换提示
                notice = f"\n\n[提示：部分回答涉及敏感信息（{reason}），已拦截。以下由合规模型代答：]\n"
                yield create_sse_chunk(chunk_id, model_name, content=notice)

                async for chunk in _oyster_stream(
                    chunk_id, model_name, request_id, prompt,
                    oyster_stream_factory, scanner_fn, hybrid_guard_fn,
                    final_audit_callback, route="oyster_response_block",
                    prefix_released=full_released,
                    pii_redact_fn=pii_redact_fn,
                ):
                    yield chunk
                return

            # 审核通过，整块释放脱敏后的 buffer
            yield create_sse_chunk(chunk_id, model_name, content=buffer)
            full_released += buffer
            buffer = ""

        # 处理末尾残留 buffer（流结束时可能没遇到触发条件）
        if buffer:
            if pii_redact_fn:
                buffer = pii_redact_fn(buffer)
            blocked, reason = await _run_funnel(
                text=(full_released + buffer)[-1500:],
                prompt=prompt,
                role="assistant",
                scanner_fn=scanner_fn,
                hybrid_guard_fn=hybrid_guard_fn,
                tok=tok,
            )
            if blocked:
                notice = f"\n\n[提示：部分回答涉及敏感信息（{reason}），已拦截。以下由合规模型代答：]\n"
                yield create_sse_chunk(chunk_id, model_name, content=notice)
                async for chunk in _oyster_stream(
                    chunk_id, model_name, request_id, prompt,
                    oyster_stream_factory, scanner_fn, hybrid_guard_fn,
                    final_audit_callback, route="oyster_response_block",
                    prefix_released=full_released,
                    pii_redact_fn=pii_redact_fn,
                ):
                    yield chunk
                return
            yield create_sse_chunk(chunk_id, model_name, content=buffer)
            full_released += buffer

    except Exception as e:
        logger.error("[stream:%s] primary 流式异常: %s", request_id, e, exc_info=True)
        yield create_sse_chunk(chunk_id, model_name, content=f"\n[错误: {e}]")
        yield "data: [DONE]\n\n"
        return

    # ── 3. 流结束后全量审核 ────────────────────────────────────────────
    yield create_sse_chunk(chunk_id, model_name, finish_reason="stop")
    yield "data: [DONE]\n\n"

    await _final_audit(
        full_text=full_released,
        prompt=prompt,
        route="primary",
        scanner_fn=scanner_fn,
        hybrid_guard_fn=hybrid_guard_fn,
        callback=final_audit_callback,
        request_id=request_id,
    )


async def _oyster_stream(
    chunk_id: str,
    model_name: str,
    request_id: str,
    prompt: str,
    oyster_stream_factory: Callable[[str], Awaitable[AsyncIterator[str]]],
    scanner_fn: Callable[[str], List[Dict[str, Any]]],
    hybrid_guard_fn: Callable[[List[Dict[str, str]]], Awaitable[Dict[str, Any]]],
    final_audit_callback: Optional[Callable[[str, str, Dict[str, Any]], Awaitable[None]]],
    route: str,
    prefix_released: str = "",
    pii_redact_fn: Optional[Callable[[str], str]] = None,
) -> AsyncIterator[str]:
    """Oyster 流式代答，带完整漏斗审核。"""
    buffer = ""
    full_released = ""

    try:
        oyster_stream = await oyster_stream_factory(prompt)
        # _strip_think_blocks 健壮剥离 <think> 块（正确处理标签跨 chunk 分割）
        async for token in _strip_think_blocks(oyster_stream):
            buffer += token

            is_punctuation = bool(_AUDIT_TRIGGER.search(token))
            is_buffer_full = len(buffer) >= _BUFFER_MAX
            in_word_boundary = bool(buffer and buffer[-1].isalnum())

            if not is_punctuation and (not is_buffer_full or in_word_boundary):
                if len(buffer) < 80:
                    continue

            if pii_redact_fn:
                buffer = pii_redact_fn(buffer)

            # 【滑动窗口审核】携带最近 1500 字上下文
            blocked, reason = await _run_funnel(
                text=(full_released + buffer)[-1500:],
                prompt=prompt,
                role="assistant",
                scanner_fn=scanner_fn,
                hybrid_guard_fn=hybrid_guard_fn,
                tok=tok,
            )

            if blocked:
                logger.info("[stream:%s] Oyster 输出也被拦截(%s)，终止", request_id, reason)
                yield create_sse_chunk(chunk_id, model_name, content=f"\n\n[合规模型回答同样涉及违规内容，已终止输出]")
                yield create_sse_chunk(chunk_id, model_name, finish_reason="content_filter")
                yield "data: [DONE]\n\n"
                await _final_audit(
                    full_text=prefix_released,  # oyster 全部拦截，只记录 primary 已释放部分
                    prompt=prompt,
                    route=route + "_oyster_blocked",
                    scanner_fn=scanner_fn,
                    hybrid_guard_fn=hybrid_guard_fn,
                    callback=final_audit_callback,
                    request_id=request_id,
                )
                return

            yield create_sse_chunk(chunk_id, model_name, content=buffer)
            full_released += buffer
            buffer = ""

        # 末尾残留
        if buffer:
            if pii_redact_fn:
                buffer = pii_redact_fn(buffer)
            blocked, reason = await _run_funnel(
                text=(full_released + buffer)[-1500:],
                prompt=prompt,
                role="assistant",
                scanner_fn=scanner_fn,
                hybrid_guard_fn=hybrid_guard_fn,
                tok=tok,
            )
            if blocked:
                yield create_sse_chunk(chunk_id, model_name, content=f"\n\n[合规模型回答同样涉及违规内容，已终止输出]")
                yield create_sse_chunk(chunk_id, model_name, finish_reason="content_filter")
                yield "data: [DONE]\n\n"
                await _final_audit(
                    full_text=prefix_released,
                    prompt=prompt,
                    route=route + "_oyster_blocked",
                    scanner_fn=scanner_fn,
                    hybrid_guard_fn=hybrid_guard_fn,
                    callback=final_audit_callback,
                    request_id=request_id,
                )
                return
            yield create_sse_chunk(chunk_id, model_name, content=buffer)
            full_released += buffer

    except Exception as e:
        logger.error("[stream:%s] Oyster 流式异常: %s", request_id, e, exc_info=True)
        yield create_sse_chunk(chunk_id, model_name, content=f"\n[Oyster错误: {e}]")
        yield "data: [DONE]\n\n"
        return

    yield create_sse_chunk(chunk_id, model_name, finish_reason="stop")
    yield "data: [DONE]\n\n"

    # Oyster 全量审核
    await _final_audit(
        full_text=prefix_released + full_released,
        prompt=prompt,
        route=route,
        scanner_fn=scanner_fn,
        hybrid_guard_fn=hybrid_guard_fn,
        callback=final_audit_callback,
        request_id=request_id,
    )


async def _final_audit(
    full_text: str,
    prompt: str,
    route: str,
    scanner_fn: Callable[[str], List[Dict[str, Any]]],
    hybrid_guard_fn: Callable[[List[Dict[str, str]]], Awaitable[Dict[str, Any]]],
    callback: Optional[Callable[[str, str, Dict[str, Any]], Awaitable[None]]],
    request_id: str,
) -> None:
    """流结束后对完整文本做全量漏斗审核，结果通过 callback 回传。"""
    if not full_text:
        return
    try:
        hits = scanner_fn(full_text)
        if hits:
            audit_result = {"safe": False, "categories": ["keyword"], "keyword_hits": hits}
        else:
            # 使用滑动窗口审核，确保完整覆盖长文本
            audit_result = await _run_funnel(
                text=full_text,
                prompt=prompt,
                role="assistant",
                scanner_fn=lambda x: [],  # 关键词已在上面检查过，这里跳过
                hybrid_guard_fn=hybrid_guard_fn,
                tok=tok,
            )
            # 将 _run_funnel 的返回值转换为审核结果格式
            if audit_result[0]:  # is_blocked
                audit_result = {"safe": False, "categories": [audit_result[1]]}
            else:
                audit_result = {"safe": True, "categories": []}

        logger.info(
            "[stream:%s] 全量审核完成 route=%s safe=%s cats=%s",
            request_id, route, audit_result.get("safe"), audit_result.get("categories"),
        )

        if callback:
            await callback(full_text, route, audit_result)

    except Exception as e:
        logger.error("[stream:%s] 全量审核异常: %s", request_id, e, exc_info=True)

