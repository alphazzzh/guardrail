"""使用路径工作流 (异步漏斗审核版)"""
import logging
import random
import re
import time
import asyncio
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, TypedDict

import httpx
from langgraph.graph import END, StateGraph
from openai import APIStatusError, APITimeoutError, OpenAI, AsyncOpenAI

from config import (
    GUARD_CHUNK_OVERLAP_TOKENS,
    GUARD_MAX_CHUNK_TOKENS,
    MODEL_OYSTER,
    OYSTER_BASE_URL,
    PRIMARY_BACKOFF_BASE,
    PRIMARY_CONTEXT_TOKEN_LIMIT,
    PRIMARY_MAX_RETRIES,
    QWEN3GUARD_BASE_URL,
    RESPONSE_GUARD_MAX_WORKERS,
    MODEL_MODERATION,
    OPENAI_API_KEY,
    PRIMARY_MODEL_BASE_URL,
    PRIMARY_MODEL_NAME,
    SENSITIVE_WORD_PATHS,
    PROVIDER_CACHE_MAX_SIZE,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_TIMEOUT,
    CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS,
    GUARD_CONNECT_TIMEOUT,
    GUARD_READ_TIMEOUT,
    GUARD_MAX_CONNECTIONS,
    GUARD_MAX_KEEPALIVE,
    OYSTER_CONNECT_TIMEOUT,
    OYSTER_READ_TIMEOUT,
    OYSTER_MAX_CONNECTIONS,
    OYSTER_MAX_KEEPALIVE,
    PRIMARY_CONNECT_TIMEOUT,
    PRIMARY_READ_TIMEOUT,
    PRIMARY_MAX_CONNECTIONS,
    PRIMARY_MAX_KEEPALIVE,
)
from guards import QwenGuard, SensitiveScanner
from guards.safetybert_guard import SafetyBERTGuard
from guards.hybrid_guard import HybridGuard
from providers import ProviderFactory
from utils.helpers import ensure_text
from utils.http_client import get_shared_http_client, get_shared_async_http_client
from utils.tokenizer import get_tokenizer
from utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpen

logger = logging.getLogger("safeguard_system")

# 初始化全局资源
tok = get_tokenizer()


async def _sliding_window_guard(
    text: str,
    prompt: str,
    guard_fn,
    window_size: int = GUARD_MAX_CHUNK_TOKENS,
    overlap: int = GUARD_CHUNK_OVERLAP_TOKENS,
) -> Dict[str, Any]:
    """
    滑动窗口审核：将长文本分割成 window_size 大小的块，步长为 (window_size - overlap)。
    若任何一个窗口被拦截，立即返回拦截结果。

    Args:
        text: 要审核的文本
        prompt: 原始 prompt（用于上下文）
        guard_fn: 审核函数，签名：async (messages: List[Dict]) -> Dict
        window_size: 窗口大小（token 数）
        overlap: 重叠大小（token 数）

    Returns:
        审核结果字典，若任何窗口被拦截则返回 {"safe": False, ...}
    """
    if not text:
        return {"safe": True, "categories": []}

    # 分词
    tokens = tok.encode(text)
    if len(tokens) <= window_size:
        # 文本短于窗口，直接审核
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
        ]
        return await guard_fn(messages)

    # 滑动窗口审核
    step = window_size - overlap
    for start_idx in range(0, len(tokens), step):
        end_idx = min(start_idx + window_size, len(tokens))
        window_tokens = tokens[start_idx:end_idx]
        window_text = tok.decode(window_tokens)

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": window_text},
        ]
        result = await guard_fn(messages)

        # 如果审核引擎出错，记录但继续（不拦截）
        if result.get("status") == "error":
            logger.warning("Guard engine error in sliding window: %s", result.get("raw"))
            continue

        # 若任何窗口被拦截，立即返回
        if not result.get("safe", True):
            return result

    # 所有窗口都通过
    return {"safe": True, "categories": []}

# 客户端初始化 (同步与异步)
shared_async_http = get_shared_async_http_client(PRIMARY_CONNECT_TIMEOUT, PRIMARY_READ_TIMEOUT, PRIMARY_MAX_CONNECTIONS, PRIMARY_MAX_KEEPALIVE)
guard_async_http = get_shared_async_http_client(GUARD_CONNECT_TIMEOUT, GUARD_READ_TIMEOUT, GUARD_MAX_CONNECTIONS, GUARD_MAX_KEEPALIVE)
oyster_async_http = get_shared_async_http_client(OYSTER_CONNECT_TIMEOUT, OYSTER_READ_TIMEOUT, OYSTER_MAX_CONNECTIONS, OYSTER_MAX_KEEPALIVE)

async_client_qwen3guard = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=QWEN3GUARD_BASE_URL, http_client=guard_async_http)
async_client_oyster = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OYSTER_BASE_URL, http_client=oyster_async_http)

# 初始化 Guard 组件
_base_qwen_guard = QwenGuard(OpenAI(api_key="EMPTY", base_url=QWEN3GUARD_BASE_URL), MODEL_MODERATION, async_client=async_client_qwen3guard)
_safety_guard = SafetyBERTGuard()
# HybridGuard 现在是漏斗模式：Qwen 发现 unsafe 才进 BERT
qwen_guard = HybridGuard(_base_qwen_guard, _safety_guard)
scanner = SensitiveScanner(SENSITIVE_WORD_PATHS)

primary_circuit_breaker = CircuitBreaker(name="primary_llm")
_provider_cache = OrderedDict()

# ================== 状态定义 ==================
class UsageState(TypedDict, total=False):
    prompt: str
    history: Optional[List[Dict[str, str]]]
    client_ip: Optional[str]
    request_id: Optional[str]
    keyword_flagged: bool
    keyword_hits: List[Dict[str, Any]]
    prompt_moderation: Dict[str, Any]
    prompt_safe: bool
    route: Literal["primary", "oyster", "reject"]
    response: str
    reasoning_content: Optional[str]
    response_moderation: Dict[str, Any]
    logs: List[str]
    provider: Optional[str]
    provider_config: Optional[Dict[str, Any]]
    model_params: Optional[Dict[str, Any]]
    primary_model_name: Optional[str]

# ================== 工具函数 ==================
def _split_thinking_and_answer(content: str) -> tuple[str, str]:
    if not content: return "", ""
    thinking = "\n\n".join(re.findall(r"<think>(.*?)</think[^>]*?>", content, flags=re.S | re.I)).strip()
    answer = re.sub(r"<think>.*?</think[^>]*?>", "", content, flags=re.S | re.I).strip()
    return thinking, answer

def _get_provider(state: UsageState):
    p_name = state.get("provider", "openai").lower()
    config = state.get("provider_config") or {}
    key = f"{p_name}:{hash(str(config))}"
    logger.info("GET_PROVIDER p_name=%s config=%s key=%s cache_hit=%s", p_name, config, key, key in _provider_cache)
    if key in _provider_cache:
        logger.info("PROVIDER_CACHE_HIT key=%s", key)
        return _provider_cache[key]
    logger.info("PROVIDER_CACHE_MISS creating new provider")
    provider = ProviderFactory.create(p_name, api_key=OPENAI_API_KEY, async_http_client=shared_async_http, **config)
    # LRU 缓存逻辑：如果超过最大容量，删除最早的条目 (问题 9 修复)
    if len(_provider_cache) >= PROVIDER_CACHE_MAX_SIZE:
        _provider_cache.popitem(last=False)

    _provider_cache[key] = provider
    return provider

async def call_primary_llm_with_provider(state: UsageState) -> str:
    provider = _get_provider(state)
    messages = [{"role": m["role"], "content": m["content"]} for m in state.get("history", [])]
    messages.append({"role": "user", "content": state.get("prompt", "")})

    async def _do_call():
        return await provider.async_chat_completion(
            messages=messages, 
            model=state.get("primary_model_name") or PRIMARY_MODEL_NAME,
            ** (state.get("model_params") or {})
        )
    
    try:
        return await primary_circuit_breaker.async_call(_do_call)
    except Exception as e:
        logger.error(f"Primary LLM Call Error: {e}")
        return ""

# ================== 工作流节点 ==================
async def keyword_scan_node(state: UsageState) -> UsageState:
    hits = scanner.scan(state.get("prompt", ""))
    logs = list(state.get("logs") or [])
    if hits:
        words = ", ".join(h["word"] for h in hits[:5])
        logs.append(f"[keyword_scan] 命中敏感词: {words}")
    else:
        logs.append("[keyword_scan] 无敏感词命中")
    return {"keyword_flagged": bool(hits), "keyword_hits": hits, "logs": logs}

async def prompt_guard_node(state: UsageState) -> UsageState:
    logs = list(state.get("logs") or [])
    if state.get("keyword_flagged"):
        logs.append("[prompt_guard] 关键词拦截，路由至 reject")
        return {"prompt_safe": False, "route": "reject", "response": "敏感词拦截", "logs": logs}

    messages = [{"role": "user", "content": state.get("prompt", "")}]
    res = await qwen_guard.async_check(messages)

    # 如果审核引擎本身出错（status="error"），不应拦截，默认认为安全
    if res.get("status") == "error":
        logger.warning("[prompt_guard] 审核引擎出错: %s，默认认为安全", res.get("raw"))
        logs.append(f"[prompt_guard] 审核引擎出错（{res.get('raw')}），默认认为安全")
        return {"prompt_moderation": res, "prompt_safe": True, "route": "primary", "logs": logs}

    route = "primary" if res.get("safe", True) else "oyster"
    logs.append(f"[prompt_guard] safe={res.get('safe')} route={route} cats={res.get('categories', [])}")
    return {"prompt_moderation": res, "prompt_safe": res.get("safe", True), "route": route, "logs": logs}

async def primary_generation_node(state: UsageState) -> UsageState:
    logs = list(state.get("logs") or [])
    full_text = await call_primary_llm_with_provider(state)
    thinking, answer = _split_thinking_and_answer(full_text)
    logs.append(f"[primary_gen] 生成完成，answer_len={len(answer)}")
    return {"response": answer, "reasoning_content": thinking, "logs": logs}

async def oyster_generation_node(state: UsageState) -> UsageState:
    """异步安全代答 (Oyster)"""
    try:
        logger.info("Triggering Oyster safety response...")
        resp = await async_client_oyster.chat.completions.create(
            model=MODEL_OYSTER,
            messages=[{"role": "user", "content": state.get("prompt", "")[-2000:]}]
        )
        content = resp.choices[0].message.content or ""
        _, answer = _split_thinking_and_answer(content)
        logs = list(state.get("logs") or [])
        logs.append(f"[oyster_gen] 安全代答完成，answer_len={len(answer)}")
        return {"response": answer, "logs": logs}
    except Exception as e:
        logger.error(f"Oyster async failure: {e}")
        logs = list(state.get("logs") or [])
        logs.append(f"[oyster_gen] 调用失败: {e}")
        return {"response": "对不起，根据安全策略，我无法回答这个问题。", "logs": logs}

async def response_guard_node(state: UsageState) -> UsageState:
    logs = list(state.get("logs") or [])
    if state.get("route") in ["reject", "oyster"]:
        logs.append("[response_guard] 跳过（route=reject/oyster）")
        return {"response_moderation": {"safe": True, "status": "skip"}, "logs": logs}

    # 【滑动窗口审核】使用真正的滑动窗口，避免超长文本导致 400 错误
    response_text = state.get("response", "")
    prompt_text = state.get("prompt", "")

    res = await _sliding_window_guard(
        text=response_text,
        prompt=prompt_text,
        guard_fn=qwen_guard.async_check,
        window_size=GUARD_MAX_CHUNK_TOKENS,
        overlap=GUARD_CHUNK_OVERLAP_TOKENS,
    )

    # 如果审核引擎本身出错（status="error"），不应拦截，记录日志后继续
    if res.get("status") == "error":
        logger.warning("[response_guard] 审核引擎出错: %s，跳过拦截", res.get("raw"))
        logs.append(f"[response_guard] 审核引擎出错（{res.get('raw')}），跳过拦截")
        return {"response_moderation": res, "logs": logs}

    if not res.get("safe", True):
        logs.append(f"[response_guard] 输出违规已拦截 cats={res.get('categories', [])}")
        return {"response_moderation": res, "response": "内容违规已拦截", "reasoning_content": None, "logs": logs}

    logs.append("[response_guard] 输出审核通过")
    return {"response_moderation": res, "logs": logs}

# 构建图
builder = StateGraph(UsageState)
builder.add_node("keyword_scan", keyword_scan_node)
builder.add_node("prompt_guard", prompt_guard_node)
builder.add_node("primary_gen", primary_generation_node)
builder.add_node("oyster_gen", oyster_generation_node)
builder.add_node("response_guard", response_guard_node)

builder.set_entry_point("keyword_scan")
builder.add_edge("keyword_scan", "prompt_guard")
builder.add_conditional_edges("prompt_guard", lambda x: x.get("route"), {
    "primary": "primary_gen", "oyster": "oyster_gen", "reject": "response_guard"
})
builder.add_edge("primary_gen", "response_guard")
builder.add_edge("oyster_gen", "response_guard")
builder.add_edge("response_guard", END)

usage_app = builder.compile()
