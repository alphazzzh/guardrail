"""
插件式安全防护服务逻辑
"""
import logging
from typing import Any, Dict, List, Optional
from utils.helpers import ensure_text, coerce_bool
from config.settings import PRIMARY_MODEL_BASE_URL, PRIMARY_MODEL_NAME, PRIMARY_DEFAULT_MAX_TOKENS

from workflows.usage_workflow import (
    keyword_scan_node,
    prompt_guard_node,
    oyster_generation_node,
    response_guard_node,
    call_primary_llm_with_provider,
    _split_thinking_and_answer,
    usage_app,
    UsageState,
)

logger = logging.getLogger("safeguard_system")


async def check_input_safety(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = ensure_text(payload.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt 不能为空")
    state: UsageState = {"prompt": prompt, "logs": []}
    state.update(await keyword_scan_node(state))
    state.update(await prompt_guard_node(state))
    return {
        "safe": state.get("prompt_safe", False),
        "route": state.get("route", "oyster"),
        "keyword_flagged": state.get("keyword_flagged", False),
        "keyword_hits": state.get("keyword_hits", []),
        "moderation": state.get("prompt_moderation", {}),
        "logs": state.get("logs", []),
    }


async def generate_safe_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = ensure_text(payload.get("prompt", "")).strip()
    state: UsageState = {"prompt": prompt, "logs": []}
    state.update(await oyster_generation_node(state))
    return {"response": state.get("response", ""), "logs": state.get("logs", [])}


async def check_output_safety(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = ensure_text(payload.get("prompt", ""))
    response = ensure_text(payload.get("response", "")).strip()
    if not response:
        raise ValueError("response 不能为空")
    state: UsageState = {"prompt": prompt, "response": response, "logs": []}
    state.update(await response_guard_node(state))
    moderation = state.get("response_moderation", {})
    is_safe = moderation.get("safe", True)
    return {
        "safe": is_safe,
        "action": "pass" if is_safe else "block",
        "moderation": moderation,
        "logs": state.get("logs", []),
    }


async def run_usage_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = ensure_text(payload.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt 不能为空")
    protection_enabled = coerce_bool(payload.get("protection_enabled"), True)
    history = payload.get("history") or []
    model_params = payload.get("model_params") or {}
    # 若调用方未指定 max_tokens，使用配置默认值，避免模型服务端默认值（通常 512）截断长推理
    if "max_tokens" not in model_params:
        model_params = {**model_params, "max_tokens": PRIMARY_DEFAULT_MAX_TOKENS}

    init_state: UsageState = {
        "prompt": prompt,
        "history": history,
        "logs": [],
        "provider": payload.get("provider", "openai"),
        "primary_model_name": payload.get("primary_model_name") or PRIMARY_MODEL_NAME,
        "model_params": model_params,
        # provider_config 由调用方（main.py / api_service）传入，不能丢弃，
        # 否则 _get_provider 无法获取 Azure endpoint、Bedrock region 等必要配置
        "provider_config": payload.get("provider_config") or {},
    }

    if not protection_enabled:
        full_response = await call_primary_llm_with_provider(init_state)
        thinking, answer = _split_thinking_and_answer(full_response)
        return {
            "response": answer,
            "reasoning_content": thinking,
            "prompt_moderation": {"status": "skipped", "safe": True},
        }

    result = await usage_app.ainvoke(init_state)

    final_result = {
        "prompt": prompt,
        "history": result.get("history", history),
        "route": result.get("route"),
        "prompt_moderation": result.get("prompt_moderation"),
        "response": result.get("response", ""),
        "reasoning_content": result.get("reasoning_content"),
        "response_moderation": result.get("response_moderation"),
        "keyword_flagged": result.get("keyword_flagged", False),
        "logs": result.get("logs", []),
    }
    logger.info("Pipeline done | route=%s", final_result["route"])
    return final_result


async def run_testing_pipeline(payload: Dict[str, Any]):
    from workflows.testing_workflow import test_app, TestingState
    state: TestingState = {
        "dataset_path": payload.get("dataset_path", ""),
        "limit_per_category": int(payload.get("limit_per_category", 50)),
        "category_field": payload.get("category_field", "category"),
        "prompt_field": payload.get("prompt_field", "prompt"),
        "logs": [],
    }
    result = await test_app.ainvoke(state)
    return {
        "total_prompt_score": result.get("total_prompt_score", 0.0),
        "prompt_scores": result.get("prompt_scores", {}),
        "logs": result.get("logs", []),
    }
