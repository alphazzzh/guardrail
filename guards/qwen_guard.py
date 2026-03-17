"""Qwen3Guard 审核功能"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI, AsyncOpenAI
from utils.helpers import ensure_text

logger = logging.getLogger("safeguard_system")

# Qwen3Guard 输出的九类风险标签
QWEN_CATEGORY_CANONICAL = [
    "Violent", "Non-violent Illegal Acts", "Sexual Content or Sexual Acts",
    "Personally Identifiable Information", "Suicide & Self-Harm", "Unethical Acts",
    "Politically Sensitive Topics", "Copyright Violation", "Jailbreak",
]

class QwenGuard:
    """Qwen3Guard 审核封装"""

    def __init__(self, client: OpenAI, model: str, async_client: Optional[AsyncOpenAI] = None):
        self.client = client
        self.async_client = async_client
        self.model = model

    def check(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """同步审核"""
        try:
            response = self.client.chat.completions.create(model=self.model, messages=messages)
            raw = ensure_text(response.choices[0].message.content).strip()
            return self._process_response(raw)
        except Exception as e:
            return self._handle_error(e)

    async def async_check(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """异步审核"""
        if not self.async_client:
            return self.check(messages)
        try:
            response = await self.async_client.chat.completions.create(model=self.model, messages=messages)
            raw = ensure_text(response.choices[0].message.content).strip()
            return self._process_response(raw)
        except Exception as e:
            return self._handle_error(e)

    def _process_response(self, raw: str) -> Dict[str, Any]:
        parsed = None
        try: parsed = json.loads(raw)
        except Exception: pass
        
        status = self._guard_status_from_text(raw, parsed)
        categories = self._extract_guard_categories(parsed, raw)
        return {
            "status": status,
            "safe": status == "safe",
            "raw": raw,
            "categories": categories,
            "normalized_categories": [c.lower() for c in categories],
        }

    def _handle_error(self, e: Exception) -> Dict[str, Any]:
        logger.error("Guard API Error: %s", e)
        # 技术错误（如超长、网络问题）不应被误判为违规
        # status="error" 表示审核引擎本身出问题，不是内容违规
        return {"status": "error", "safe": True, "raw": str(e), "categories": []}

    @staticmethod
    def _guard_status_from_text(raw: str, parsed: Optional[dict]) -> str:
        """从响应文本中提取审核状态。

        优先级：
        1. 解析后的 JSON 中的 status/result/label 字段
        2. 如果 JSON 解析失败，才进行文本匹配
        3. 文本匹配时使用词边界，避免误伤（如"unsafe"作为代码注释）
        """
        if isinstance(parsed, dict):
            for k in ("status", "result", "label"):
                v = parsed.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip().lower()

        # 只有在 JSON 解析失败时才进行文本匹配
        # 使用词边界匹配，避免在代码/文本中的"unsafe"被误判
        s = raw.strip().lower()

        # 检查是否包含明确的拒绝关键词（作为独立词汇，不是子串）
        if re.search(r'\b(unsafe|block|reject)\b', s):
            return "unsafe"

        return "safe"

    @staticmethod
    def _extract_guard_categories(parsed: Optional[dict], raw: str) -> List[str]:
        """从响应中提取违规类别。

        优先级：
        1. 解析后的 JSON 中的 categories/category/labels 字段
        2. 如果 JSON 解析失败，才进行文本匹配
        3. 文本匹配时使用词边界，避免误伤
        """
        cats = []

        # 优先从 JSON 中提取
        if isinstance(parsed, dict):
            for k in ("categories", "category", "labels"):
                v = parsed.get(k)
                if isinstance(v, list):
                    cats.extend([str(x) for x in v])
                elif v:
                    cats.append(str(v))

        # 只有在 JSON 中没有找到类别时，才进行文本匹配
        if not cats:
            for c in QWEN_CATEGORY_CANONICAL:
                # 使用词边界匹配，避免子串误伤
                if re.search(r'\b' + re.escape(c.lower()) + r'\b', raw.lower()):
                    cats.append(c)

        return list(set(cats))
