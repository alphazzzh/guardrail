"""Anthropic Messages API Provider 实现"""
import logging
from typing import Any, Dict, Iterator, List, Optional, Union

from .base import BaseProvider, ProviderFactory

logger = logging.getLogger("safeguard_system")


class AnthropicProvider(BaseProvider):
    """Anthropic Messages API Provider"""

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install it with: pip install anthropic"
            )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """调用 Anthropic Messages API"""
        # Anthropic 需要将 system 消息分离
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                anthropic_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

        params: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 1024,
        }

        if system_message:
            params["system"] = system_message
        if temperature is not None:
            params["temperature"] = temperature

        params.update(kwargs)

        try:
            if stream:
                return self._stream_completion(params)
            else:
                response = self.client.messages.create(**params)
                # Anthropic 返回的 content 是一个列表
                return "".join([block.text for block in response.content if hasattr(block, "text")])

        except Exception as e:
            logger.error(f"Anthropic API Error: {e}")
            raise

    def _stream_completion(self, params: Dict[str, Any]) -> Iterator[str]:
        """处理流式响应"""
        params["stream"] = True
        try:
            with self.client.messages.stream(**params) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic stream error: {e}")
            raise

    def get_provider_name(self) -> str:
        return "anthropic"


# 修复问题 J：注册统一由 providers/__init__.py 负责，此处删除重复注册
