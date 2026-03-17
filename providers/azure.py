"""Azure OpenAI Provider 实现"""
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from openai import AzureOpenAI, AsyncAzureOpenAI

from .base import BaseProvider, ProviderFactory

logger = logging.getLogger("safeguard_system")


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI Provider"""

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str = "2024-02-15-preview",
        http_client=None,
        # 修复问题 E：显式接受 async_http_client，使共享连接池生效
        async_http_client=None,
        **kwargs,
    ):
        super().__init__(api_key, azure_endpoint, **kwargs)
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            http_client=http_client,
        )
        self.async_client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            # async_http_client 优先；未提供则回退到同步 http_client
            http_client=async_http_client if async_http_client is not None else http_client,
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
        params = self._prepare_params(messages, model, max_tokens, temperature, stream, **kwargs)
        response = self.client.chat.completions.create(**params)
        if stream:
            return (
                chunk.choices[0].delta.content
                for chunk in response
                if chunk.choices and chunk.choices[0].delta.content
            )
        return response.choices[0].message.content or ""

    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        params = self._prepare_params(messages, model, max_tokens, temperature, stream, **kwargs)
        response = await self.async_client.chat.completions.create(**params)
        if stream:
            async def gen():
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            return gen()
        return response.choices[0].message.content or ""

    def _prepare_params(self, messages, model, max_tokens, temperature, stream, **kwargs):
        params = {"model": model, "messages": messages, "stream": stream}
        # 修复：用 is not None 判断，避免将合法的 0 / 0.0 当作 falsy 过滤掉
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        params.update({k: v for k, v in kwargs.items() if k != "enable_thinking"})
        return params

    def get_provider_name(self) -> str:
        return "azure"

# 修复问题 J：注册统一由 providers/__init__.py 负责，此处删除重复注册
