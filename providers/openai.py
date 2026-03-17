"""OpenAI Provider 实现"""
import json
import logging
from typing import Any, Dict, Iterator, List, Optional, Union

from openai import APIStatusError, APITimeoutError, OpenAI, AsyncOpenAI

from .base import BaseProvider, ProviderFactory

logger = logging.getLogger("safeguard_system")

_THINK_TAGS = ("<think>", "</think>")


def _content_safe_len(text: str) -> int:
    """
    返回 text 中可以安全 yield 的前缀长度。

    从头扫描：若某个后缀 text[i:] 是任意 think 标签的真前缀（即标签尚未完整），
    则 text[:i] 是安全的，text[i:] 需作为 carry 留到下一个 chunk 再判断。
    若所有位置都安全，返回 len(text)。

    示例：
      "hello <thi"  → 6  (carry "<thi" 可能是 <think> 的开头)
      "hello <think>" → 14  (完整标签，全部安全)
      "normal text" → 11  (无标签，全部安全)
    """
    for i in range(len(text)):
        suffix = text[i:]
        if any(tag.startswith(suffix) and suffix != tag for tag in _THINK_TAGS):
            return i
    return len(text)


class OpenAIProvider(BaseProvider):
    """OpenAI Chat Completions API Provider (支持 SSE streaming)"""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        http_client=None,
        # 显式接受 async_http_client，使共享连接池生效
        async_http_client=None,
        **kwargs,
    ):
        super().__init__(api_key, base_url, **kwargs)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            # async_http_client 优先；未提供则回退到同步 http_client（兼容旧用法）
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
        """同步调用 OpenAI Chat Completions API"""
        params = self._prepare_params(messages, model, max_tokens, temperature, stream, **kwargs)

        try:
            response = self.client.chat.completions.create(**params)

            if stream:
                return self._stream_response(response)
            else:
                return self._parse_message(response.choices[0].message)

        except (APITimeoutError, APIStatusError) as e:
            logger.error(f"OpenAI API Error: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI unexpected error: {e}")
            raise

    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, "AsyncIterator[str]"]:
        """异步调用 OpenAI Chat Completions API"""
        params = self._prepare_params(messages, model, max_tokens, temperature, stream, **kwargs)

        try:
            response = await self.async_client.chat.completions.create(**params)

            if stream:
                return self._async_stream_response(response)
            else:
                return self._parse_message(response.choices[0].message)

        except (APITimeoutError, APIStatusError) as e:
            logger.error(f"OpenAI Async API Error: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI Async unexpected error: {e}")
            raise

    def _prepare_params(self, messages, model, max_tokens, temperature, stream, **kwargs):
        params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature

        incompatible_keys = {"enable_thinking"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in incompatible_keys}
        params.update(filtered_kwargs)
        params["stream"] = stream
        return params

    def _parse_message(self, msg):
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", "")
        if reasoning:
            return f"<think>\n{reasoning}\n</think>\n{content}"
        return content

    def _stream_response(self, response) -> Iterator[str]:
        """处理流式响应，保留 reasoning_content 作为 <think> 块"""
        in_reasoning = False
        content_buf = ""
        try:
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # 【修改这里】：兼容主流的所有思考字段命名规范
                reasoning = (
                    getattr(delta, "reasoning_content", None) or
                    getattr(delta, "reasoning", None) or
                    getattr(delta, "thinking", None)
                )
                content = getattr(delta, "content", None)

                if reasoning:
                    if content_buf:
                        yield content_buf
                        content_buf = ""
                    if not in_reasoning:
                        yield "<think>"
                        in_reasoning = True
                    yield reasoning

                if content:
                    if in_reasoning:
                        yield "</think>"
                        in_reasoning = False
                        content_buf = ""
                    content_buf += content
                    safe_len = _content_safe_len(content_buf)
                    if safe_len:
                        yield content_buf[:safe_len]
                        content_buf = content_buf[safe_len:]
        except GeneratorExit:
            # 同步生成器收到关闭信号，直接退出
            logger.info("客户端连接断开，停止同步流式输出")
            return
        except Exception as e:
            logger.error(f"Stream error: {e}")
            raise

        # 兜底逻辑移出 finally 块
        if in_reasoning:
            yield "</think>"
        if content_buf:
            yield content_buf

    async def _async_stream_response(self, response) -> "AsyncIterator[str]":
        """处理异步流式响应，保留 reasoning_content 作为 <think> 块"""
        in_reasoning = False
        content_buf = ""
        try:
            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # 【修改这里】：兼容主流的所有思考字段命名规范
                reasoning = (
                    getattr(delta, "reasoning_content", None) or
                    getattr(delta, "reasoning", None) or
                    getattr(delta, "thinking", None)
                )
                content = getattr(delta, "content", None)

                if reasoning:
                    if content_buf:
                        yield content_buf
                        content_buf = ""
                    if not in_reasoning:
                        yield "<think>"
                        in_reasoning = True
                    yield reasoning

                if content:
                    if in_reasoning:
                        yield "</think>"
                        in_reasoning = False
                        content_buf = ""
                    content_buf += content
                    safe_len = _content_safe_len(content_buf)
                    if safe_len:
                        yield content_buf[:safe_len]
                        content_buf = content_buf[safe_len:]
        except GeneratorExit:
            # 异步生成器收到关闭信号，直接退出
            logger.info("客户端连接断开，停止异步流式输出")
            return
        except Exception as e:
            logger.error(f"Async stream error: {e}")
            raise

        # 兜底逻辑移出 finally 块
        if in_reasoning:
            yield "</think>"
        if content_buf:
            yield content_buf

    def get_provider_name(self) -> str:
        return "openai"



