"""HTTP 转发 Provider - 将请求转发到外部服务"""
import logging
import random
import time
import json
import asyncio
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import httpx

from .base import BaseProvider

logger = logging.getLogger("safeguard_system")

# 默认重试配置
DEFAULT_MAX_RETRIES = 2
DEFAULT_BACKOFF_BASE = 0.5

_THINK_TAGS = ("<think>", "</think>")


def _content_safe_len(text: str) -> int:
    """返回 text 中可以安全 yield 的前缀长度，防止 <think>/<</think> 标签被分割在两个 yield 之间。"""
    for i in range(len(text)):
        suffix = text[i:]
        if any(tag.startswith(suffix) and suffix != tag for tag in _THINK_TAGS):
            return i
    return len(text)


class HTTPForwardProvider(BaseProvider):
    """
    HTTP 转发 Provider

    将请求原封不动地转发到外部 HTTP 服务，支持透传所有参数。
    适用于对接已有的完整服务（包含 thinking、RAG 等复杂逻辑）。
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        endpoint: str = "/chat/completions",
        timeout: float = 120.0,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        **kwargs
    ):
        super().__init__(api_key, base_url, **kwargs)
        self.endpoint = endpoint
        self.timeout = timeout
        self.http_client = http_client or httpx.Client(timeout=timeout)
        self.async_http_client = async_http_client or httpx.AsyncClient(timeout=timeout)
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """转发请求到外部服务（同步版）"""
        if stream:
            raise NotImplementedError("HTTP Forward currently does not support synchronous streaming")

        url, payload, headers = self._prepare_request(messages, model, max_tokens, temperature, **kwargs)
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.http_client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return self._parse_response(response.json())
            except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as e:
                last_exc = e
                if isinstance(e, httpx.HTTPStatusError) and 400 <= e.response.status_code < 500:
                    break
            except Exception as e:
                last_exc = e
            
            if attempt < self.max_retries:
                time.sleep(self.backoff_base * (2 ** attempt))

        logger.error("HTTP Forward failed: %s", last_exc)
        return ""

    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """转发请求到外部服务（异步版）"""
        url, payload, headers = self._prepare_request(messages, model, max_tokens, temperature, stream=stream, **kwargs)

        if stream:
            return self._async_stream_response(url, payload, headers)

        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.async_http_client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return self._parse_response(response.json())
            except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as e:
                last_exc = e
                if isinstance(e, httpx.HTTPStatusError) and 400 <= e.response.status_code < 500:
                    break
            except Exception as e:
                last_exc = e

            if attempt < self.max_retries:
                await asyncio.sleep(self.backoff_base * (2 ** attempt))

        logger.error("HTTP Forward Async failed: %s", last_exc)
        return ""

    async def _async_stream_response(self, url: str, payload: Dict, headers: Dict) -> AsyncIterator[str]:
        """处理异步流式响应，解析 SSE 数据行并 yield 纯文本 token，保留 reasoning_content 作为 <think> 块"""
        in_reasoning = False
        content_buf = ""
        try:
            async with self.async_http_client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}

                    # 【修改这里】：兼容主流的所有思考字段命名规范
                    reasoning = (
                        delta.get("reasoning_content") or
                        delta.get("reasoning") or
                        delta.get("thinking")
                    )
                    content = delta.get("content")

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
            logger.info("客户端连接断开，停止 HTTP Forward 流式输出")
            return
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.error("HTTP Forward stream error: %s", e)
            raise
        except Exception as e:
            logger.error("HTTP Forward stream unexpected error: %s", e)
            raise

        # 兜底逻辑移出 finally 块
        if in_reasoning:
            yield "</think>"
        if content_buf:
            yield content_buf

    def _prepare_request(self, messages, model, max_tokens, temperature, stream: bool = False, **kwargs):
        if not self.base_url:
            raise ValueError("base_url 必须设置")
        url = f"{self.base_url.rstrip('/')}{self.endpoint}"
        payload = {"messages": messages, "model": model, "stream": stream, **kwargs}
        if max_tokens is not None: payload["max_tokens"] = max_tokens
        if temperature is not None: payload["temperature"] = temperature
        headers = {"Content-Type": "application/json"}
        if self.api_key: headers["Authorization"] = f"Bearer {self.api_key}"
        return url, payload, headers

    def _parse_response(self, result: Dict[str, Any]) -> str:
        """解析响应（兼容多种格式）"""
        # OpenAI 格式
        if "choices" in result and len(result["choices"]) > 0:
            msg = result["choices"][0].get("message", {})
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")
            
            # 如果后端 API 将思考和正文分开了，我们需要把它们重新组合成 <think> 格式
            # 因为我们的 workflow 是通过正则表达式提取 <think> 标签的
            if reasoning:
                return f"<think>\n{reasoning}\n</think>\n{content}"
            return content
            
        # 简化格式
        possible_contents = [
            result.get("response"),
            result.get("content")
        ]
        # 过滤掉 None，并将有效值转为字符串，取长度最长的一个
        valid_contents = [str(x) for x in possible_contents if x is not None]
        content = max(valid_contents, key=len) if valid_contents else ""
        
        reasoning = result.get("reasoning_content") or ""
            
        if content or reasoning:
            if reasoning:
                return f"<think>\n{reasoning}\n</think>\n{content}"
            return content
            
        # 兜底：记录警告并返回错误提示
        logger.warning("无法解析响应格式: %s", result)
        return result.get("error", "响应格式不支持")

    def get_provider_name(self) -> str:
        return "http_forward"
