"""Google Gemini Provider 实现 (支持 AI Studio 和 Vertex AI)"""
import logging
from typing import Any, Dict, Iterator, List, Optional, Union

from .base import BaseProvider, ProviderFactory

logger = logging.getLogger("safeguard_system")


class GeminiProvider(BaseProvider):
    """Google Gemini Provider (AI Studio / Vertex AI)"""

    def __init__(
        self,
        api_key: str = "",
        project_id: Optional[str] = None,
        location: str = "us-central1",
        use_vertex: bool = False,
        **kwargs
    ):
        """
        Args:
            api_key: Google API Key (AI Studio 使用)
            project_id: GCP Project ID (Vertex AI 使用)
            location: Vertex AI 区域
            use_vertex: 是否使用 Vertex AI (False 则使用 AI Studio)
        """
        super().__init__(api_key, None, **kwargs)
        self.use_vertex = use_vertex
        self.project_id = project_id
        self.location = location

        try:
            import google.generativeai as genai

            if use_vertex:
                # Vertex AI 模式
                if not project_id:
                    raise ValueError("project_id is required for Vertex AI")
                import vertexai
                vertexai.init(project=project_id, location=location)
                self.is_vertex = True
            else:
                # AI Studio 模式
                if not api_key:
                    raise ValueError("api_key is required for AI Studio")
                genai.configure(api_key=api_key)
                self.is_vertex = False

            self.genai = genai

        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install it with: pip install google-generativeai"
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
        """同步调用 Gemini API"""
        gemini_messages, system_instruction = self._prepare_messages(messages)
        generation_config = self._prepare_config(max_tokens, temperature)

        try:
            model_instance = self._get_model_instance(model, system_instruction)

            if len(gemini_messages) == 1 and gemini_messages[0]["role"] == "user":
                prompt = gemini_messages[0]["parts"][0]
                if stream:
                    return self._stream_generate(model_instance, prompt, generation_config)
                else:
                    response = model_instance.generate_content(prompt, generation_config=generation_config)
                    return response.text
            else:
                chat = model_instance.start_chat(history=gemini_messages[:-1])
                last_message = gemini_messages[-1]["parts"][0]
                if stream:
                    return self._stream_chat(chat, last_message, generation_config)
                else:
                    response = chat.send_message(last_message, generation_config=generation_config)
                    return response.text

        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            raise

    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """异步调用 Gemini API"""
        gemini_messages, system_instruction = self._prepare_messages(messages)
        generation_config = self._prepare_config(max_tokens, temperature)

        try:
            model_instance = self._get_model_instance(model, system_instruction)

            if len(gemini_messages) == 1 and gemini_messages[0]["role"] == "user":
                prompt = gemini_messages[0]["parts"][0]
                if stream:
                    return self._async_stream_generate(model_instance, prompt, generation_config)
                else:
                    response = await model_instance.generate_content_async(prompt, generation_config=generation_config)
                    return response.text
            else:
                chat = model_instance.start_chat(history=gemini_messages[:-1])
                last_message = gemini_messages[-1]["parts"][0]
                if stream:
                    return self._async_stream_chat(chat, last_message, generation_config)
                else:
                    response = await chat.send_message_async(last_message, generation_config=generation_config)
                    return response.text

        except Exception as e:
            logger.error(f"Gemini Async API Error: {e}")
            raise

    def _prepare_messages(self, messages):
        gemini_messages = []
        system_instruction = None
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})
            else:
                gemini_messages.append({"role": "user", "parts": [content]})
        return gemini_messages, system_instruction

    def _prepare_config(self, max_tokens, temperature):
        generation_config = {}
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        if temperature is not None:
            generation_config["temperature"] = temperature
        return generation_config if generation_config else None

    def _get_model_instance(self, model, system_instruction):
        if self.is_vertex:
            from vertexai.generative_models import GenerativeModel
            return GenerativeModel(model, system_instruction=system_instruction)
        else:
            return self.genai.GenerativeModel(model, system_instruction=system_instruction)

    def _stream_generate(self, model, prompt: str, generation_config: Dict) -> Iterator[str]:
        """处理同步流式生成"""
        try:
            response = model.generate_content(prompt, generation_config=generation_config, stream=True)
            for chunk in response:
                if chunk.text: yield chunk.text
        except Exception as e:
            logger.error(f"Gemini stream error: {e}")
            raise

    async def _async_stream_generate(self, model, prompt: str, generation_config: Dict) -> AsyncIterator[str]:
        """处理异步流式生成"""
        try:
            response = await model.generate_content_async(prompt, generation_config=generation_config, stream=True)
            async for chunk in response:
                if chunk.text: yield chunk.text
        except Exception as e:
            logger.error(f"Gemini async stream error: {e}")
            raise

    def _stream_chat(self, chat, message: str, generation_config: Dict) -> Iterator[str]:
        """处理同步流式对话"""
        try:
            response = chat.send_message(message, generation_config=generation_config, stream=True)
            for chunk in response:
                if chunk.text: yield chunk.text
        except Exception as e:
            logger.error(f"Gemini chat stream error: {e}")
            raise

    async def _async_stream_chat(self, chat, message: str, generation_config: Dict) -> AsyncIterator[str]:
        """处理异步流式对话"""
        try:
            response = await chat.send_message_async(message, generation_config=generation_config, stream=True)
            async for chunk in response:
                if chunk.text: yield chunk.text
        except Exception as e:
            logger.error(f"Gemini async chat stream error: {e}")
            raise

    def get_provider_name(self) -> str:
        return "gemini-vertex" if self.is_vertex else "gemini"


# 修复问题 J：注册统一由 providers/__init__.py 负责，此处删除重复注册
