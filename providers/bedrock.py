"""AWS Bedrock Provider 实现"""
import json
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from .base import BaseProvider

logger = logging.getLogger("safeguard_system")


class BedrockProvider(BaseProvider):
    """AWS Bedrock Provider (支持 Claude 等模型)"""

    def __init__(
        self,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__("", None, **kwargs)
        try:
            import boto3
            # 注意：boto3 本身不直接支持 async，生产中通常使用 aiobotocore 
            # 这里由于时间限制，异步版通过 run_in_executor 模拟
            self.session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
            )
            self.client = self.session.client("bedrock-runtime")
        except ImportError:
            raise ImportError("boto3 package not installed. Run: pip install boto3")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        # 简化版实现，仅示意
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or 512,
            "messages": [m for m in messages if m["role"] != "system"],
            "system": next((m["content"] for m in messages if m["role"] == "system"), "")
        })
        
        response = self.client.invoke_model(body=body, modelId=model)
        response_body = json.loads(response.get("body").read())
        return response_body.get("content", [{}])[0].get("text", "")

    async def async_chat_completion(self, *args, **kwargs):
        """异步模拟实现 (问题 8 修复)"""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.chat_completion(*args, **kwargs))

    def get_provider_name(self) -> str: return "bedrock"
