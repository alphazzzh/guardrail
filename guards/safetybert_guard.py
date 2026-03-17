"""
SafetyBERT Guard 客户端
对接独立的 FastAPI BERT+BGE 推理服务
"""
import logging
from typing import Any, Dict, List, Optional
from config.settings import (
    SAFETY_SERVICE_URL,
    SAFETY_ENSEMBLE_STRATEGY,
    SAFETY_THRESHOLD,
    GUARD_CONNECT_TIMEOUT,
    GUARD_READ_TIMEOUT,
    GUARD_MAX_CONNECTIONS,
    GUARD_MAX_KEEPALIVE,
)
from utils.http_client import get_shared_async_http_client, get_shared_http_client

logger = logging.getLogger("safeguard_system")

class SafetyBERTGuard:
    def __init__(self, 
                 url: str = SAFETY_SERVICE_URL, 
                 strategy: str = SAFETY_ENSEMBLE_STRATEGY,
                 threshold: float = SAFETY_THRESHOLD):
        self.url = url
        self.strategy = strategy
        self.threshold = threshold
        
        # 复用全局共享客户端，防止连接句柄泄露
        self.async_client = get_shared_async_http_client(
            GUARD_CONNECT_TIMEOUT,
            GUARD_READ_TIMEOUT,
            GUARD_MAX_CONNECTIONS,
            GUARD_MAX_KEEPALIVE,
        )
        self.sync_client = get_shared_http_client(
            GUARD_CONNECT_TIMEOUT,
            GUARD_READ_TIMEOUT,
            GUARD_MAX_CONNECTIONS,
            GUARD_MAX_KEEPALIVE,
        )

    async def async_check(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """异步调用安全服务进行类别判定"""
        full_text = "\n".join([m.get("content", "") for m in messages if m.get("content")]).strip()
        if not full_text: return {"safe": True, "categories": []}

        try:
            payload = {"text": full_text, "strategy": self.strategy, "threshold": self.threshold}
            response = await self.async_client.post(self.url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            return {
                "safe": data.get("safe", True),
                "status": "safe" if data.get("safe") else "unsafe",
                "categories": data.get("categories", []),
                "scores": data.get("scores", {}),
                "raw": f"Strategy: {self.strategy}"
            }
        except Exception as e:
            logger.error(f"Safety Service Error: {e}")
            return {"safe": True, "status": "error", "raw": str(e), "categories": []}

    def check(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """同步版 (复用连接池)"""
        full_text = "\n".join([m.get("content", "") for m in messages if m.get("content")]).strip()
        if not full_text: return {"safe": True, "categories": []}
        try:
            payload = {"text": full_text, "strategy": self.strategy, "threshold": self.threshold}
            response = self.sync_client.post(self.url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Safety Service Sync Error: {e}")
            return {"safe": True, "status": "error", "categories": []}
