"""Provider 基类定义"""
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union


class BaseProvider(ABC):
    """AI 服务提供商的抽象基类"""

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """
        初始化 Provider

        Args:
            api_key: API 密钥
            base_url: 服务基础 URL (可选)
            **kwargs: 其他配置参数
        """
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """同步调用聊天补全 API"""
        pass

    @abstractmethod
    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """异步调用聊天补全 API"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """返回 Provider 名称"""
        pass


class ProviderFactory:
    """Provider 工厂类"""

    _providers: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, provider_class: type):
        """注册 Provider"""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def create(cls, provider_name: str, **kwargs) -> BaseProvider:
        """
        创建 Provider 实例

        Args:
            provider_name: Provider 名称
            **kwargs: 初始化参数

        Returns:
            Provider 实例

        Raises:
            ValueError: 如果 Provider 不存在
        """
        provider_class = cls._providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {', '.join(cls._providers.keys())}"
            )
        return provider_class(**kwargs)

    @classmethod
    def list_providers(cls) -> List[str]:
        """返回所有已注册的 Provider 名称"""
        return list(cls._providers.keys())
