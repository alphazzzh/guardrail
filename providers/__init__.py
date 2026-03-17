from .base import BaseProvider, ProviderFactory
from .openai import OpenAIProvider
from .azure import AzureOpenAIProvider
from .anthropic import AnthropicProvider
from .bedrock import BedrockProvider
# from .gemini import GeminiProvider  # 暂时屏蔽
from .http_forward import HTTPForwardProvider

# 注册所有 Providers
ProviderFactory.register("openai", OpenAIProvider)
ProviderFactory.register("azure", AzureOpenAIProvider)
ProviderFactory.register("anthropic", AnthropicProvider)
ProviderFactory.register("bedrock", BedrockProvider)
# ProviderFactory.register("gemini", GeminiProvider)  # 暂时屏蔽
ProviderFactory.register("http_forward", HTTPForwardProvider)

__all__ = [
    "BaseProvider",
    "ProviderFactory",
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "AnthropicProvider",
    "BedrockProvider",
    # "GeminiProvider",  # 暂时屏蔽
    "HTTPForwardProvider",
]
