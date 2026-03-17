"""配置管理模块"""
import os
from pathlib import Path

# ================== API Keys ==================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "")

# ================== 服务地址配置 ==================
# 生产部署时必须通过环境变量显式设置这些地址。
QWEN3GUARD_BASE_URL = os.getenv("QWEN3GUARD_BASE_URL", "")
OYSTER_BASE_URL = os.getenv("OYSTER_BASE_URL", "")
PRIMARY_MODEL_BASE_URL = os.getenv("PRIMARY_MODEL_BASE_URL", "")

if not QWEN3GUARD_BASE_URL:
    import warnings
    warnings.warn("QWEN3GUARD_BASE_URL 未设置，Guard 功能将不可用", stacklevel=2)
if not OYSTER_BASE_URL:
    import warnings
    warnings.warn("OYSTER_BASE_URL 未设置，Oyster 安全代答将不可用", stacklevel=2)

# ================== BERT + BGE 模型服务配置 ==================
SAFETY_SERVICE_URL = os.getenv("SAFETY_SERVICE_URL", "http://localhost:8008/predict")
SAFETY_ENSEMBLE_STRATEGY = os.getenv("SAFETY_ENSEMBLE_STRATEGY", "intersection")
SAFETY_THRESHOLD = float(os.getenv("SAFETY_THRESHOLD", "0.5"))

# Qwen 直接映射表
QWEN_DIRECT_MAP = {
    "Violent": "违反社会主义核心价值观",
    "Sexual Content or Sexual Acts": "违反社会主义核心价值观",
    "Politically Sensitive Topics": "违反社会主义核心价值观",
    "PII": "侵犯他人合法权益",
    "Suicide & Self-Harm": "无法满足特定服务类型的安全需求",
    "Copyright Violation": "商业违法违规",
    "Jailbreak": "提示词越狱",
}

# ================== 模型名称配置 ==================
MODEL_MODERATION = os.getenv("QWEN3GUARD_MODEL", "Qwen3Guard-Gen-8B")
MODEL_OYSTER = os.getenv("OYSTER_MODEL", "oyster")
PRIMARY_MODEL_NAME = os.getenv("PRIMARY_MODEL_NAME", "primary")

# ================== 超时与并发控制 ==================
SSE_MAX_INFLIGHT = int(os.getenv("SSE_MAX_INFLIGHT", "50"))          # 最大并发流式请求数，0 表示不限制
SSE_ACQUIRE_TIMEOUT = float(os.getenv("SSE_ACQUIRE_TIMEOUT", "10.0")) # 等待获取流式信号量的超时（秒）

PRIMARY_CONNECT_TIMEOUT = float(os.getenv("PRIMARY_CONNECT_TIMEOUT", "5.0"))
PRIMARY_READ_TIMEOUT = float(os.getenv("PRIMARY_READ_TIMEOUT", "120.0"))
PRIMARY_MAX_RETRIES = int(os.getenv("PRIMARY_MAX_RETRIES", "2"))
PRIMARY_BACKOFF_BASE = float(os.getenv("PRIMARY_BACKOFF_BASE", "0.6"))
PRIMARY_MAX_CONNECTIONS = int(os.getenv("PRIMARY_MAX_CONNECTIONS", "100"))
PRIMARY_MAX_KEEPALIVE = int(os.getenv("PRIMARY_MAX_KEEPALIVE", "60"))
PRIMARY_CONTEXT_TOKEN_LIMIT = int(os.getenv("PRIMARY_CONTEXT_TOKEN_LIMIT", "32768"))
PRIMARY_DEFAULT_MAX_TOKENS = int(os.getenv("PRIMARY_DEFAULT_MAX_TOKENS", "16384"))

GUARD_CONNECT_TIMEOUT = float(os.getenv("GUARD_CONNECT_TIMEOUT", "3.0"))
GUARD_READ_TIMEOUT = float(os.getenv("GUARD_READ_TIMEOUT", "30.0"))
GUARD_MAX_CONNECTIONS = int(os.getenv("GUARD_MAX_CONNECTIONS", "50"))
GUARD_MAX_KEEPALIVE = int(os.getenv("GUARD_MAX_KEEPALIVE", "30"))

OYSTER_CONNECT_TIMEOUT = float(os.getenv("OYSTER_CONNECT_TIMEOUT", "5.0"))
OYSTER_READ_TIMEOUT = float(os.getenv("OYSTER_READ_TIMEOUT", "60.0"))
OYSTER_MAX_CONNECTIONS = int(os.getenv("OYSTER_MAX_CONNECTIONS", "50"))
OYSTER_MAX_KEEPALIVE = int(os.getenv("OYSTER_MAX_KEEPALIVE", "30"))

# 新增/补全的缺失常量
RESPONSE_GUARD_MAX_WORKERS = int(os.getenv("RESPONSE_GUARD_MAX_WORKERS", "5"))
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
CIRCUIT_BREAKER_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))
CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS = int(os.getenv("CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS", "3"))

PROVIDER_CACHE_MAX_SIZE = int(os.getenv("PROVIDER_CACHE_MAX_SIZE", "16"))
GUARD_MAX_CHUNK_TOKENS = int(os.getenv("GUARD_MAX_CHUNK_TOKENS", "512"))
GUARD_CHUNK_OVERLAP_TOKENS = int(os.getenv("GUARD_CHUNK_OVERLAP_TOKENS", "128"))

# Tokenizer 路径
BASE_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", str(BASE_DIR / "assets" / "tokenizer"))

# ================== PII 脱敏配置 ==================
PII_ENABLED = os.getenv("PII_ENABLED", "true").lower() in ("true", "1", "yes")
# 启用 spaCy 中文 NER（需安装 zh_core_web_sm；false 则退回纯规则模式）
PII_USE_SPACY_ZH = os.getenv("PII_USE_SPACY_ZH", "true").lower() in ("true", "1", "yes")
# 用户自定义黑名单词汇文件路径（YAML 格式，见 config/pii_deny_list.yaml）
PII_DENY_LIST_PATH = os.getenv("PII_DENY_LIST_PATH", str(BASE_DIR / "config" / "pii_deny_list.yaml"))

# ================== 安全与跨域配置 ==================
# 生产环境应严格配置，例如 ["https://yourdomain.com"]
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000").split(",")

# ================== 敏感词库路径 ==================
_WORDBAGS_DIR = Path(os.getenv("WORDBAGS_DIR", str(BASE_DIR / "assets" / "wordbags")))
SENSITIVE_WORD_PATHS = [
    str(_WORDBAGS_DIR / "政治类型.txt"),
    str(_WORDBAGS_DIR / "COVID-19词库.txt"),
    str(_WORDBAGS_DIR / "GFW补充词库.txt"),
    str(_WORDBAGS_DIR / "反动词库.txt"),
    str(_WORDBAGS_DIR / "网易前端过滤敏感词库.txt"),
]
