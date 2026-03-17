# LLM Safety Guard

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.9+-blue.svg"></a>
<a href="https://github.com/langchain-ai/langgraph"><img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-Powered-green.svg"></a>
<a href="https://fastapi.tiangolo.com/"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.100+-teal.svg"></a>

一个可插拔的 LLM 输入/输出安全防护系统，支持多 AI 服务商适配与灵活的安全策略组合。

## 核心特性

### 多层安全防护
- **输入安全检查**：敏感词扫描（AC 自动机） + 漏斗式审核（Qwen Guard → SafetyBERT）
- **输出安全检查**：流式每个 buffer 块做完整漏斗审核，通过才下发给客户端
- **安全回答生成**：对不安全输入/输出自动切换 Oyster 合规模型流式代答
- **Fail-Closed 策略**：审核失败时默认拒绝请求
- **全量收尾审核**：流式结束后对完整文本再次全量扫描并记录日志

### 多 AI 服务商支持
支持 6 种主流 AI 服务商，统一接口调用：
- **OpenAI** - GPT 系列模型
- **Azure OpenAI** - 企业级 Azure 部署
- **Anthropic** - Claude 系列模型
- **AWS Bedrock** - 亚马逊托管服务
- **Google Gemini** - Gemini 系列模型（支持 Vertex AI）
- **HTTP Forward** - 通用 HTTP 转发，适配任意自定义 API

### 灵活的集成方式
- **完整流程模式**：一次调用完成输入检查 → 内容生成 → 输出审核
- **插件式模式**：独立调用输入检查、安全生成、输出审核任意步骤
- **真实流式响应**：SSE 流式输出，每个 buffer 块经过漏斗安全门控后才下发
- **参数透传**：支持透传自定义参数（thinking、RAG、nothink 等）

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                     │
│                     (api_service.py)                        │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│                      Service Layer                           │
│                   (services/api_service.py)                 │
│  ┌──────────────┬──────────────┬──────────────┬──────────┐ │
│  │check_input   │generate_safe │check_output  │run_usage │ │
│  │_safety()     │_response()   │_safety()     │_pipeline│ │
│  └──────────────┴──────────────┴──────────────┴──────────┘ │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                        │
│                (workflows/usage_workflow.py)                │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │ Keyword  │──▶│ Prompt   │──▶│ Primary/ │──▶│Response │ │
│  │  Scan    │   │  Guard   │   │  Oyster  │   │  Guard  │ │
│  └──────────┘   └──────────┘   └──────────┘   └─────────┘ │
└───────────────┬─────────────────────────────────────────────┘
                │
        ┌───────┴───────┬────────────┬────────────┐
        ▼               ▼            ▼            ▼
┌──────────────┐ ┌─────────┐ ┌─────────┐ ┌────────────┐
│  Providers   │ │ Guards  │ │  Utils  │ │   Config   │
│              │ │         │ │         │ │            │
│ • OpenAI     │ │ • Qwen  │ │ • Token │ │ • Settings │
│ • Azure      │ │   Guard │ │   izer  │ │ • .env     │
│ • Anthropic  │ │ • Safety│ │ • HTTP  │ └────────────┘
│ • Bedrock    │ │   BERT  │ │   Client│
│ • Gemini     │ │ • Hybrid│ │ • SSE   │
│ • HTTP Fwd   │ │   Guard │ │   Stream│
└──────────────┘ │ • Sensi │ └─────────┘
                 │   tive  │
                 │   Scan  │
                 └─────────┘
```

## 快速开始

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd stream_demo

# 安装核心依赖
pip install -r requirements.txt

# 可选：安装特定 Provider 依赖
pip install anthropic boto3 google-generativeai
```

### 配置环境变量

复制 `.env.example` 创建 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env` 配置必要的 API 密钥：

```bash
# OpenAI 配置
OPENAI_API_KEY=sk-...
PRIMARY_MODEL_BASE_URL=https://api.openai.com/v1
PRIMARY_MODEL_NAME=gpt-4

# 安全模块配置
QWEN3GUARD_BASE_URL=http://localhost:8001/v1
OYSTER_BASE_URL=http://localhost:8002/v1

# Tokenizer 配置
TOKENIZER_PATH=/path/to/tokenizer

# 可选：其他 Provider 配置
AZURE_OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

### 启动服务

```bash
# 方式 1：使用 Makefile
make run

# 方式 2：直接使用 uvicorn
uvicorn api_service:app --reload --host 0.0.0.0 --port 8000

# 方式 3：使用 Python 启动
python -m uvicorn api_service:app --reload
```

服务启动后访问：
- API 服务：http://localhost:8000
- 交互式文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## 使用方式

### 1. 完整流程模式（推荐）

一次调用完成全部防护流程：

```python
import requests

response = requests.post("http://localhost:8000/moderate", json={
    "prompt": "你好，请介绍一下人工智能",
    "provider": "openai",
    "primary_model_name": "gpt-4",
    "protection_enabled": True,
    "stream": False
})

result = response.json()
print(f"安全路由: {result['route']}")
print(f"回答: {result['response']}")
```

### 2. 流式响应模式

```python
import requests

response = requests.post("http://localhost:8000/moderate", json={
    "prompt": "解释一下量子计算",
    "provider": "openai",
    "stream": True,
    "stream_chunk_size": 8
}, stream=True)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### 3. 插件式调用（Python API）

仅调用需要的防护步骤：

```python
from services import check_input_safety, generate_safe_response, check_output_safety

# 步骤 1：输入安全检查
input_check = check_input_safety({
    "prompt": "用户输入的内容"
})

if not input_check["safe"]:
    # 步骤 2：生成安全回答
    safe_resp = generate_safe_response({
        "prompt": "用户输入的内容"
    })
    response_text = safe_resp["response"]
else:
    # 调用你自己的 LLM 服务
    response_text = your_llm_service(prompt)

    # 步骤 3：输出安全检查
    output_check = check_output_safety({
        "prompt": "用户输入的内容",
        "response": response_text
    })

    if not output_check["safe"]:
        response_text = "抱歉，无法回答该问题"
```

### 4. HTTP Forward 模式（对接现有服务）

适合对接已有的 LLM 服务，透传自定义参数：

```python
from services import run_usage_pipeline

result = run_usage_pipeline({
    "prompt": "用户问题",
    "provider": "http_forward",
    "primary_model_url": "http://your-service.com/api",
    "primary_model_name": "your-model",
    "provider_config": {
        "endpoint": "/chat/completions",
        "forward_params": {
            "enable_thinking": True,
            "use_rag": True,
            "nothink": False,
            "custom_param": "custom_value"
        }
    },
    "model_params": {
        "temperature": 0.7,
        "top_p": 0.9
    }
})
```

### 5. 使用不同的 Provider

```python
# Azure OpenAI
result = run_usage_pipeline({
    "prompt": "你好",
    "provider": "azure",
    "provider_config": {
        "api_key": "your-azure-key",
        "azure_endpoint": "https://your-resource.openai.azure.com/",
        "api_version": "2024-02-15-preview"
    },
    "primary_model_name": "gpt-4"
})

# Anthropic Claude
result = run_usage_pipeline({
    "prompt": "你好",
    "provider": "anthropic",
    "provider_config": {
        "api_key": "your-anthropic-key"
    },
    "primary_model_name": "claude-3-opus-20240229"
})

# AWS Bedrock
result = run_usage_pipeline({
    "prompt": "你好",
    "provider": "bedrock",
    "provider_config": {
        "region": "us-east-1",
        "aws_access_key_id": "...",
        "aws_secret_access_key": "..."
    },
    "primary_model_name": "anthropic.claude-3-opus-20240229-v1:0"
})

# Google Gemini
result = run_usage_pipeline({
    "prompt": "你好",
    "provider": "gemini",
    "provider_config": {
        "api_key": "your-gemini-key"
    },
    "primary_model_name": "gemini-pro"
})
```

### 6. CLI 命令行模式

```bash
# 基本使用
python main.py use --prompt "你好" --provider openai --model gpt-4

# 使用 HTTP Forward
python main.py use \
  --prompt "介绍一下深度学习" \
  --provider http_forward \
  --url http://your-service.com/api \
  --model your-model

# 测试数据集
python main.py test \
  --path dataset.jsonl \
  --limit 100 \
  --generator-url http://localhost:8000/v1 \
  --generator-model gpt-4
```

## API 接口文档

详细的 API 接口文档请查看 [API_DOCS.md](API_DOCS.md)

主要端点：
- `POST /moderate` - 主审核接口（支持流式/非流式）
- `POST /v1/stop` - 停止正在进行的非流式请求
- `POST /test_dataset` - 测试数据集
- `POST /reload_sensitive` - 热更新敏感词
- `GET /health` - 健康检查
- `GET /ready` - 就绪检查

## 项目结构

```
stream_demo/
├── api_service.py              # FastAPI 应用入口
├── main.py                     # CLI 命令行工具
├── config/                     # 配置管理
│   └── settings.py            # 环境变量配置
├── services/                   # 服务层（插件式步骤）
│   └── api_service.py         # 三个独立服务 + 完整流程
├── workflows/                  # LangGraph 工作流编排
│   ├── usage_workflow.py      # 主工作流（防护流程）
│   └── testing_workflow.py    # 测试数据集评估
├── providers/                  # Provider 抽象层
│   ├── base.py                # BaseProvider 基类 + 工厂
│   ├── openai.py              # OpenAI Provider
│   ├── azure.py               # Azure OpenAI Provider
│   ├── anthropic.py           # Anthropic Provider
│   ├── bedrock.py             # AWS Bedrock Provider
│   ├── gemini.py              # Google Gemini Provider
│   └── http_forward.py        # HTTP 通用转发
├── guards/                     # 内容审核模块
│   ├── scanner.py             # 敏感词扫描（AC 自动机）
│   └── qwen_guard.py          # Qwen3Guard 审核
├── utils/                      # 工具库
│   ├── tokenizer.py           # Tokenizer 加载与缓存
│   ├── http_client.py         # HTTP 连接池
│   ├── sse_stream.py          # SSE 流式响应
│   └── helpers.py             # 通用工具函数
├── tests/                      # 测试文件
│   └── test_services.py       # 服务层测试
├── examples.py                 # Provider 使用示例
├── plugin_examples.py          # 插件式服务示例
├── frontend_examples.js        # 前端集成示例
├── pyproject.toml             # 项目配置
├── requirements.txt           # 依赖列表
└── Makefile                   # 开发命令
```

## 开发指南

### 安装开发依赖

```bash
make install
# 或
pip install -r requirements.txt
pip install -e .
```

### 运行测试

```bash
# 运行所有测试
make test

# 运行测试并生成覆盖率报告
make test-cov

# 运行特定测试
pytest tests/test_services.py -v
```

### 代码格式化与检查

```bash
# 格式化代码
make format

# 代码检查
make lint

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### 扩展 Provider

1. 在 `providers/` 目录创建新文件，继承 `BaseProvider`：

```python
from providers.base import BaseProvider
from typing import Dict, Any

class CustomProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        # 初始化逻辑

    def chat_completion(self, messages: list, model: str, **kwargs) -> str:
        # 实现 chat_completion 方法
        # 返回生成的文本
        pass
```

2. 注册到 `ProviderFactory`：

```python
# 在 providers/__init__.py 中
from providers.custom import CustomProvider

ProviderFactory.register("custom", CustomProvider)
```

3. 使用新 Provider：

```python
result = run_usage_pipeline({
    "prompt": "测试",
    "provider": "custom",
    "provider_config": {
        "api_key": "your-key"
    }
})
```

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | - |
| `PRIMARY_MODEL_BASE_URL` | 主模型 API 地址 | `https://api.openai.com/v1` |
| `PRIMARY_MODEL_NAME` | 主模型名称 | `gpt-4` |
| `QWEN3GUARD_BASE_URL` | Qwen3Guard 服务地址 | - |
| `OYSTER_BASE_URL` | Oyster 安全回答服务地址 | - |
| `TOKENIZER_PATH` | Tokenizer 文件路径 | - |
| `MAX_INFLIGHT` | 非流式最大并发请求数 | `50` |
| `SSE_MAX_INFLIGHT` | 流式最大并发请求数（0=不限） | `50` |
| `SSE_ACQUIRE_TIMEOUT` | 流式信号量等待超时（秒） | `10.0` |
| `REQUEST_TIMEOUT` | 非流式请求超时（秒） | `120` |
| `PRIMARY_MAX_RETRIES` | 主模型最大重试次数 | `2` |
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | 熔断器触发阈值 | `5` |
| `CIRCUIT_BREAKER_TIMEOUT` | 熔断器恢复等待（秒） | `60` |

### 服务配置

在 `config/settings.py` 中可以配置更多详细参数：
- Token 限制
- 重试策略
- 超时设置
- 敏感词库路径
- 审核模型参数

## 性能优化

### HTTP 连接池
- 使用 httpx 连接池，复用 TCP 连接
- 默认最大连接数：100
- 单个主机最大连接数：20
- 连接超时：5 秒，读取超时：30 秒

### 并发审核
- 输出内容按 token 切块并行审核
- 支持 fail-fast 策略，发现问题立即中断
- 默认最大并发数：4

### 缓存机制
- Tokenizer 全局缓存，避免重复加载
- Provider 实例缓存，减少初始化开销
- HTTP Client 全局共享

### 资源限制
- 最大并发请求数：50（可配置）
- 单个请求超时：120 秒（可配置）
- 输入截断：根据模型上下文窗口自动截断

## 安全特性

- **敏感词扫描**：基于 AC 自动机，O(n) 时间复杂度，支持热更新
- **内容审核**：Qwen3Guard 支持 9 类风险检测
- **Fail-Closed**：审核失败时默认拒绝，保证安全
- **并行审核**：输出内容分块并行检查，提高效率
- **API 密钥保护**：支持环境变量配置，不硬编码

## 限制与注意事项

1. **流式安全延迟**：每个 buffer 块（约 100 字符或遇到句末标点）需经过漏斗审核（关键词+Qwen Guard+SafetyBERT），会有轻微延迟
2. **多轮对话**：当前仅支持单轮对话（基于 `prompt` 字段），不支持 `messages` 数组
3. **Provider 接管**：`run_usage_pipeline` 会接管模型调用，如需保留原有流程建议使用插件式模式
4. **审核延迟**：输出审核需要时间，会增加响应延迟

## 常见问题

### Q: 如何关闭防护功能？
A: 设置 `protection_enabled: false` 即可跳过所有防护步骤。

### Q: 如何对接已有的 LLM 服务？
A: 使用 `http_forward` Provider，通过 `forward_params` 透传自定义参数。

### Q: 如何添加自定义敏感词？
A: 修改敏感词库文件后，调用 `POST /reload_sensitive` 热更新。

### Q: 输出审核失败怎么办？
A: 系统会返回安全回答（oyster）或拒绝请求，具体策略可在代码中配置。

### Q: 支持多轮对话吗？
A: 当前版本仅支持单轮对话，多轮对话支持计划中。

## License

MIT License

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 联系方式

如有问题或建议，请通过 Issue 联系我们。
