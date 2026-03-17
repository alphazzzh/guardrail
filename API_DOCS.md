# LLM Safety Guard - API 接口文档

本文档面向前端开发者，详细说明如何调用 LLM Safety Guard 服务的各个 API 接口。

## 目录

- [服务信息](#服务信息)
- [通用说明](#通用说明)
- [认证](#认证)
- [接口列表](#接口列表)
  - [健康检查](#1-健康检查)
  - [就绪检查](#2-就绪检查)
  - [主审核接口](#3-主审核接口)
  - [测试数据集](#4-测试数据集)
  - [停止正在进行的请求](#5-停止正在进行的请求)
  - [热更新敏感词](#6-热更新敏感词)
- [错误处理](#错误处理)
- [前端集成示例](#前端集成示例)

---

## 服务信息

- **Base URL**: `http://localhost:8000` (开发环境)
- **协议**: HTTP/HTTPS
- **数据格式**: JSON
- **字符编码**: UTF-8
- **API 版本**: v0.1.0

## 通用说明

### 请求头

所有 POST 请求应包含以下请求头：

```http
Content-Type: application/json
```

可选请求头：

```http
X-Request-ID: <唯一请求ID>  # 用于请求追踪，如不提供会自动生成
```

### 响应头

所有响应都包含以下响应头：

```http
X-Request-ID: <请求ID>  # 与请求中的 ID 对应或自动生成的 ID
```

### CORS

服务已配置 CORS，允许所有来源的跨域请求：

- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Methods: *`
- `Access-Control-Allow-Headers: *`

## 认证

当前版本不需要 API 认证。如需在生产环境中部署，建议添加 API Key 或 JWT 认证。

---

## 接口列表

### 1. 健康检查

检查服务是否正常运行。

#### 请求

```http
GET /health
```

#### 响应

**状态码**: `200 OK`

**响应体**:

```json
{
  "status": "healthy",
  "service": "llm-safety-guard",
  "version": "0.1.0"
}
```

#### 前端示例

```javascript
// Fetch API
fetch('http://localhost:8000/health')
  .then(response => response.json())
  .then(data => console.log('Service status:', data.status));

// Axios
axios.get('http://localhost:8000/health')
  .then(response => console.log('Service status:', response.data.status));
```

---

### 2. 就绪检查

检查服务是否准备好接受请求（检查依赖服务状态）。

#### 请求

```http
GET /ready
```

#### 响应

**状态码**: `200 OK`

**响应体**:

```json
{
  "status": "ready",
  "inflight": 50,          // 当前并发请求数
  "max_inflight": 50       // 最大并发请求数
}
```

#### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 服务状态，固定为 "ready" |
| `inflight` | number/string | 当前可用并发槽位，"unlimited" 表示无限制 |
| `max_inflight` | number | 最大并发请求数 |

---

### 3. 主审核接口

**核心接口**，用于对用户输入进行安全审核并生成回答。

#### 请求

```http
POST /moderate
```

#### 请求体

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `prompt` | string | ✅ | - | 用户输入的文本内容 |
| `protection_enabled` | boolean | ❌ | `true` | 是否启用安全防护 |
| `provider` | string | ❌ | `"openai"` | AI 服务商：`openai`、`azure`、`anthropic`、`bedrock`、`gemini`、`http_forward` |
| `primary_model_url` | string | ❌ | 环境变量 | 主模型 API 地址 |
| `primary_model_name` | string | ❌ | 环境变量 | 主模型名称 |
| `primary_max_tokens` | number | ❌ | `1024` | 生成的最大 token 数 |
| `provider_config` | object | ❌ | `{}` | Provider 专用配置（见下方） |
| `model_params` | object | ❌ | `{}` | 模型参数（temperature、top_p 等） |
| `stream` | boolean | ❌ | `false` | 是否以流式格式返回 |

#### Provider 配置 (`provider_config`)

不同 Provider 支持的配置参数：

**OpenAI**:
```json
{
  "api_key": "sk-..."  // 可选，默认使用环境变量
}
```

**Azure OpenAI**:
```json
{
  "api_key": "your-azure-key",
  "azure_endpoint": "https://your-resource.openai.azure.com/",
  "api_version": "2024-02-15-preview"
}
```

**Anthropic**:
```json
{
  "api_key": "sk-ant-..."
}
```

**AWS Bedrock**:
```json
{
  "region": "us-east-1",
  "aws_access_key_id": "...",
  "aws_secret_access_key": "..."
}
```

**Google Gemini**:
```json
{
  "api_key": "your-gemini-key",
  "project_id": "your-project-id",  // Vertex AI 需要
  "location": "us-central1",        // Vertex AI 需要
  "use_vertex": false               // 是否使用 Vertex AI
}
```

**HTTP Forward** (透传模式):
```json
{
  "api_key": "optional-key",
  "endpoint": "/chat/completions",
  "timeout": 120.0,
  "forward_params": {
    "enable_thinking": true,   // 自定义参数会透传给目标服务
    "use_rag": true,
    "nothink": false,
    "custom_param": "value"
  }
}
```

#### 模型参数 (`model_params`)

通用模型参数（会与 `forward_params` 合并，优先级更高）：

```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "enable_thinking": true,
  "nothink": false
}
```

#### 响应 (非流式)

**状态码**: `200 OK`

**响应体**:

```json
{
  "prompt": "用户输入的内容",
  "route": "primary",
  "prompt_moderation": {
    "safe": true,
    "status": "safe",
    "raw": "...",
    "normalized_categories": []
  },
  "response": "模型生成的回答",
  "response_moderation": {
    "safe": true,
    "status": "safe",
    "raw": "...",
    "normalized_categories": []
  },
  "keyword_flagged": false,
  "protection_enabled": true,
  "logs": [
    "敏感词扫描完成: 命中=False hits=0",
    "prompt guard done: status=safe",
    "primary generation done",
    "回答审核完成: status=safe chunk=all"
  ],
  "metadata": {
    "primary_model_url": "https://api.openai.com/v1",
    "primary_model_name": "gpt-4",
    "primary_max_tokens": 1024,
    "protection_enabled": true,
    "provider": "openai"
  }
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `prompt` | string | 原始用户输入 |
| `route` | string | 路由决策：`primary`（安全，使用主模型）或 `oyster`（不安全，使用安全回答模型） |
| `prompt_moderation` | object | 输入审核结果 |
| `prompt_moderation.safe` | boolean | 输入是否安全 |
| `prompt_moderation.status` | string | 审核状态：`safe`、`unsafe`、`blocked`、`error` |
| `prompt_moderation.normalized_categories` | array | 检测到的风险类别数组 |
| `response` | string | 生成的回答文本 |
| `response_moderation` | object | 输出审核结果（结构同 `prompt_moderation`） |
| `keyword_flagged` | boolean | 是否命中敏感词 |
| `protection_enabled` | boolean | 防护是否启用 |
| `logs` | array | 处理日志数组 |
| `metadata` | object | 元数据信息 |

#### 响应 (流式)

当 `stream: true` 时，返回 SSE (Server-Sent Events) 格式。流式路径会对每个输出 buffer 执行完整安全漏斗审核（关键词 → Qwen Guard → SafetyBERT），通过后才向客户端释放；若检测到违规，自动切换 Oyster 合规模型代答。流结束后还会对完整文本执行一次全量审核并记录日志。

> **并发限制**：流式请求受 `SSE_MAX_INFLIGHT` 控制（默认 50）。超出上限且等待超过 `SSE_ACQUIRE_TIMEOUT`（默认 10 秒）时返回 `503`。

**状态码**: `200 OK`

**响应头**:
```http
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

**响应体** (OpenAI 兼容格式):

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"你好"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"，很"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"高兴"},"finish_reason":null}]}

data: [DONE]
```

**安全事件 chunk（违规时插入）**：

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":1234567890,"model":"primary","choices":[{"index":0,"delta":{"content":"\n\n[提示：部分回答涉及敏感信息（违规类别），已拦截。以下由合规模型代答：]\n"},"finish_reason":null}]}

#### 请求示例

**基础请求**:
```json
{
  "prompt": "你好，请介绍一下人工智能",
  "provider": "openai",
  "primary_model_name": "gpt-4",
  "protection_enabled": true
}
```

**关闭防护**:
```json
{
  "prompt": "你好",
  "protection_enabled": false
}
```

**使用 HTTP Forward 透传参数**:
```json
{
  "prompt": "解释一下量子计算",
  "provider": "http_forward",
  "primary_model_url": "http://your-service.com/api",
  "primary_model_name": "your-model",
  "provider_config": {
    "endpoint": "/chat/completions",
    "forward_params": {
      "enable_thinking": true,
      "use_rag": true
    }
  },
  "model_params": {
    "temperature": 0.7,
    "top_p": 0.9
  }
}
```

**流式请求**:
```json
{
  "prompt": "写一首诗",
  "stream": true
}
```

#### 前端示例

**基础调用 (Fetch API)**:
```javascript
async function moderateMessage(userInput) {
  const response = await fetch('http://localhost:8000/moderate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Request-ID': generateUUID()
    },
    body: JSON.stringify({
      prompt: userInput,
      provider: 'openai',
      primary_model_name: 'gpt-4'
    })
  });

  const data = await response.json();

  if (data.keyword_flagged || !data.prompt_moderation?.safe) {
    console.log('输入不安全，路由到安全回答');
  }

  return data.response;
}
```

**流式调用 (Fetch API)**:
```javascript
async function streamModerate(userInput, onChunk) {
  const response = await fetch('http://localhost:8000/moderate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      prompt: userInput,
      stream: true,
      stream_chunk_size: 8
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') {
          console.log('Stream completed');
          return;
        }

        try {
          const parsed = JSON.parse(data);
          const content = parsed.choices[0]?.delta?.content;
          if (content) {
            onChunk(content);
          }
        } catch (e) {
          console.error('Parse error:', e);
        }
      }
    }
  }
}

// 使用示例
streamModerate('你好', (chunk) => {
  console.log('Received:', chunk);
  // 更新 UI 显示流式内容
  document.getElementById('output').innerText += chunk;
});
```

**使用 Axios**:
```javascript
import axios from 'axios';

async function moderateWithAxios(userInput) {
  try {
    const response = await axios.post('http://localhost:8000/moderate', {
      prompt: userInput,
      provider: 'openai',
      protection_enabled: true
    }, {
      headers: {
        'X-Request-ID': generateUUID()
      },
      timeout: 120000  // 120 秒超时
    });

    return response.data;
  } catch (error) {
    if (error.response) {
      // 服务器返回错误状态码
      console.error('Server error:', error.response.status, error.response.data);
    } else if (error.request) {
      // 请求已发送但未收到响应
      console.error('Network error:', error.request);
    } else {
      console.error('Error:', error.message);
    }
    throw error;
  }
}
```

**React Hook 示例**:
```javascript
import { useState } from 'react';

function useModerate() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const moderate = async (prompt, options = {}) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/moderate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          provider: options.provider || 'openai',
          protection_enabled: options.protectionEnabled ?? true,
          ...options
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { moderate, loading, error };
}

// 组件中使用
function ChatComponent() {
  const { moderate, loading, error } = useModerate();
  const [response, setResponse] = useState('');

  const handleSubmit = async (userInput) => {
    const result = await moderate(userInput);
    setResponse(result.response);
  };

  return (
    <div>
      {loading && <p>处理中...</p>}
      {error && <p>错误: {error}</p>}
      {response && <p>回答: {response}</p>}
    </div>
  );
}
```

**Vue 3 Composition API 示例**:
```javascript
import { ref } from 'vue';

export function useModerate() {
  const loading = ref(false);
  const error = ref(null);

  const moderate = async (prompt, options = {}) => {
    loading.value = true;
    error.value = null;

    try {
      const response = await fetch('http://localhost:8000/moderate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          provider: options.provider || 'openai',
          protection_enabled: options.protectionEnabled ?? true,
          ...options
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      return await response.json();
    } catch (err) {
      error.value = err.message;
      throw err;
    } finally {
      loading.value = false;
    }
  };

  return { moderate, loading, error };
}
```

---

### 4. 测试数据集

批量测试数据集的审核效果。

#### 请求

```http
POST /test_dataset
```

#### 请求体

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `dataset_path` | string | ✅ | - | 数据集文件路径（JSONL 格式） |
| `limit_per_category` | number | ❌ | `100` | 每个类别的最大测试数量 |
| `prompt_field` | string | ❌ | `"prompt"` | JSONL 中 prompt 字段名 |
| `category_field` | string | ❌ | `"category"` | JSONL 中类别字段名 |
| `generator_url` | string | ❌ | 环境变量 | 生成器 API 地址 |
| `generator_model` | string | ❌ | 环境变量 | 生成器模型名称 |

#### 响应

**状态码**: `200 OK`

**响应体**:

```json
{
  "dataset_path": "/path/to/dataset.jsonl",
  "limit_per_category": 100,
  "prompt_scores": {
    "政治敏感": {
      "total": 50,
      "blocked": 48,
      "score": 0.96
    },
    "正常对话": {
      "total": 100,
      "blocked": 2,
      "score": 0.98
    }
  },
  "total_prompt_score": 0.97,
  "response_scores": {
    "政治敏感": {
      "total": 50,
      "safe": 49,
      "score": 0.98
    }
  },
  "total_response_score": 0.98,
  "logs": [...],
  "generator": {
    "url": "http://localhost:8000/v1",
    "model": "gpt-4"
  }
}
```

#### 前端示例

```javascript
async function testDataset(datasetPath) {
  const response = await fetch('http://localhost:8000/test_dataset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_path: datasetPath,
      limit_per_category: 100
    })
  });

  const result = await response.json();
  console.log('总分数:', result.total_prompt_score);
  return result;
}
```

---

### 5. 停止正在进行的请求

取消一个非流式的 in-flight 请求（通过 request_id 标识）。

#### 请求

```http
POST /v1/stop
```

#### 请求体

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `request_id` | string | ✅ | 要取消的请求 ID（来自响应头 `X-Request-ID`） |

```json
{
  "request_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
}
```

#### 响应

**取消成功**:
```json
{ "status": "ok" }
```

**未找到或已结束**:
```json
{ "status": "not_found", "message": "Task not found or already finished" }
```

---

### 6. 热更新敏感词

#### 请求

```http
POST /reload_sensitive
```

**无需请求体**

#### 响应

**状态码**: `200 OK`

**响应体**:

```json
{
  "ok": true,
  "word_count": 15234
}
```

#### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `ok` | boolean | 是否重载成功 |
| `word_count` | number | 加载的敏感词总数 |

#### 前端示例

```javascript
async function reloadSensitiveWords() {
  const response = await fetch('http://localhost:8000/reload_sensitive', {
    method: 'POST'
  });

  const result = await response.json();
  if (result.ok) {
    console.log(`敏感词库已更新，共 ${result.word_count} 个词`);
  }
  return result;
}
```

---

## 错误处理

### HTTP 状态码

| 状态码 | 说明 | 处理建议 |
|--------|------|----------|
| `200` | 请求成功 | - |
| `400` | 请求参数错误 | 检查请求参数格式和内容 |
| `422` | JSON 格式错误或字段校验失败 | 检查 JSON 格式和必填字段 |
| `499` | 客户端断开连接 | 检查网络连接或增加超时时间 |
| `500` | 服务器内部错误 | 联系管理员或查看日志 |
| `503` | 流式并发已达上限（SSE_MAX_INFLIGHT）| 稍后重试，或联系管理员调大限制 |
| `504` | 请求超时 | 减少内容长度或增加超时配置 |

### 错误响应格式

```json
{
  "detail": "错误详细信息"
}
```

### 前端错误处理示例

```javascript
async function moderateWithErrorHandling(prompt) {
  try {
    const response = await fetch('http://localhost:8000/moderate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });

    // 检查 HTTP 状态码
    if (!response.ok) {
      const error = await response.json();

      switch (response.status) {
        case 400:
          throw new Error(`参数错误: ${error.detail}`);
        case 422:
          throw new Error(`格式错误: ${JSON.stringify(error.detail)}`);
        case 499:
          throw new Error('请求被取消');
        case 500:
          throw new Error('服务器错误，请稍后重试');
        case 503:
          throw new Error('流式并发已满，请稍后重试');
        case 504:
          throw new Error('请求超时，请减少内容长度或稍后重试');
        default:
          throw new Error(`未知错误 (${response.status}): ${error.detail}`);
      }
    }

    return await response.json();
  } catch (error) {
    if (error instanceof TypeError) {
      // 网络错误
      console.error('网络错误:', error.message);
      throw new Error('网络连接失败，请检查网络设置');
    }
    throw error;
  }
}
```

---

## 前端集成示例

### 完整的聊天组件示例 (React)

```javascript
import React, { useState, useRef, useEffect } from 'react';

function ChatComponent() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 非流式发送
  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/moderate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: input,
          provider: 'openai',
          protection_enabled: true
        })
      });

      const data = await response.json();

      const assistantMessage = {
        role: 'assistant',
        content: data.response,
        safe: data.prompt_moderation?.safe,
        route: data.route
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'error',
        content: '发送失败: ' + error.message
      }]);
    } finally {
      setLoading(false);
    }
  };

  // 流式发送
  const handleStreamSend = async () => {
    if (!input.trim() || streaming) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setStreaming(true);

    // 添加一个占位消息用于流式更新
    const messageId = Date.now();
    setMessages(prev => [...prev, {
      id: messageId,
      role: 'assistant',
      content: '',
      streaming: true
    }]);

    try {
      const response = await fetch('http://localhost:8000/moderate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: input,
          stream: true,
          stream_chunk_size: 8
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              setMessages(prev => prev.map(msg =>
                msg.id === messageId
                  ? { ...msg, streaming: false }
                  : msg
              ));
              continue;
            }

            try {
              const parsed = JSON.parse(data);
              const content = parsed.choices[0]?.delta?.content;
              if (content) {
                setMessages(prev => prev.map(msg =>
                  msg.id === messageId
                    ? { ...msg, content: msg.content + content }
                    : msg
                ));
              }
            } catch (e) {
              console.error('Parse error:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Stream error:', error);
      setMessages(prev => prev.map(msg =>
        msg.id === messageId
          ? { ...msg, content: '发送失败: ' + error.message, streaming: false }
          : msg
      ));
    } finally {
      setStreaming(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="content">{msg.content}</div>
            {msg.route && (
              <div className="metadata">
                路由: {msg.route} | 安全: {msg.safe ? '✓' : '✗'}
              </div>
            )}
            {msg.streaming && <span className="cursor">▊</span>}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="输入消息..."
          disabled={loading || streaming}
        />
        <button onClick={handleSend} disabled={loading || streaming}>
          {loading ? '发送中...' : '发送'}
        </button>
        <button onClick={handleStreamSend} disabled={loading || streaming}>
          {streaming ? '流式中...' : '流式发送'}
        </button>
      </div>
    </div>
  );
}

export default ChatComponent;
```

### TypeScript 类型定义

```typescript
// types.ts
export interface ModerateRequest {
  prompt: string;
  protection_enabled?: boolean;
  provider?: 'openai' | 'azure' | 'anthropic' | 'bedrock' | 'gemini' | 'http_forward';
  primary_model_url?: string;
  primary_model_name?: string;
  primary_max_tokens?: number;
  provider_config?: ProviderConfig;
  model_params?: ModelParams;
  stream?: boolean;
}

export interface ProviderConfig {
  api_key?: string;
  azure_endpoint?: string;
  api_version?: string;
  region?: string;
  aws_access_key_id?: string;
  aws_secret_access_key?: string;
  project_id?: string;
  location?: string;
  use_vertex?: boolean;
  endpoint?: string;
  timeout?: number;
  forward_params?: Record<string, any>;
}

export interface ModelParams {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  enable_thinking?: boolean;
  nothink?: boolean;
  [key: string]: any;
}

export interface ModerationResult {
  safe: boolean;
  status: 'safe' | 'unsafe' | 'blocked' | 'error';
  raw?: string;
  normalized_categories?: string[];
}

export interface ModerateResponse {
  prompt: string;
  route: 'primary' | 'oyster';
  prompt_moderation: ModerationResult;
  response: string;
  response_moderation: ModerationResult;
  keyword_flagged: boolean;
  protection_enabled: boolean;
  logs: string[];
  metadata: {
    primary_model_url: string;
    primary_model_name: string;
    primary_max_tokens: number;
    protection_enabled: boolean;
    provider: string;
  };
}

export interface StreamChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: 'assistant';
      content?: string;
    };
    finish_reason: string | null;
  }>;
}

// API 客户端
export class SafetyGuardClient {
  constructor(private baseURL: string = 'http://localhost:8000') {}

  async moderate(request: ModerateRequest): Promise<ModerateResponse> {
    const response = await fetch(`${this.baseURL}/moderate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  async *moderateStream(request: ModerateRequest): AsyncGenerator<string> {
    const response = await fetch(`${this.baseURL}/moderate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...request, stream: true })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') return;

          try {
            const parsed: StreamChunk = JSON.parse(data);
            const content = parsed.choices[0]?.delta?.content;
            if (content) yield content;
          } catch (e) {
            console.error('Parse error:', e);
          }
        }
      }
    }
  }

  async health(): Promise<{ status: string; service: string; version: string }> {
    const response = await fetch(`${this.baseURL}/health`);
    return response.json();
  }

  async reloadSensitive(): Promise<{ ok: boolean; word_count: number }> {
    const response = await fetch(`${this.baseURL}/reload_sensitive`, {
      method: 'POST'
    });
    return response.json();
  }
}
```

### 使用 TypeScript 客户端

```typescript
import { SafetyGuardClient } from './types';

const client = new SafetyGuardClient('http://localhost:8000');

// 非流式
async function example1() {
  const result = await client.moderate({
    prompt: '你好',
    provider: 'openai',
    protection_enabled: true
  });
  console.log(result.response);
}

// 流式
async function example2() {
  for await (const chunk of client.moderateStream({
    prompt: '写一首诗',
    stream_chunk_size: 8
  })) {
    process.stdout.write(chunk);
  }
}

// 健康检查
async function example3() {
  const health = await client.health();
  console.log('Service status:', health.status);
}
```

---

## 性能建议

1. **使用连接池**: 复用 HTTP 连接，避免频繁建立连接
2. **设置合理超时**: 推荐客户端超时时间 ≥ 120 秒
3. **流式优先**: 对于长文本生成，优先使用流式接口提升用户体验
4. **错误重试**: 对于超时和网络错误，可以实现指数退避重试
5. **请求去重**: 避免短时间内重复发送相同请求
6. **缓存健康检查**: 健康检查结果可以缓存 10-30 秒

## 安全建议

1. **HTTPS**: 生产环境务必使用 HTTPS
2. **敏感信息**: 不要在前端硬编码 API 密钥
3. **请求限流**: 在前端实现请求频率限制
4. **输入验证**: 前端也应进行基础的输入验证
5. **错误信息**: 不要向用户暴露详细的错误堆栈

## 相关链接

- [项目 README](README.md)
- [FastAPI 交互式文档](http://localhost:8000/docs)
- [FastAPI ReDoc 文档](http://localhost:8000/redoc)

---

**文档版本**: v0.2.0
**最后更新**: 2026-03-10
