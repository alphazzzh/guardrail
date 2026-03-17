# LLM Safety Guard - HTTP 状态码文档

## 成功状态码

| 状态码 | 说明 | 场景 |
|--------|------|------|
| `200` | 请求成功 | 所有正常的 API 请求 |

## 客户端错误状态码 (4xx)

| 状态码 | 说明 | 触发条件 | 返回格式 |
|--------|------|----------|----------|
| `400` | 请求参数错误 | - 缺少必填字段<br>- 参数类型错误<br>- 参数值不合法 | `{"detail": "错误详细信息"}` |
| `422` | 请求格式错误 | - JSON 格式错误<br>- Pydantic 字段校验失败<br>- 数据类型不匹配 | `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` |
| `499` | 客户端断开连接 | 客户端在服务器响应前主动断开连接 | `{"detail": "client disconnected"}` |

## 服务端错误状态码 (5xx)

| 状态码 | 说明 | 触发条件 | 返回格式 |
|--------|------|----------|----------|
| `500` | 服务器内部错误 | - 未捕获的异常<br>- 业务逻辑错误<br>- 依赖服务异常 | `{"detail": "审核失败：{错误信息}"}` |
| `504` | 请求超时 | - 请求处理时间超过 `REQUEST_TIMEOUT` (默认 120 秒)<br>- LLM 调用超时<br>- Guard 审核超时 | `{"detail": "请求超时"}` |

---

## 各端点的状态码使用

### 1. GET /health

**正常响应**:
- `200`: 服务正常运行

```json
{
  "status": "healthy",
  "service": "llm-safety-guard",
  "version": "0.1.0"
}
```

---

### 2. GET /ready

**正常响应**:
- `200`: 服务就绪，可接受请求

```json
{
  "status": "ready",
  "inflight": 50,
  "max_inflight": 50
}
```

---

### 3. POST /moderate

**正常响应**:
- `200`: 审核完成（无论是否通过审核）

**错误响应**:
| 状态码 | 场景 | 示例 |
|--------|------|------|
| `400` | `prompt` 为空 | `{"detail": "prompt 不能为空"}` |
| `422` | 请求体格式错误 | `{"detail": "请求体不是合法 JSON：..."}` |
| `422` | 字段类型错误 | `{"detail": [{"loc": ["body", "stream"], "msg": "value is not a valid boolean", "type": "type_error.bool"}]}` |
| `499` | 客户端断开 | `{"detail": "client disconnected"}` |
| `500` | 内部错误 | `{"detail": "审核失败：Connection refused"}` |
| `504` | 处理超时 | `{"detail": "请求超时"}` |

**流式响应特殊说明**:
- 当 `stream: true` 时，HTTP 状态码始终为 `200`
- 内容类型为 `text/event-stream`
- 错误不会通过状态码反映，而是在 SSE 流中断或无数据

---

### 4. POST /test_dataset

**正常响应**:
- `200`: 测试完成

**错误响应**:
| 状态码 | 场景 | 示例 |
|--------|------|------|
| `400` | `dataset_path` 为空 | `{"detail": "dataset_path 为必填字段"}` |
| `400` | 文件不存在 | `{"detail": "数据集文件不存在"}` |
| `422` | 请求格式错误 | `{"detail": "请求体必须是 JSON 对象"}` |
| `499` | 客户端断开 | `{"detail": "client disconnected"}` |
| `500` | 内部错误 | `{"detail": "测试失败：{错误信息}"}` |
| `504` | 处理超时 | `{"detail": "请求超时"}` |

---

### 5. POST /reload_sensitive

**正常响应**:
- `200`: 重载成功

```json
{
  "ok": true,
  "word_count": 15234
}
```

**错误响应**:
| 状态码 | 场景 | 示例 |
|--------|------|------|
| `499` | 客户端断开 | `{"detail": "client disconnected"}` |
| `500` | 重载失败 | `{"detail": "敏感词重载失败：{错误信息}"}` |
| `504` | 处理超时 | `{"detail": "请求超时"}` |

---

## HTTP Forward Provider 特有状态码

当使用 `provider: "http_forward"` 时，可能遇到以下额外的错误：

| 场景 | 返回状态码 | 说明 |
|------|-----------|------|
| 目标服务不可达 | `500` | 网络连接失败、DNS 解析失败 |
| 目标服务 404 | `500` | 目标 URL 不存在（会重试） |
| 目标服务 4xx | `500` | 目标服务拒绝请求（不会重试） |
| 目标服务 5xx | `500` | 目标服务内部错误（会重试） |
| 目标服务超时 | `504` 或 `500` | 根据是否超过 `REQUEST_TIMEOUT` |

---

## 错误处理最佳实践

### 前端错误处理示例

```javascript
async function handleAPIRequest(url, data) {
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    // 处理各种状态码
    switch (response.status) {
      case 200:
        return await response.json();

      case 400:
        const error400 = await response.json();
        throw new Error(`参数错误: ${error400.detail}`);

      case 422:
        const error422 = await response.json();
        const fieldErrors = Array.isArray(error422.detail)
          ? error422.detail.map(e => e.msg).join('; ')
          : error422.detail;
        throw new Error(`格式错误: ${fieldErrors}`);

      case 499:
        throw new Error('请求被取消');

      case 500:
        const error500 = await response.json();
        throw new Error(`服务器错误: ${error500.detail}`);

      case 504:
        throw new Error('请求超时，请稍后重试');

      default:
        throw new Error(`未知错误 (${response.status})`);
    }
  } catch (error) {
    if (error.name === 'TypeError') {
      // 网络错误
      throw new Error('网络连接失败，请检查网络设置');
    }
    throw error;
  }
}
```

### 重试策略建议

| 状态码 | 是否重试 | 重试次数 | 退避策略 |
|--------|---------|---------|---------|
| `400` | ❌ 否 | - | - |
| `422` | ❌ 否 | - | - |
| `499` | ✅ 是 | 1-2 次 | 立即或短延迟 |
| `500` | ✅ 是 | 2-3 次 | 指数退避 |
| `504` | ✅ 是 | 1-2 次 | 指数退避 |

### 超时配置建议

```javascript
// 根据不同场景配置超时时间
const timeouts = {
  health: 5000,        // 健康检查: 5 秒
  moderate: 120000,    // 普通审核: 120 秒
  moderateStream: 0,   // 流式响应: 不设超时（让服务端控制）
  testDataset: 300000, // 批量测试: 300 秒
  reload: 10000        // 重载: 10 秒
};

// 使用 AbortController 实现超时
async function fetchWithTimeout(url, options, timeout) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(id);
    return response;
  } catch (error) {
    clearTimeout(id);
    if (error.name === 'AbortError') {
      throw new Error('请求超时');
    }
    throw error;
  }
}
```

---

## 日志中的状态码

服务端日志格式:
```
2026-01-08 10:30:45,123 | INFO | api | request done | id=abc123 | path=/moderate | status=200 OK | ms=1234.5
```

日志字段说明:
- `id`: 请求 ID (来自 `X-Request-ID` 头或自动生成)
- `path`: 请求路径
- `status`: HTTP 状态码 + 状态描述
- `ms`: 处理耗时（毫秒）

---

## 环境变量影响

以下环境变量会影响状态码行为:

| 变量 | 默认值 | 影响 |
|------|--------|------|
| `REQUEST_TIMEOUT` | `120` | 超过此时间返回 `504` |
| `MAX_INFLIGHT` | `50` | 超过并发限制时请求会排队（不返回错误） |

---

## 常见问题排查

### Q: 为什么收到 `499` 错误？
A: 客户端在服务器完成响应前断开了连接。检查：
- 客户端超时设置是否太短
- 网络连接是否稳定
- 是否有负载均衡器提前断开连接

### Q: 为什么流式请求卡住不返回数据？
A: 流式请求始终返回 `200`，但可能因为：
- nginx 等代理缓冲了响应（设置 `X-Accel-Buffering: no`）
- 客户端没有正确处理 SSE 格式
- 防火墙干扰了长连接

### Q: 为什么 HTTP Forward 总是返回 `500`？
A: 检查日志中的详细错误信息：
- `full_url` 是否正确
- 目标服务是否可访问
- 目标服务是否返回标准格式

---

**文档版本**: v1.0
**最后更新**: 2026-01-08
