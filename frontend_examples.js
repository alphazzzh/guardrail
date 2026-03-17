// ============================================
// 前端接口使用示例
// ============================================

/**
 * 方案 1：保持向后兼容（推荐）
 * 前端不需要任何修改，继续使用原有字段
 */
function callAPI_Compatible(prompt) {
  return fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: prompt,
      protection_enabled: true,
      primary_model_url: "https://api.openai.com/v1",
      primary_model_name: "gpt-4",
      primary_max_tokens: 512
      // 不传 provider，默认使用 OpenAI
    })
  });
}


/**
 * 方案 2：动态选择 Provider
 * 让用户在前端选择使用哪个 AI 服务商
 */
function callAPI_WithProvider(prompt, selectedProvider) {
  const basePayload = {
    prompt: prompt,
    protection_enabled: true,
    primary_max_tokens: 512
  };

  // 根据不同 provider 构建不同的配置
  let payload = { ...basePayload };

  switch (selectedProvider) {
    case 'openai':
      payload.provider = 'openai';
      payload.primary_model_url = 'https://api.openai.com/v1';
      payload.primary_model_name = 'gpt-4';
      // API key 从环境变量读取，或者在 provider_config 中传入
      break;

    case 'azure':
      payload.provider = 'azure';
      payload.primary_model_name = 'my-gpt4-deployment';
      payload.provider_config = {
        api_key: getAzureAPIKey(),  // 从前端配置获取
        azure_endpoint: 'https://your-resource.openai.azure.com',
        api_version: '2024-02-15-preview'
      };
      break;

    case 'anthropic':
      payload.provider = 'anthropic';
      payload.primary_model_name = 'claude-3-5-sonnet-20241022';
      payload.provider_config = {
        api_key: getAnthropicAPIKey()
      };
      break;

    case 'bedrock':
      payload.provider = 'bedrock';
      payload.primary_model_name = 'anthropic.claude-3-sonnet-20240229-v1:0';
      payload.provider_config = {
        region: 'us-east-1',
        aws_access_key_id: getAWSKeyId(),
        aws_secret_access_key: getAWSSecret()
      };
      break;

    case 'gemini':
      payload.provider = 'gemini';
      payload.primary_model_name = 'gemini-1.5-pro';
      payload.provider_config = {
        api_key: getGoogleAPIKey()
      };
      break;
  }

  return fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
}


/**
 * 方案 3：从配置文件读取 Provider 配置
 * 适合在后端统一管理 API keys
 */
function callAPI_SimpleProvider(prompt, providerName = 'openai') {
  return fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: prompt,
      protection_enabled: true,
      primary_model_name: getModelName(providerName),
      primary_max_tokens: 512,
      provider: providerName
      // provider_config 不传，使用后端环境变量中的配置
    })
  });
}


/**
 * 方案 4：完整的类型定义（TypeScript）
 */
interface ChatRequest {
  // 必填字段
  prompt: string;

  // 原有可选字段（向后兼容）
  protection_enabled?: boolean;
  primary_model_url?: string;
  primary_model_name?: string;
  primary_max_tokens?: number;

  // 新增可选字段
  provider?: 'openai' | 'azure' | 'anthropic' | 'bedrock' | 'gemini' | 'gemini-vertex';
  provider_config?: {
    // OpenAI
    api_key?: string;

    // Azure
    azure_endpoint?: string;
    api_version?: string;

    // AWS Bedrock
    region?: string;
    aws_access_key_id?: string;
    aws_secret_access_key?: string;

    // Google Gemini
    project_id?: string;
    location?: string;
  };
}

async function callAPI_TypeScript(request: ChatRequest): Promise<any> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
  return response.json();
}


/**
 * 使用示例
 */
async function examples() {
  // 示例 1：原有方式，不需要任何修改
  await callAPI_Compatible("介绍一下人工智能");

  // 示例 2：使用 Anthropic
  await callAPI_WithProvider("介绍一下人工智能", 'anthropic');

  // 示例 3：简化调用
  await callAPI_SimpleProvider("介绍一下人工智能", 'gemini');

  // 示例 4：TypeScript 类型安全
  await callAPI_TypeScript({
    prompt: "介绍一下人工智能",
    protection_enabled: true,
    provider: "anthropic",
    primary_model_name: "claude-3-5-sonnet-20241022",
    provider_config: {
      api_key: "sk-ant-xxx"
    }
  });
}


/**
 * 工具函数示例
 */
function getModelName(provider) {
  const modelMap = {
    'openai': 'gpt-4',
    'azure': 'my-gpt4-deployment',
    'anthropic': 'claude-3-5-sonnet-20241022',
    'bedrock': 'anthropic.claude-3-sonnet-20240229-v1:0',
    'gemini': 'gemini-1.5-pro'
  };
  return modelMap[provider] || 'gpt-4';
}

function getAzureAPIKey() {
  // 从前端配置、localStorage 或用户输入获取
  return localStorage.getItem('azure_api_key') || '';
}

function getAnthropicAPIKey() {
  return localStorage.getItem('anthropic_api_key') || '';
}

function getAWSKeyId() {
  return localStorage.getItem('aws_key_id') || '';
}

function getAWSSecret() {
  return localStorage.getItem('aws_secret') || '';
}

function getGoogleAPIKey() {
  return localStorage.getItem('google_api_key') || '';
}
