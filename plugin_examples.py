"""
插件式安全防护使用示例

展示如何在不同场景下使用三个独立的安全服务
"""
import json
from services import check_input_safety, generate_safe_response, check_output_safety


# ==================== 示例 1：基础插件式使用 ====================
def example_basic_plugin():
    """最基础的插件式使用"""
    print("\n========== 示例 1：基础插件式使用 ==========")

    user_prompt = "介绍一下人工智能的发展历史"

    # 步骤 1: 输入安全检查
    print("步骤 1: 检查输入安全性...")
    input_check = check_input_safety({"prompt": user_prompt})
    print(f"输入安全: {input_check['safe']}")
    print(f"建议路由: {input_check['route']}")

    if not input_check["safe"]:
        # 输入不安全，生成安全回答
        print("步骤 2: 输入不安全，使用 oyster 生成安全回答...")
        result = generate_safe_response({"prompt": user_prompt})
        final_response = result["response"]
        print(f"最终回答: {final_response}")
        return final_response

    # 输入安全，调用原有系统
    print("步骤 2: 输入安全，调用原有系统...")
    # 这里模拟原有系统的调用（可以包含 thinking、RAG 等）
    model_response = simulate_original_system(user_prompt)
    print(f"模型回答: {model_response}")

    # 步骤 3: 输出安全检查
    print("步骤 3: 检查输出安全性...")
    output_check = check_output_safety({
        "prompt": user_prompt,
        "response": model_response
    })
    print(f"输出安全: {output_check['safe']}")
    print(f"建议操作: {output_check['action']}")

    if output_check["safe"]:
        final_response = model_response
    else:
        final_response = "抱歉，无法回答这个问题"

    print(f"最终回答: {final_response}")
    return final_response


# ==================== 示例 2：处理不安全输入 ====================
def example_unsafe_input():
    """处理不安全输入的示例"""
    print("\n========== 示例 2：处理不安全输入 ==========")

    # 这里使用一个可能不安全的输入（根据实际情况修改）
    user_prompt = "这是一个测试输入"

    # 输入检查
    input_check = check_input_safety({"prompt": user_prompt})
    print(f"输入安全: {input_check['safe']}")
    print(f"敏感词命中: {input_check['keyword_flagged']}")
    print(f"命中的敏感词: {input_check['keyword_hits']}")

    if not input_check["safe"]:
        # 生成安全回答
        result = generate_safe_response({"prompt": user_prompt})
        print(f"安全回答: {result['response']}")
        return result["response"]

    return "输入安全，继续处理..."


# ==================== 示例 3：集成到 FastAPI ====================
def example_fastapi_integration():
    """
    FastAPI 集成示例（伪代码）

    这展示了如何在 FastAPI 中使用插件式架构
    """
    print("\n========== 示例 3：FastAPI 集成 ==========")

    # 伪代码示例
    code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services import check_input_safety, generate_safe_response, check_output_safety

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str
    use_thinking: bool = True
    use_rag: bool = True

class ChatResponse(BaseModel):
    response: str
    route: str
    input_safe: bool
    output_safe: bool
    logs: list

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口 - 插件式安全防护"""

    # 插件 1: 输入安全检查
    input_check = check_input_safety({"prompt": request.prompt})

    if not input_check["safe"]:
        # 输入不安全，使用 oyster 生成安全回答
        result = generate_safe_response({"prompt": request.prompt})
        return ChatResponse(
            response=result["response"],
            route="oyster",
            input_safe=False,
            output_safe=True,  # oyster 的输出默认安全
            logs=input_check["logs"] + result["logs"]
        )

    # 输入安全，继续原有流程
    # 这里可以包含任意复杂的逻辑
    model_response = await call_original_system(
        prompt=request.prompt,
        use_thinking=request.use_thinking,
        use_rag=request.use_rag
    )

    # 插件 2: 输出安全检查
    output_check = check_output_safety({
        "prompt": request.prompt,
        "response": model_response
    })

    if not output_check["safe"]:
        # 输出不安全，返回错误
        return ChatResponse(
            response="抱歉，无法回答这个问题",
            route="blocked",
            input_safe=True,
            output_safe=False,
            logs=input_check["logs"] + output_check["logs"]
        )

    # 一切正常，返回结果
    return ChatResponse(
        response=model_response,
        route="primary",
        input_safe=True,
        output_safe=True,
        logs=input_check["logs"] + output_check["logs"]
    )


async def call_original_system(prompt: str, use_thinking: bool, use_rag: bool) -> str:
    """
    原有系统的调用逻辑
    可以包含 thinking、RAG、多模型协作等任意复杂逻辑
    """
    result = prompt  # 初始化

    if use_thinking:
        # 执行 thinking 逻辑
        thinking_result = await do_thinking(prompt)
        result = f"[Thinking: {thinking_result}] {result}"

    if use_rag:
        # 执行 RAG 检索
        context = await retrieve_from_rag(prompt)
        result = f"[Context: {context}] {result}"

    # 调用模型
    model_output = await call_model(result)

    return model_output
'''

    print("FastAPI 集成代码示例:")
    print(code)


# ==================== 示例 4：流式输出的安全检查 ====================
def example_streaming_with_safety():
    """
    流式输出场景下的安全检查

    注意：流式输出时，需要先完整接收再检查，或者分块检查
    """
    print("\n========== 示例 4：流式输出的安全检查 ==========")

    code = '''
async def streaming_chat_with_safety(prompt: str):
    """流式输出 + 安全检查"""

    # 输入检查
    input_check = check_input_safety({"prompt": prompt})

    if not input_check["safe"]:
        # 输入不安全，返回安全回答（非流式）
        result = generate_safe_response({"prompt": prompt})
        yield result["response"]
        return

    # 输入安全，流式生成
    # 方案 1: 先完整接收，再检查（推荐）
    full_response = ""
    async for chunk in stream_from_model(prompt):
        full_response += chunk
        yield chunk  # 实时返回给用户

    # 流式结束后，检查完整输出
    output_check = check_output_safety({
        "prompt": prompt,
        "response": full_response
    })

    if not output_check["safe"]:
        # 输出不安全，发送警告
        yield "\\n\\n[警告：部分内容可能不安全]"


    # 方案 2: 分块检查（更安全，但可能中断流式）
    buffer = ""
    async for chunk in stream_from_model(prompt):
        buffer += chunk

        # 每 N 个字符检查一次
        if len(buffer) >= 100:
            output_check = check_output_safety({
                "prompt": prompt,
                "response": buffer
            })

            if not output_check["safe"]:
                # 立即中断流式输出
                yield "\\n\\n[内容被拦截]"
                return

        yield chunk
'''

    print("流式输出安全检查示例:")
    print(code)


# ==================== 示例 5：批量处理 ====================
def example_batch_processing():
    """批量处理多个请求"""
    print("\n========== 示例 5：批量处理 ==========")

    prompts = [
        "介绍一下人工智能",
        "什么是机器学习",
        "深度学习的应用",
    ]

    results = []

    for prompt in prompts:
        print(f"\n处理: {prompt}")

        # 输入检查
        input_check = check_input_safety({"prompt": prompt})

        if not input_check["safe"]:
            result = generate_safe_response({"prompt": prompt})
            results.append({
                "prompt": prompt,
                "response": result["response"],
                "route": "oyster"
            })
            continue

        # 调用原有系统
        model_response = simulate_original_system(prompt)

        # 输出检查
        output_check = check_output_safety({
            "prompt": prompt,
            "response": model_response
        })

        if output_check["safe"]:
            results.append({
                "prompt": prompt,
                "response": model_response,
                "route": "primary"
            })
        else:
            results.append({
                "prompt": prompt,
                "response": "无法回答",
                "route": "blocked"
            })

    print("\n批量处理结果:")
    print(json.dumps(results, ensure_ascii=False, indent=2))


# ==================== 示例 6：与现有系统集成 ====================
def example_integration_with_existing_system():
    """与现有系统集成的完整示例"""
    print("\n========== 示例 6：与现有系统集成 ==========")

    class ExistingChatSystem:
        """模拟现有的聊天系统"""

        def __init__(self):
            self.history = []

        def chat(self, user_input: str) -> str:
            """
            原有的聊天逻辑
            可能包含复杂的 thinking、RAG、多轮对话等
            """
            # 模拟复杂的处理流程
            thinking = self._do_thinking(user_input)
            context = self._retrieve_context(user_input)
            response = self._generate_response(user_input, thinking, context)

            self.history.append({"user": user_input, "assistant": response})
            return response

        def _do_thinking(self, input_text: str) -> str:
            return f"[Thinking about: {input_text}]"

        def _retrieve_context(self, input_text: str) -> str:
            return f"[Retrieved context for: {input_text}]"

        def _generate_response(self, input_text: str, thinking: str, context: str) -> str:
            return f"Based on {thinking} and {context}, here's the answer to '{input_text}'"

    class SafeChatSystem(ExistingChatSystem):
        """添加安全防护的聊天系统（插件式）"""

        def chat(self, user_input: str) -> str:
            """重写 chat 方法，添加安全检查"""

            # 插件 1: 输入安全检查
            input_check = check_input_safety({"prompt": user_input})

            if not input_check["safe"]:
                # 输入不安全，使用安全回答
                result = generate_safe_response({"prompt": user_input})
                return result["response"]

            # 输入安全，调用原有逻辑（保留所有功能）
            response = super().chat(user_input)

            # 插件 2: 输出安全检查
            output_check = check_output_safety({
                "prompt": user_input,
                "response": response
            })

            if not output_check["safe"]:
                return "抱歉，无法回答这个问题"

            return response

    # 使用示例
    print("创建安全聊天系统...")
    safe_chat = SafeChatSystem()

    print("\n测试对话 1:")
    response1 = safe_chat.chat("介绍一下人工智能")
    print(f"回答: {response1}")

    print("\n测试对话 2:")
    response2 = safe_chat.chat("什么是机器学习")
    print(f"回答: {response2}")


# ==================== 工具函数 ====================
def simulate_original_system(prompt: str) -> str:
    """模拟原有系统的调用（包含 thinking、RAG 等）"""
    # 这里模拟复杂的处理流程
    thinking = f"[Thinking: 分析问题 '{prompt}']"
    rag_context = f"[RAG: 检索到相关信息]"
    model_output = f"基于 {thinking} 和 {rag_context}，这是对 '{prompt}' 的回答"
    return model_output


# ==================== 主函数 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("插件式安全防护使用示例")
    print("=" * 60)

    # 运行示例（根据需要取消注释）
    example_basic_plugin()
    # example_unsafe_input()
    # example_fastapi_integration()
    # example_streaming_with_safety()
    # example_batch_processing()
    # example_integration_with_existing_system()

    print("\n" + "=" * 60)
    print("所有示例运行完成")
    print("=" * 60)
