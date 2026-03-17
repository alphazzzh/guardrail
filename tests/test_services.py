"""Services API 基础测试
修复问题 H：所有被测函数均为 async def，必须使用 pytest-asyncio 异步运行
"""
import pytest
import pytest_asyncio


# 确保 pytest-asyncio 以 auto 模式运行（也可在 pytest.ini 中全局设置）
pytestmark = pytest.mark.asyncio


class TestRunUsagePipeline:
    """run_usage_pipeline 测试"""

    async def test_empty_prompt_raises_error(self):
        """空 prompt 应该抛出 ValueError"""
        from services import run_usage_pipeline

        with pytest.raises(ValueError, match="prompt 不能为空"):
            await run_usage_pipeline({"prompt": ""})

    async def test_protection_disabled_skips_guard(self):
        """protection_enabled=False 应该跳过防护，直接返回主模型响应"""
        from services import run_usage_pipeline

        result = await run_usage_pipeline({
            "prompt": "test",
            "protection_enabled": False,
        })

        # 修复问题 F：实际返回字段与代码一致
        assert "response" in result
        assert result.get("prompt_moderation", {}).get("status") == "skipped"


class TestCheckInputSafety:
    """check_input_safety 测试"""

    async def test_empty_prompt_raises_error(self):
        """空 prompt 应该抛出 ValueError"""
        from services import check_input_safety

        with pytest.raises(ValueError, match="prompt 不能为空"):
            await check_input_safety({"prompt": ""})

    async def test_returns_safety_result(self):
        """应该返回安全检查结果"""
        from services import check_input_safety

        result = await check_input_safety({"prompt": "你好，今天天气怎么样？"})

        assert "safe" in result
        assert "route" in result
        assert "keyword_flagged" in result
        assert "logs" in result


class TestCheckOutputSafety:
    """check_output_safety 测试"""

    async def test_empty_response_raises_error(self):
        """空 response 应该抛出 ValueError"""
        from services import check_output_safety

        with pytest.raises(ValueError, match="response 不能为空"):
            await check_output_safety({"prompt": "test", "response": ""})

    async def test_returns_safety_result(self):
        """应该返回安全检查结果"""
        from services import check_output_safety

        result = await check_output_safety({
            "prompt": "你好",
            "response": "你好！有什么可以帮助你的吗？",
        })

        # 修复问题 F：断言与 check_output_safety 实际返回字段一致
        assert "safe" in result
        assert "action" in result
        assert "logs" in result
