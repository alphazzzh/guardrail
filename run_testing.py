#!/usr/bin/env python3
"""测试工作流运行脚本"""
import asyncio
import json
import sys
from pathlib import Path

from workflows.testing_workflow import test_app


async def run_test(
    dataset_path: str,
    test_mode: str = "raw_model",
    provider: str = "openai",
    primary_model_name: str = "gpt-4",
    scorer_url: str = "http://localhost:8000/v1",
    scorer_model: str = "Qwen2.5-14B-Instruct",
    limit_per_category: int = 50,
    output_dir: str = "./test_results",
):
    """运行测试工作流

    Args:
        dataset_path: 违规样本集路径（JSONL格式）
        test_mode: 测试模式，可选值：
            - "raw_model": 测试原始LLM（不带安全防护）
            - "safety_suite": 测试完整的安全套件（包括所有Guard检查）
        provider: Primary模型的provider（openai/azure/anthropic等）
        primary_model_name: Primary模型名称
        scorer_url: 14B LLM评分器的URL（OpenAI格式）
        scorer_model: 14B LLM评分器的模型名称
        limit_per_category: 每个分类的样本数量限制
        output_dir: 结果输出目录
    """
    # 构建初始状态
    init_state = {
        "dataset_path": dataset_path,
        "limit_per_category": limit_per_category,
        "category_field": "category",
        "prompt_field": "prompt",
        # 测试模式
        "test_mode": test_mode,
        # Primary模型配置
        "provider": provider,
        "primary_model_name": primary_model_name,
        "model_params": {
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "provider_config": {
            # 根据provider类型填写配置
            # 例如OpenAI: {"api_key": "sk-xxx", "base_url": "https://api.openai.com/v1"}
        },
        # 14B LLM评分器配置
        "scorer_url": scorer_url,
        "scorer_model": scorer_model,
        # 输出目录
        "output_dir": output_dir,
        "logs": [],
    }

    print(f"开始测试工作流...")
    print(f"测试模式: {test_mode}")
    print(f"数据集: {dataset_path}")
    print(f"Primary模型: {provider}/{primary_model_name}")
    print(f"评分器: {scorer_url}/{scorer_model}")
    print(f"每分类样本数: {limit_per_category}")
    print("-" * 60)

    # 运行工作流
    result = await test_app.ainvoke(init_state)

    # 打印日志
    print("\n执行日志:")
    for log in result.get("logs", []):
        print(f"  {log}")

    # 打印统计
    print("\n测试结果:")
    print(f"  Prompt评分: {result.get('total_prompt_score', 0)}")
    print(f"  Response评分: {result.get('total_response_score', 0)}")
    print(f"  分数分布: {result.get('response_scores', {}).get('distribution', {})}")

    return result


async def main():
    if len(sys.argv) < 2:
        print("用法: python run_testing.py <dataset_path> [options]")
        print("\n示例:")
        print("  # 测试原始模型（不带安全防护）")
        print("  python run_testing.py ./data/violations.jsonl --mode raw_model")
        print("\n  # 测试安全套件（带所有Guard检查）")
        print("  python run_testing.py ./data/violations.jsonl --mode safety_suite")
        print("\n  # 完整参数示例")
        print("  python run_testing.py ./data/violations.jsonl \\")
        print("    --mode safety_suite \\")
        print("    --provider openai \\")
        print("    --model gpt-4 \\")
        print("    --scorer-url http://localhost:8000/v1 \\")
        print("    --scorer-model Qwen2.5-14B-Instruct \\")
        print("    --limit 50 \\")
        print("    --output ./test_results")
        sys.exit(1)

    dataset_path = sys.argv[1]

    # 简单的参数解析（可以用argparse改进）
    kwargs = {
        "dataset_path": dataset_path,
    }

    # 解析可选参数
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--mode" and i + 1 < len(sys.argv):
            kwargs["test_mode"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--provider" and i + 1 < len(sys.argv):
            kwargs["provider"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            kwargs["primary_model_name"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--scorer-url" and i + 1 < len(sys.argv):
            kwargs["scorer_url"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--scorer-model" and i + 1 < len(sys.argv):
            kwargs["scorer_model"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--limit" and i + 1 < len(sys.argv):
            kwargs["limit_per_category"] = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            kwargs["output_dir"] = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    result = await run_test(**kwargs)
    print("\n测试完成！")


if __name__ == "__main__":
    asyncio.run(main())
