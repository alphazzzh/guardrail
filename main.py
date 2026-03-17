#!/usr/bin/env python3
"""LLM Safety Guard System - 主入口文件"""
import argparse
import asyncio
import json
import logging

from services import run_usage_pipeline, run_testing_pipeline
from config import PRIMARY_MODEL_BASE_URL, PRIMARY_MODEL_NAME

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("safeguard_system")


async def async_main():
    parser = argparse.ArgumentParser(description="LLM Safety Guard System")
    subparsers = parser.add_subparsers(dest="mode")

    # 使用模式
    use_p = subparsers.add_parser("use", help="使用防护系统进行文本生成")
    use_p.add_argument("--prompt", required=True, help="输入提示词")
    use_p.add_argument("--url", default=PRIMARY_MODEL_BASE_URL, help="主模型 URL")
    use_p.add_argument("--model", default=PRIMARY_MODEL_NAME, help="主模型名称")
    use_p.add_argument("--max_tokens", type=int, default=512, help="最大生成 token 数")
    use_p.add_argument("--provider", default="openai", help="Provider 名称 (openai/azure/anthropic/bedrock/gemini)")
    use_p.add_argument("--provider-config", type=str, help="Provider 配置 JSON 字符串")

    # 测试模式
    test_p = subparsers.add_parser("test", help="测试数据集评估")
    test_p.add_argument("--path", required=True, help="数据集路径")
    test_p.add_argument("--limit", type=int, default=10, help="每个分类的样本数量限制")
    test_p.add_argument("--category_field", type=str, default="category", help="分类字段名")
    test_p.add_argument("--prompt_field", type=str, default="prompt", help="提示词字段名")

    args = parser.parse_args()

    if args.mode == "use":
        provider_config = {}
        if args.provider_config:
            try:
                provider_config = json.loads(args.provider_config)
            except json.JSONDecodeError:
                logger.error("Invalid provider-config JSON")
                return

        payload = {
            "prompt": args.prompt,
            "primary_model_url": args.url,
            "primary_model_name": args.model,
            "primary_max_tokens": args.max_tokens,
            "protection_enabled": True,
            "provider": args.provider,
            "provider_config": provider_config,
        }
        # 修复问题 B：run_usage_pipeline 是 async def，必须 await
        result = await run_usage_pipeline(payload)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.mode == "test":
        payload = {
            "dataset_path": args.path,
            "limit_per_category": args.limit,
            "category_field": args.category_field,
            "prompt_field": args.prompt_field,
        }
        # 修复问题 B：run_testing_pipeline 是 async def，必须 await
        result = await run_testing_pipeline(payload)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(async_main())
