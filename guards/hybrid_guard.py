"""
智能漏斗式混合护栏
逻辑：
1. 优先调用 Qwen3Guard 进行初步判定。
2. 如果判定为 unsafe，先检查 Qwen 返回的类别是否能直接映射到业务类别 (从 config 加载)。
3. 如果 Qwen 类别无法直接映射，则调用 BERT+BGE 进行精细化分类。
"""
import logging
import asyncio
from typing import Any, Dict, List
from config.settings import QWEN_DIRECT_MAP

logger = logging.getLogger("safeguard_system")


class HybridGuard:
    def __init__(self, qwen_guard, safety_guard):
        self.qwen_guard = qwen_guard
        self.safety_guard = safety_guard

    async def async_check(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """异步智能分类审核"""
        # 1. 第一道防线：Qwen3Guard
        res_qwen = await self.qwen_guard.async_check(messages)

        # 如果 Qwen 本身出错（技术问题，如超长），直接返回错误，不继续审核
        if res_qwen.get("status") == "error":
            logger.warning("Qwen Guard encountered error: %s, skipping further checks", res_qwen.get("raw"))
            return res_qwen

        # Qwen 认为安全，直接返回
        if res_qwen.get("safe", True):
            return res_qwen

        # 2. Qwen 判定不安全，尝试直接从配置映射类别
        qwen_cats = res_qwen.get("categories", []) or res_qwen.get("normalized_categories", [])
        mapped_categories = set()

        for cat in qwen_cats:
            for k, v in QWEN_DIRECT_MAP.items():
                if k.lower() in cat.lower():
                    mapped_categories.add(v)

        # 映射成功，直接返回，不调用 BERT
        if mapped_categories:
            logger.info("Qwen categories %s directly mapped to %s", qwen_cats, mapped_categories)
            return {
                "safe": False,
                "status": "unsafe",
                "categories": sorted(list(mapped_categories)),
                "normalized_categories": sorted(list(mapped_categories)),
                "raw": f"Directly mapped from Qwen: {qwen_cats}",
                "scores": {},
            }

        # 3. 映射失败，调用 BERT+BGE 精细分类（兜底）
        logger.info("Qwen category mapping failed, triggering BERT+BGE classification...")
        res_bert = await self.safety_guard.async_check(messages)

        final_cats = res_bert.get("categories", ["敏感内容"])
        return {
            "safe": False,
            "status": "unsafe",
            "categories": final_cats,
            "normalized_categories": final_cats,
            "raw": f"Qwen flagged, BERT categorized: {final_cats}",
            "scores": res_bert.get("scores", {}),
        }

    def check(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """同步版本（仅为兼容旧调用方；新代码应始终使用 async_check）"""
        try:
            # get_event_loop() 在 Python 3.10+ 无运行循环线程中已废弃
            # 先尝试获取正在运行的循环；若无运行循环则用 asyncio.run()
            loop = asyncio.get_running_loop()
            # 已在事件循环中（如在协程里被同步调用）：提交到线程池避免死锁
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.async_check(messages))
                return future.result()
        except RuntimeError:
            # 没有正在运行的事件循环（如纯同步脚本调用），直接 run
            return asyncio.run(self.async_check(messages))
        except Exception:
            # 极速降级
            return self.qwen_guard.check(messages)
