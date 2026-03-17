"""
压力测试脚本
测试项：
  1. 并发非流式请求 —— 验证信号量、超时、错误率
  2. 并发流式请求   —— 验证无信号量时的吞吐与稳定性
  3. 熔断器行为     —— 验证高失败率下熔断器是否正确触发

使用方法：
  # 启动服务后执行（默认连接 localhost:8000）
  python tests/stress_test.py --mode all --concurrency 20 --requests 100

  # 只测非流式
  python tests/stress_test.py --mode non_stream --concurrency 30 --requests 200

  # 只测流式
  python tests/stress_test.py --mode stream --concurrency 10 --requests 50

  # 只测熔断器（单元测试，不需要启动服务）
  python tests/stress_test.py --mode circuit_breaker
"""

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import httpx


# ─── 配置 ──────────────────────────────────────────────────────────────────────

BASE_URL = "http://192.192.140.3:20005"
# 主模型配置（按实际情况修改）
PRIMARY_MODEL_URL = "http://192.192.140.7:8000/v1"
PRIMARY_MODEL_NAME = "qwen-32b"
PROVIDER = "openai"   # 兼容 OpenAI API 格式的服务用 "openai"；纯 HTTP 转发用 "http_forward"

DEFAULT_PROMPT = "你好，请用简短的语言介绍一下量子计算。"
STREAM_PROMPT = "请列举三种常见的机器学习算法并简要说明。"

# 正常 prompt（安全）
SAFE_PROMPTS = [
    "量子计算的基本原理是什么？",
    "请介绍一下深度学习的发展历史。",
    "Python 和 Go 语言各有什么优缺点？",
    "解释一下 CAP 定理。",
    "什么是微服务架构？",
]


# ─── 统计容器 ──────────────────────────────────────────────────────────────────

@dataclass
class Stats:
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    status_codes: List[int] = field(default_factory=list)
    success_count: int = 0
    total: int = 0

    def record_success(self, latency: float, status: int):
        self.latencies.append(latency)
        self.status_codes.append(status)
        self.success_count += 1
        self.total += 1

    def record_error(self, err: str):
        self.errors.append(err)
        self.total += 1

    def report(self, title: str):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print(f"  总请求数  : {self.total}")
        print(f"  成功      : {self.success_count}")
        print(f"  失败      : {len(self.errors)}")
        print(f"  错误率    : {len(self.errors)/self.total*100:.1f}%")
        if self.latencies:
            print(f"  延迟 (ms):")
            print(f"    最小    : {min(self.latencies)*1000:.0f}")
            print(f"    最大    : {max(self.latencies)*1000:.0f}")
            print(f"    平均    : {statistics.mean(self.latencies)*1000:.0f}")
            print(f"    中位数  : {statistics.median(self.latencies)*1000:.0f}")
            if len(self.latencies) >= 2:
                print(f"    P95     : {_percentile(self.latencies, 95)*1000:.0f}")
                print(f"    P99     : {_percentile(self.latencies, 99)*1000:.0f}")
        if self.errors[:5]:
            print(f"  前5个错误:")
            for e in self.errors[:5]:
                print(f"    - {e}")
        print()


def _percentile(data: List[float], p: int) -> float:
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


# ─── 非流式并发测试 ────────────────────────────────────────────────────────────

async def _single_non_stream(client: httpx.AsyncClient, prompt: str, stats: Stats):
    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{BASE_URL}/moderate",
            json={
                "prompt": prompt,
                "primary_model_url": PRIMARY_MODEL_URL,
                "primary_model_name": PRIMARY_MODEL_NAME,
                "provider": PROVIDER,
            },
            timeout=60.0,
        )
        latency = time.perf_counter() - start
        if resp.status_code == 200:
            stats.record_success(latency, resp.status_code)
        else:
            stats.record_error(f"HTTP {resp.status_code}: {resp.text[:100]}")
    except httpx.TimeoutException:
        stats.record_error("timeout")
    except Exception as e:
        stats.record_error(str(e)[:100])


async def test_non_stream(concurrency: int, total_requests: int):
    stats = Stats()
    semaphore = asyncio.Semaphore(concurrency)
    prompts = [SAFE_PROMPTS[i % len(SAFE_PROMPTS)] for i in range(total_requests)]

    async def bounded(prompt):
        async with semaphore:
            await _single_non_stream(client, prompt, stats)

    print(f"\n[非流式] 并发={concurrency}, 总请求={total_requests} ...")
    wall_start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[bounded(p) for p in prompts])
    wall_time = time.perf_counter() - wall_start

    stats.report("非流式并发测试")
    print(f"  总耗时    : {wall_time:.2f}s")
    print(f"  QPS       : {total_requests / wall_time:.1f}")


# ─── 流式并发测试 ──────────────────────────────────────────────────────────────

async def _single_stream(client: httpx.AsyncClient, prompt: str, stats: Stats):
    start = time.perf_counter()
    try:
        async with client.stream(
            "POST",
            f"{BASE_URL}/moderate",
            json={
                "prompt": prompt,
                "stream": True,
                "primary_model_url": PRIMARY_MODEL_URL,
                "primary_model_name": PRIMARY_MODEL_NAME,
                "provider": PROVIDER,
            },
            timeout=60.0,
        ) as resp:
            if resp.status_code != 200:
                stats.record_error(f"HTTP {resp.status_code}")
                return
            chunk_count = 0
            async for _ in resp.aiter_lines():
                chunk_count += 1
            latency = time.perf_counter() - start
            stats.record_success(latency, resp.status_code)
    except httpx.TimeoutException:
        stats.record_error("timeout")
    except Exception as e:
        stats.record_error(str(e)[:100])


async def test_stream(concurrency: int, total_requests: int):
    stats = Stats()
    semaphore = asyncio.Semaphore(concurrency)
    prompts = [SAFE_PROMPTS[i % len(SAFE_PROMPTS)] for i in range(total_requests)]

    async def bounded(prompt):
        async with semaphore:
            await _single_stream(client, prompt, stats)

    print(f"\n[流式] 并发={concurrency}, 总请求={total_requests} ...")
    wall_start = time.perf_counter()
    # 流式需要更长超时的连接
    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0)) as client:
        await asyncio.gather(*[bounded(p) for p in prompts])
    wall_time = time.perf_counter() - wall_start

    stats.report("流式并发测试")
    print(f"  总耗时    : {wall_time:.2f}s")
    print(f"  QPS       : {total_requests / wall_time:.1f}")


# ─── 熔断器单元测试（不需要服务在线）─────────────────────────────────────────

async def test_circuit_breaker():
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpen, CircuitState

    print("\n[熔断器单元测试]")
    passed = 0
    failed = 0

    def check(name: str, condition: bool):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}")
            failed += 1

    # 1. 初始状态
    cb = CircuitBreaker(failure_threshold=3, timeout=1, half_open_max_calls=2, name="test")
    check("初始状态为 CLOSED", cb.state == CircuitState.CLOSED)

    # 2. 失败计数触发 OPEN
    for _ in range(3):
        try:
            await cb.async_call(lambda: _async_fail())
        except Exception:
            pass
    check("3次失败后变为 OPEN", cb.state == CircuitState.OPEN)

    # 3. OPEN 状态下请求直接被拒绝
    try:
        await cb.async_call(lambda: _async_ok())
        check("OPEN 状态拒绝请求", False)
    except CircuitBreakerOpen:
        check("OPEN 状态拒绝请求", True)

    # 4. 等待 timeout 后进入 HALF_OPEN
    await asyncio.sleep(1.1)
    try:
        await cb.async_call(lambda: _async_ok())
    except Exception:
        pass
    check("超时后进入 HALF_OPEN 并允许探测请求", cb.state == CircuitState.CLOSED)

    # 5. HALF_OPEN 下成功恢复为 CLOSED
    cb2 = CircuitBreaker(failure_threshold=2, timeout=1, half_open_max_calls=2, name="test2")
    for _ in range(2):
        try:
            await cb2.async_call(lambda: _async_fail())
        except Exception:
            pass
    await asyncio.sleep(1.1)
    await cb2.async_call(lambda: _async_ok())  # noqa
    check("HALF_OPEN 成功后恢复 CLOSED", cb2.state == CircuitState.CLOSED)

    # 6. HALF_OPEN 下再次失败重回 OPEN
    cb3 = CircuitBreaker(failure_threshold=2, timeout=1, half_open_max_calls=2, name="test3")
    for _ in range(2):
        try:
            await cb3.async_call(lambda: _async_fail())
        except Exception:
            pass
    await asyncio.sleep(1.1)
    try:
        await cb3.async_call(lambda: _async_fail())
    except Exception:
        pass
    check("HALF_OPEN 失败后重回 OPEN", cb3.state == CircuitState.OPEN)

    # 7. HALF_OPEN 超过 max_calls 被拒绝
    cb4 = CircuitBreaker(failure_threshold=1, timeout=1, half_open_max_calls=1, name="test4")
    try:
        await cb4.async_call(lambda: _async_fail())
    except Exception:
        pass
    await asyncio.sleep(1.1)
    # 第1次探测通过（进入 HALF_OPEN，调用成功回 CLOSED）
    # 这里测试的是：超额调用在 HALF_OPEN 时被拒
    cb4._state = CircuitState.HALF_OPEN  # type: ignore
    cb4._half_open_calls = 1  # type: ignore
    try:
        await cb4.async_call(lambda: _async_ok())
        check("HALF_OPEN 超额调用被拒绝", False)
    except CircuitBreakerOpen:
        check("HALF_OPEN 超额调用被拒绝", True)

    # 8. 并发下的线程安全（多协程同时触发失败）
    cb5 = CircuitBreaker(failure_threshold=5, timeout=10, name="concurrent")
    results = []

    async def concurrent_fail():
        try:
            await cb5.async_call(lambda: _async_fail())
        except Exception:
            results.append("err")

    await asyncio.gather(*[concurrent_fail() for _ in range(10)])
    check("并发失败下失败计数正确（≤10）", cb5._failure_count <= 10)  # type: ignore
    check("并发失败触发 OPEN", cb5.state == CircuitState.OPEN)

    print(f"\n  结果: {passed} 通过, {failed} 失败")


async def _async_ok():
    return "ok"


async def _async_fail():
    raise RuntimeError("模拟失败")


# ─── 信号量边界测试 ────────────────────────────────────────────────────────────

async def test_semaphore_boundary():
    """验证超出 MAX_INFLIGHT 时请求排队而不是被拒绝"""
    stats = Stats()
    # 发送 60 个并发（超过默认 MAX_INFLIGHT=50），验证不报错
    prompts = [DEFAULT_PROMPT] * 60
    payload_extra = {
        "primary_model_url": PRIMARY_MODEL_URL,
        "primary_model_name": PRIMARY_MODEL_NAME,
        "provider": PROVIDER,
    }

    print("\n[信号量边界测试] 发送 60 个并发（MAX_INFLIGHT=50）...")
    async with httpx.AsyncClient() as client:
        tasks = [_single_non_stream(client, p, stats) for p in prompts]
        await asyncio.gather(*tasks)

    stats.report("信号量边界测试（60并发）")
    print("  预期：所有请求都能完成（排队等待信号量），无 429 错误")


# ─── 主入口 ───────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="LLM Safety Guard 压力测试")
    parser.add_argument(
        "--mode",
        choices=["all", "non_stream", "stream", "circuit_breaker", "semaphore"],
        default="all",
    )
    parser.add_argument("--concurrency", type=int, default=20, help="并发数")
    parser.add_argument("--requests", type=int, default=100, help="总请求数")
    args = parser.parse_args()

    print("=" * 60)
    print("  LLM Safety Guard 压力测试")
    print(f"  目标: {BASE_URL}")
    print("=" * 60)

    if args.mode in ("all", "circuit_breaker"):
        await test_circuit_breaker()

    if args.mode in ("all", "non_stream"):
        await test_non_stream(args.concurrency, args.requests)

    if args.mode in ("all", "stream"):
        await test_stream(
            concurrency=min(args.concurrency, 10),  # 流式天然更重，降低并发
            total_requests=min(args.requests, 30),
        )

    if args.mode in ("all", "semaphore"):
        await test_semaphore_boundary()


if __name__ == "__main__":
    asyncio.run(main())
