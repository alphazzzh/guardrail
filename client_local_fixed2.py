import argparse
import atexit
import json
import logging
import os
import random
import threading
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import ahocorasick
import httpx
from langgraph.graph import END, StateGraph
from openai import APIStatusError, APITimeoutError, OpenAI
from transformers import AutoTokenizer

# ================== 基础配置与日志 ==================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("safeguard_system")

openai_api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

# 服务地址配置
qwen3guard_base_url = os.getenv("QWEN3GUARD_BASE_URL", "http://192.192.140.3:8025/v1")
oyster_base_url = os.getenv("OYSTER_BASE_URL", "http://192.192.140.3:8026/v1")
primary_model_base_url = os.getenv("PRIMARY_MODEL_BASE_URL", "")

# 模型名称配置
model_moderation = os.getenv("QWEN3GUARD_MODEL", "model/Qwen3Guard-Gen-8B")
model_oyster = os.getenv("OYSTER_MODEL", "model/oyster")
primary_model_name = os.getenv("PRIMARY_MODEL_NAME", "model/primary")

# 超时与并发控制
PRIMARY_CONNECT_TIMEOUT = float(os.getenv("PRIMARY_CONNECT_TIMEOUT", "5.0"))
PRIMARY_READ_TIMEOUT = float(os.getenv("PRIMARY_READ_TIMEOUT", "30.0"))
PRIMARY_MAX_RETRIES = int(os.getenv("PRIMARY_MAX_RETRIES", "2"))
PRIMARY_BACKOFF_BASE = float(os.getenv("PRIMARY_BACKOFF_BASE", "0.6"))
PRIMARY_MAX_CONNECTIONS = int(os.getenv("PRIMARY_MAX_CONNECTIONS", "100"))
PRIMARY_MAX_KEEPALIVE = int(os.getenv("PRIMARY_MAX_KEEPALIVE", "20"))
PRIMARY_CLIENT_CACHE_SIZE = int(os.getenv("PRIMARY_CLIENT_CACHE_SIZE", "8"))
PRIMARY_CONTEXT_TOKEN_LIMIT = int(os.getenv("PRIMARY_CONTEXT_TOKEN_LIMIT", "8192"))
RESPONSE_GUARD_MAX_WORKERS = int(os.getenv("RESPONSE_GUARD_MAX_WORKERS", "5"))

# Guard 切块配置（token 级）
GUARD_MAX_CHUNK_TOKENS = int(os.getenv("GUARD_MAX_CHUNK_TOKENS", "512"))
GUARD_CHUNK_OVERLAP_TOKENS = int(os.getenv("GUARD_CHUNK_OVERLAP_TOKENS", "48"))

# ================== Tokenizer 路径自适应优化 ==================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TOK_PATH = str(BASE_DIR / "assets" / "tokenizer")
model_path = os.getenv("TOKENIZER_PATH", DEFAULT_TOK_PATH)
if not os.path.exists(model_path):
    model_path = "/home/zzh/oyster_1"  # fallback

try:
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    logger.info("成功加载 Tokenizer: %s", model_path)
except Exception as e:
    # tokenizer 不可用会直接影响切块/token 统计，建议直接 fail-fast
    raise SystemExit(f"Tokenizer 加载失败，请检查路径: {model_path}, Error: {e}")


# ================== 共享 HTTP 客户端池 ==================
# 1) 共享 httpx.Client 全局复用，避免重复建连
# 2) 淘汰/更新 OpenAI client 时，不要 close() ——因为底层共享同一个 httpx.Client，close 会误关连接池
# 3) 统一在进程退出时 close 所有共享 httpx.Client

_HTTPX_CLIENTS: Dict[Tuple[float, float, int, int, float, float], httpx.Client] = {}
_HTTPX_CLIENTS_LOCK = threading.Lock()


def get_shared_http_client(
    connect_timeout: float,
    read_timeout: float,
    max_connections: int = 100,
    max_keepalive: int = 20,
    write_timeout: float = 5.0,
    pool_timeout: Optional[float] = None,
) -> httpx.Client:
    pool_timeout = read_timeout if pool_timeout is None else pool_timeout
    key = (
        float(connect_timeout),
        float(read_timeout),
        int(max_connections),
        int(max_keepalive),
        float(write_timeout),
        float(pool_timeout),
    )
    with _HTTPX_CLIENTS_LOCK:
        cli = _HTTPX_CLIENTS.get(key)
        if cli is not None:
            return cli

        timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=write_timeout, pool=pool_timeout)
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_keepalive)
        cli = httpx.Client(timeout=timeout, limits=limits)
        _HTTPX_CLIENTS[key] = cli
        return cli


@atexit.register
def _close_shared_http_clients() -> None:
    with _HTTPX_CLIENTS_LOCK:
        clients = list(_HTTPX_CLIENTS.values())
        _HTTPX_CLIENTS.clear()
    for client in clients:
        try:
            client.close()
        except Exception:
            logger.debug("close shared httpx client failed", exc_info=True)


# 初始化固定的 OpenAI 客户端（共享 httpx.Client）
shared_http = get_shared_http_client(
    PRIMARY_CONNECT_TIMEOUT, PRIMARY_READ_TIMEOUT, PRIMARY_MAX_CONNECTIONS, PRIMARY_MAX_KEEPALIVE
)
client_qwen3guard = OpenAI(api_key=openai_api_key, base_url=qwen3guard_base_url, http_client=shared_http)
client_oyster = OpenAI(api_key=openai_api_key, base_url=oyster_base_url, http_client=shared_http)

_primary_clients: Dict[str, OpenAI] = {}
_primary_client_order: List[str] = []  # LRU 顺序（老 -> 新）


# ================== 状态定义 ==================
class UsageState(TypedDict, total=False):
    prompt: str
    keyword_flagged: bool
    keyword_hits: List[Dict[str, Any]]
    prompt_moderation: Dict[str, Any]
    prompt_safe: bool
    route: Literal["primary", "oyster"]
    response: str
    response_moderation: Dict[str, Any]
    logs: List[str]
    primary_model_url: Optional[str]
    primary_model_name: Optional[str]
    primary_max_tokens: int


class TestingState(TypedDict, total=False):
    dataset_path: str
    limit_per_category: int
    category_field: str
    prompt_field: str
    generator_url: Optional[str]
    generator_model: Optional[str]
    samples: Dict[str, List[Dict[str, Any]]]
    prompt_scores: Dict[str, Any]
    response_scores: Dict[str, Any]
    total_prompt_score: float
    total_response_score: float
    logs: List[str]


# ================== 敏感词扫描（AC 自动机，支持热更新） ==================
class SensitiveScanner:
    """
    AC 自动机热更新：reload() 构建新 automaton 后一次性替换 state（scan 无锁，热路径轻量）
    """

    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths
        self._state: Tuple[Optional[ahocorasick.Automaton], Dict[str, List[str]]] = (None, {})
        self.word_count = 0
        self._reload_lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        new_words: set[str] = set()
        new_sources: Dict[str, set[str]] = defaultdict(set)

        for path in self.file_paths:
            p = Path(path)
            if not p.exists():
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip()
                        if w:
                            new_words.add(w)
                            new_sources[w].add(p.name)
            except Exception as e:
                logger.warning("加载词库 %s 失败: %s", p.name, e)

        if not new_words:
            with self._reload_lock:
                self._state = (None, {})
                self.word_count = 0
            logger.warning("敏感词库为空：未加载到任何词条")
            return

        # 稳定排序，避免 set 导致 automaton 构建不确定
        words_sorted = sorted(new_words, key=lambda x: (len(x), x))
        automaton = ahocorasick.Automaton()
        for i, word in enumerate(words_sorted):
            automaton.add_word(word, (i, word))
        automaton.make_automaton()

        with self._reload_lock:
            self._state = (automaton, {k: sorted(list(v)) for k, v in new_sources.items()})
            self.word_count = len(words_sorted)

        logger.info("敏感词库已更新，总计 %d 条词汇", len(words_sorted))

    def scan(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        automaton, sources = self._state
        if automaton is None:
            return []
        hits: List[Dict[str, Any]] = []
        seen = set()
        for _, (_, word) in automaton.iter(text):
            if word in seen:
                continue
            seen.add(word)
            hits.append({"word": word, "sources": sources.get(word, [])})
        return hits


scanner = SensitiveScanner(
    [
        "/home/zzh/llms/LLMSafe/wordbags/政治类型.txt",
        "/home/zzh/llms/LLMSafe/wordbags/COVID-19词库.txt",
        "/home/zzh/llms/LLMSafe/wordbags/GFW补充词库.txt",
        "/home/zzh/llms/LLMSafe/wordbags/反动词库.txt",
        "/home/zzh/llms/LLMSafe/wordbags/网易前端过滤敏感词库.txt",
    ]
)


# ================== 通用工具与解析 ==================
def _ensure_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return str(x)

def coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

# Qwen3Guard 输出的九类风险标签（可按需扩展）
QWEN_CATEGORY_CANONICAL = [
    "Violent",
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "Personally Identifiable Information",
    "Suicide & Self-Harm",
    "Unethical Acts",
    "Politically Sensitive Topics",
    "Copyright Violation",
    "Jailbreak",
]


def _guard_status_from_text(raw: str, parsed: Optional[dict]) -> str:
    """
    尽量从 parsed.status / parsed.result / parsed.label 中取值；否则从 raw 中做容错判定
    """
    if isinstance(parsed, dict):
        for k in ("status", "result", "label"):
            v = parsed.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()

    s = raw.strip().lower()
    if "unsafe" in s or "block" in s or "reject" in s:
        return "unsafe"
    if "safe" in s or "pass" in s or "allow" in s:
        return "safe"
    return "unknown"


def _is_safe_status(status: str) -> bool:
    return _ensure_text(status).strip().lower() == "safe"


def _extract_guard_categories(parsed: Optional[dict], raw: str) -> List[str]:
    cats: List[str] = []

    # JSON 结构优先
    if isinstance(parsed, dict):
        for k in ("categories", "category", "labels", "label"):
            v = parsed.get(k)
            if isinstance(v, list):
                cats.extend([_ensure_text(x) for x in v if x])
            elif isinstance(v, str) and v.strip():
                cats.append(v.strip())

    # raw fallback：从 canonical 做子串匹配（不要求完全一致）
    raw_l = raw.lower()
    for c in QWEN_CATEGORY_CANONICAL:
        if c.lower() in raw_l:
            cats.append(c)

    # 去重保持顺序
    seen = set()
    out = []
    for c in cats:
        c = _ensure_text(c).strip()
        if not c:
            continue
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def call_qwen_guard(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    修复点：不要用 'unsafe' 子串做唯一判断；尽量兼容 JSON 输出 + raw fallback。
    """
    try:
        response = client_qwen3guard.chat.completions.create(model=model_moderation, messages=messages)
        raw = _ensure_text(response.choices[0].message.content).strip()
    except Exception as e:
        logger.error("Guard API Error: %s", e)
        return {"status": "error", "safe": False, "raw": _ensure_text(e), "normalized_categories": []}

    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None

    status = _guard_status_from_text(raw, parsed)
    categories = _extract_guard_categories(parsed, raw)
    normalized_categories = [c.strip().lower() for c in categories]

    return {
        "status": status,
        "safe": _is_safe_status(status),
        "raw": raw,
        "parsed": parsed,
        "categories": categories,
        "normalized_categories": normalized_categories,
    }


def _chunk_text_for_guard(text: str) -> List[str]:
    """
    修复点：用 token 级切块（而不是字符 800），避免中文/英文长度差异导致超限。
    """
    text = _ensure_text(text)
    if not text.strip():
        return []

    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= GUARD_MAX_CHUNK_TOKENS:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(ids):
        end = min(len(ids), start + GUARD_MAX_CHUNK_TOKENS)
        sub = ids[start:end]
        chunks.append(tok.decode(sub))
        if end == len(ids):
            break
        start = max(0, end - GUARD_CHUNK_OVERLAP_TOKENS)
    return chunks


def _get_primary_client(base_url: str) -> OpenAI:
    """
    修复点：LRU 维护 + 复用共享 httpx.Client；淘汰时不要 close（共享连接池）
    """
    key = (base_url or "").rstrip("/")
    if not key:
        raise ValueError("primary base_url is empty; set PRIMARY_MODEL_BASE_URL or pass --url")

    cli = _primary_clients.get(key)
    if cli is not None:
        # LRU move-to-end
        if key in _primary_client_order:
            _primary_client_order.remove(key)
        _primary_client_order.append(key)
        return cli

    cli = OpenAI(api_key=openai_api_key, base_url=key, http_client=shared_http)
    _primary_clients[key] = cli
    _primary_client_order.append(key)

    if len(_primary_client_order) > PRIMARY_CLIENT_CACHE_SIZE:
        old_key = _primary_client_order.pop(0)
        _primary_clients.pop(old_key, None)

    return cli


def call_primary_llm(prompt: str, base_url: str, model: str, max_tokens: int) -> str:
    client = _get_primary_client(base_url)

    last_exc: Optional[Exception] = None
    for attempt in range(PRIMARY_MAX_RETRIES + 1):
        input_budget = PRIMARY_CONTEXT_TOKEN_LIMIT - max_tokens - 100 
        input_ids = tok.encode(prompt, add_special_tokens=False)
        if len(input_ids) > input_budget:
            prompt = tok.decode(input_ids[-input_budget:]) # 保留末尾（最新的）信息
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return resp.choices[0].message.content or ""
        except (APITimeoutError, httpx.TimeoutException) as e:
            last_exc = e
        except APIStatusError as e:
            last_exc = e
            status = getattr(e, "status_code", None)
            if status and 400 <= int(status) < 500:
                break  # 4xx 不重试
        except Exception as e:
            last_exc = e

        if attempt < PRIMARY_MAX_RETRIES:
            backoff = PRIMARY_BACKOFF_BASE * (2 ** attempt) + random.random() * 0.2
            time.sleep(backoff)

    logger.error("Primary LLM Call failed after retries: %s", last_exc)
    return ""


# ================== 使用路径 Nodes ==================
def keyword_scan_node(state: UsageState) -> UsageState:
    hits = scanner.scan(state.get("prompt", "") or "")
    logs = list(state.get("logs") or [])
    logs.append(f"敏感词扫描完成: 命中={bool(hits)} hits={len(hits)}")
    return {"keyword_flagged": bool(hits), "keyword_hits": hits, "logs": logs}


def prompt_guard_node(state: UsageState) -> UsageState:
    logs = list(state.get("logs") or [])
    if state.get("keyword_flagged"):
        logs.append("敏感词命中：跳过 prompt guard，route=oyster")
        return {
            "prompt_safe": False,
            "route": "oyster",
            "prompt_moderation": {"status": "blocked", "safe": False, "raw": "blocked_by_sensitive_keywords"},
            "logs": logs,
        }

    res = call_qwen_guard([{"role": "user", "content": state.get("prompt", "") or ""}])
    logs.append(f"prompt guard done: status={res.get('status')}")
    return {
        "prompt_moderation": res,
        "prompt_safe": bool(res.get("safe", False)),
        "route": "primary" if res.get("safe", False) else "oyster",
        "logs": logs,
    }


def primary_generation_node(state: UsageState) -> UsageState:
    logs = list(state.get("logs") or [])
    ans = call_primary_llm(
        state.get("prompt", "") or "",
        state.get("primary_model_url") or primary_model_base_url,
        state.get("primary_model_name") or primary_model_name,
        int(state.get("primary_max_tokens") or 512),
    )
    logs.append("primary generation done")
    return {"response": ans, "logs": logs}


def _extract_answer(content: str) -> str:
    """删除被 <think>...</think> 包裹的内容"""
    if not content: return ""
    content = re.sub(r"<\s*think\b[^>]*>.*?<\s*/\s*think\s*>", "", content, flags=re.DOTALL | re.IGNORECASE)
    return content.strip()

def oyster_generation_node(state: UsageState) -> UsageState:
    logs = list(state.get("logs") or [])
    try:
        resp = client_oyster.chat.completions.create(
            model=model_oyster, messages=[{"role": "user", "content": state.get("prompt", "") or ""}]
        )
        logs.append("oyster generation done")
        text_out = resp.choices[0].message.content or ""
        return {"response": _extract_answer(text_out), "logs": logs}
    except Exception as e:
        logs.append(f"oyster generation failed: {e}")
        return {"response": f"服务繁忙，请稍后再试 ({e})", "logs": logs}


def response_guard_node(state: UsageState) -> UsageState:
    """
    1) 并行审核 fail-fast
    2) 任务异常 fail-closed（任何异常视为 unsafe/error）
    3) max_workers 动态 min(len(chunks), RESPONSE_GUARD_MAX_WORKERS)
    4) 切块改为 token 级，避免超限
    """
    logs = list(state.get("logs") or [])

    if state.get("keyword_flagged"):
        logs.append("敏感词命中：回答审核跳过")
        return {"response_moderation": {"safe": True, "status": "skip"}, "logs": logs}

    text = state.get("response", "") or ""
    chunks = _chunk_text_for_guard(text)
    if not chunks:
        logs.append("回答为空：审核直接 safe")
        return {"response_moderation": {"safe": True, "status": "safe", "raw": "empty_response"}, "logs": logs}

    max_workers = max(1, min(len(chunks), RESPONSE_GUARD_MAX_WORKERS))
    logs.append(f"开始并行审核 chunks={len(chunks)} workers={max_workers}")

    final_res: Dict[str, Any] = {"safe": True, "status": "safe"}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, c in enumerate(chunks, start=1):
            fut = executor.submit(
                call_qwen_guard,
                [{"role": "user", "content": state.get("prompt", "") or ""}, {"role": "assistant", "content": c}],
            )
            futures[fut] = i

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                logger.error("并行审核任务异常 idx=%s err=%s", idx, e)
                res = {"status": "error", "safe": False, "raw": _ensure_text(e), "normalized_categories": []}

            if not res.get("safe", True):
                res["chunk_index"] = idx
                final_res = res
                # 尝试取消其他未开始任务
                for f2 in futures:
                    if f2 is not fut:
                        f2.cancel()
                break

    logs.append(f"回答审核完成: status={final_res.get('status')} chunk={final_res.get('chunk_index', 'all')}")
    return {"response_moderation": final_res, "logs": logs}


# ================== 构建 Usage Graph ==================
usage_graph = StateGraph(UsageState)
usage_graph.add_node("keyword_scan", keyword_scan_node)
usage_graph.add_node("prompt_guard", prompt_guard_node)
usage_graph.add_node("primary_gen", primary_generation_node)
usage_graph.add_node("oyster_gen", oyster_generation_node)
usage_graph.add_node("response_guard", response_guard_node)

usage_graph.set_entry_point("keyword_scan")
usage_graph.add_edge("keyword_scan", "prompt_guard")
usage_graph.add_conditional_edges(
    "prompt_guard", lambda x: x.get("route", "oyster"), {"primary": "primary_gen", "oyster": "oyster_gen"}
)
usage_graph.add_edge("primary_gen", "response_guard")
usage_graph.add_edge("oyster_gen", "response_guard")
usage_graph.add_edge("response_guard", END)
usage_app = usage_graph.compile()


# ================== 测试路径 Nodes ==================
def load_dataset_node(state: TestingState) -> TestingState:
    path = Path(state["dataset_path"])
    samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    limit = int(state.get("limit_per_category") or 50)
    cat_field = state.get("category_field") or "category"
    prompt_field = state.get("prompt_field") or "prompt"

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cat = _ensure_text(item.get(cat_field, "default"))
            if len(samples[cat]) < limit:
                # 只保留必要字段（减少内存）
                samples[cat].append({cat_field: cat, prompt_field: item.get(prompt_field, "")})

    logs = list(state.get("logs") or [])
    logs.append(f"成功加载数据集：分类数={len(samples)} limit_per_category={limit}")
    return {"samples": dict(samples), "logs": logs}


def evaluate_prompts_node(state: TestingState) -> TestingState:
    samples = state.get("samples") or {}
    prompt_field = state.get("prompt_field") or "prompt"

    scores: Dict[str, Any] = {}
    total_correct = 0
    total_evaluated = 0

    for cat, records in samples.items():
        stats = {
            "total": len(records),
            "unsafe": 0,
            "safe": 0,
            "correct": 0,
            "flagged": 0,
            "accuracy": 0.0
        }
        
        if not records:
            scores[cat] = stats
            continue

        correct = 0
        unsafe_count = 0
        
        for r in records:
            p = _ensure_text(r.get(prompt_field, ""))
            guard = call_qwen_guard([{"role": "user", "content": p}])

            is_safe = guard.get("safe", True)
            if not is_safe:
                unsafe_count += 1
                correct += 1 # 假设数据集均为恶意样本
            
        stats["unsafe"] = unsafe_count
        stats["safe"] = len(records) - unsafe_count
        stats["flagged"] = unsafe_count
        stats["correct"] = correct
        stats["accuracy"] = correct / len(records)
        
        scores[cat] = stats
        total_correct += correct
        total_evaluated += len(records)

    total_prompt_score = 0.0
    if total_evaluated > 0:
        total_prompt_score = round(100.0 * total_correct / total_evaluated, 2)

    logs = list(state.get("logs") or [])
    logs.append(f"prompt 评测完成: total_score={total_prompt_score}")
    return {"prompt_scores": scores, "total_prompt_score": total_prompt_score, "logs": logs}


test_graph = StateGraph(TestingState)
test_graph.add_node("load_data", load_dataset_node)
test_graph.add_node("eval_prompts", evaluate_prompts_node)
test_graph.set_entry_point("load_data")
test_graph.add_edge("load_data", "eval_prompts")
test_graph.add_edge("eval_prompts", END)
test_app = test_graph.compile()


# ================== API Service Wrappers ==================

def run_usage_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    供 api_service 调用，封装 usage_app。
    保持与原 client_local.py 的返回值结构一致。
    """
    prompt = _ensure_text(payload.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt 不能为空")

    protection_enabled = coerce_bool(payload.get("protection_enabled"), True)
    
    primary_url = payload.get("primary_model_url") or primary_model_base_url
    primary_name = payload.get("primary_model_name") or primary_model_name
    max_tokens = int(payload.get("primary_max_tokens") or 512)

    # 构造 metadata，前端可能需要
    metadata = {
        "primary_model_url": primary_url,
        "primary_model_name": primary_name,
        "primary_max_tokens": max_tokens,
        "protection_enabled": protection_enabled,
    }

    if not protection_enabled:
        return {
            "prompt": prompt,
            "route": None,
            "prompt_moderation": None,
            "response": None,
            "response_moderation": None,
            "keyword_flagged": False,
            "protection_enabled": False,
            "logs": ["防护状态关闭，跳过防护流程"],
            "metadata": metadata,
        }

    init_state: UsageState = {
        "prompt": prompt,
        "logs": [],
        "primary_model_url": primary_url,
        "primary_model_name": primary_name,
        "primary_max_tokens": max_tokens,
    }

    result = usage_app.invoke(init_state)
    
    return {
        "prompt": prompt,
        "route": result.get("route"),
        "prompt_moderation": result.get("prompt_moderation"),
        "response": result.get("response", ""),
        "response_moderation": result.get("response_moderation"),
        "keyword_flagged": result.get("keyword_flagged", False),
        "protection_enabled": True,
        "logs": result.get("logs", []),
        "metadata": metadata,
    }


def run_testing_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    供 api_service 调用，封装 test_app。
    注意：此版本的 optimized client 尚未包含 evaluate_responses 节点，
    因此 response_scores 将返回空字典，以保持接口兼容性。
    """
    dataset_path = payload.get("dataset_path")
    if not dataset_path:
        raise ValueError("dataset_path 为必填字段")
        
    limit = int(payload.get("limit_per_category", 100))
    prompt_field = payload.get("prompt_field", "prompt")
    category_field = payload.get("category_field", "category")
    generator_url = payload.get("generator_url") or primary_model_base_url
    generator_model = payload.get("generator_model") or primary_model_name

    init_state: TestingState = {
        "dataset_path": dataset_path,
        "limit_per_category": limit,
        "category_field": category_field,
        "prompt_field": prompt_field,
        "generator_url": generator_url,
        "generator_model": generator_model,
        "logs": [],
    }

    result = test_app.invoke(init_state)

    return {
        "dataset_path": dataset_path,
        "limit_per_category": limit,
        "prompt_scores": result.get("prompt_scores", {}),
        "total_prompt_score": result.get("total_prompt_score", 0.0),
        "response_scores": result.get("response_scores", {}), # 兼容接口，默认为空
        "total_response_score": result.get("total_response_score", 0.0),
        "logs": result.get("logs", []),
        "generator": {
            "url": generator_url,
            "model": generator_model,
        },
    }


# ================== CLI 入口 ==================
def main():
    parser = argparse.ArgumentParser(description="LLM Safety Guard System")
    subparsers = parser.add_subparsers(dest="mode")

    use_p = subparsers.add_parser("use")
    use_p.add_argument("--prompt", required=True)
    use_p.add_argument("--url", default=primary_model_base_url)
    use_p.add_argument("--model", default=primary_model_name)
    use_p.add_argument("--max_tokens", type=int, default=512)

    test_p = subparsers.add_parser("test")
    test_p.add_argument("--path", required=True)
    test_p.add_argument("--limit", type=int, default=10)
    test_p.add_argument("--category_field", type=str, default="category")
    test_p.add_argument("--prompt_field", type=str, default="prompt")

    args = parser.parse_args()

    if args.mode == "use":
        payload = {
            "prompt": args.prompt,
            "primary_model_url": args.url,
            "primary_model_name": args.model,
            "primary_max_tokens": args.max_tokens,
            "protection_enabled": True
        }
        print(json.dumps(run_usage_pipeline(payload), ensure_ascii=False, indent=2))

    elif args.mode == "test":
        payload = {
            "dataset_path": args.path,
            "limit_per_category": args.limit,
            "category_field": args.category_field,
            "prompt_field": args.prompt_field,
        }
        print(json.dumps(run_testing_pipeline(payload), ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()