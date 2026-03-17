"""通用工具函数"""
import json
from typing import Any


def ensure_text(x: Any) -> str:
    """将任意类型转换为字符串"""
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
    """将任意类型转换为布尔值"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
