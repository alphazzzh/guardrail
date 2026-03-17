"""SafetyBERT 对外导出入口。"""

from .constants import SAFETY_LABELS, UNSAFE_LABELS, UNSAFE_LABEL_INDEX  # 标签常量
from .data import load_safety_datasets  # 数据加载与预处理
from .model import SafetyBertModel  # 原始分类模型
from .proto_model import BGEPrototypeModel, PrototypeMatcher  # BGE原型匹配模型

__all__ = [
    "SAFETY_LABELS",
    "UNSAFE_LABELS",
    "UNSAFE_LABEL_INDEX",
    "load_safety_datasets",
    "SafetyBertModel",
    "BGEPrototypeModel",
    "PrototypeMatcher",
]
