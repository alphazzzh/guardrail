import os
import re
import math

from datasets import load_dataset

from .constants import SAFETY_LABELS, UNSAFE_LABELS, UNSAFE_LABEL_INDEX

_SAFETY_LABEL_TO_ID = {label.lower(): idx for idx, label in enumerate(SAFETY_LABELS)}
_UNSAFE_LABEL_TO_ID = {label.lower(): idx for idx, label in enumerate(UNSAFE_LABELS)}


def _infer_dataset_type(path):
    """根据文件扩展名推断数据集格式。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".json", ".jsonl"}:
        return "json"
    if ext == ".csv":
        return "csv"
    raise ValueError(f"Unsupported dataset format: {ext}")


def _normalize_text(value):
    """统一将输入文本转换为字符串。"""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _normalize_safety_label(value):
    """将 safety_label 统一为索引形式。"""
    if value is None:
        raise ValueError("safety_label is required")
    # 处理 NaN (常见于 CSV)
    if isinstance(value, float) and math.isnan(value):
        raise ValueError("safety_label cannot be NaN")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        idx = int(value)
        if 0 <= idx < len(SAFETY_LABELS):
            return idx
    
    label = str(value).strip().lower()
    idx = _SAFETY_LABEL_TO_ID.get(label)
    if idx is not None:
        return idx
    raise ValueError(f"Unknown safety_label: {value}")


def _normalize_unsafe_labels(value):
    """将 unsafe_labels 统一为字符串列表 (增强版：支持列表内含逗号拆分)。"""
    if value is None:
        return []
    # 处理 NaN
    if isinstance(value, float) and math.isnan(value):
        return []

    raw_labels = []
    # 1. 如果是字符串，直接分割
    if isinstance(value, str):
        raw_labels = re.split(r"[|,;，]", value)
    # 2. 如果是列表/元组，遍历每个元素并尝试分割
    elif isinstance(value, (list, tuple)):
        for v in value:
            if isinstance(v, str):
                # 即使在列表中，也允许字符串包含分隔符 (兼容 ["A,B"] 这种格式)
                raw_labels.extend(re.split(r"[|,;，]", v))
            else:
                raw_labels.append(v)
    # 3. 其他情况
    else:
        raw_labels = [value]
    
    labels = []
    for item in raw_labels:
        if item is None or (isinstance(item, float) and math.isnan(item)):
            continue
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            idx = int(item)
            if 0 <= idx < len(UNSAFE_LABELS):
                labels.append(UNSAFE_LABELS[idx])
                continue
        name = str(item).strip().lower()
        if not name:
            continue
        idx = _UNSAFE_LABEL_TO_ID.get(name)
        if idx is None:
            raise ValueError(f"Unknown unsafe label: {name}")
        labels.append(UNSAFE_LABELS[idx])
    
    # 去重并保持顺序 (可选)
    return list(dict.fromkeys(labels))


def _build_unsafe_vector(label_list):
    """将不安全标签列表转换为多标签向量。"""
    vec = [0] * len(UNSAFE_LABELS)
    for label in label_list:
        idx = _UNSAFE_LABEL_TO_ID.get(str(label).lower())
        if idx is None:
            raise ValueError(f"Unknown unsafe label: {label}")
        vec[idx] = 1
    return vec


def load_safety_datasets(
    train_file,
    validation_file=None,
    test_file=None,
    text_field="text",
    tokenizer=None,
    max_length=256,
):
    """读取安全数据集并进行标签规范化与分词。"""
    if tokenizer is None:
        raise ValueError("tokenizer is required")
    data_files = {"train": train_file}
    if validation_file:
        data_files["validation"] = validation_file
    if test_file:
        data_files["test"] = test_file

    dataset_type = _infer_dataset_type(train_file)
    for name, path in (("validation", validation_file), ("test", test_file)):
        if path and _infer_dataset_type(path) != dataset_type:
            raise ValueError(f"{name}_file must match train_file format: {dataset_type}")
    # 加载原始数据集
    raw_datasets = load_dataset(dataset_type, data_files=data_files)

    def add_labels(example):
        """标准化标签，并构建多标签向量。"""
        if text_field not in example:
            raise ValueError(f"Missing text field: {text_field}")
        text = _normalize_text(example[text_field])
        safety_id = _normalize_safety_label(example.get("safety_label"))
        unsafe_list = _normalize_unsafe_labels(example.get("unsafe_labels"))
        if safety_id != UNSAFE_LABEL_INDEX and unsafe_list:
            raise ValueError("unsafe_labels should be empty for non-unsafe samples")
        return {
            text_field: text,
            "labels": safety_id,
            "unsafe_labels": _build_unsafe_vector(unsafe_list),
        }

    # 生成带 labels / unsafe_labels 的数据集
    labeled_datasets = raw_datasets.map(add_labels)

    def tokenize_function(examples):
        """批量分词，输出模型可用的输入。"""
        return tokenizer(
            examples[text_field],
            truncation=True,
            max_length=max_length,
        )

    columns_to_remove = [
        col
        for col in labeled_datasets["train"].column_names
        if col not in {"labels", "unsafe_labels"}
    ]
    # 删除非训练字段，保留 labels / unsafe_labels
    tokenized_datasets = labeled_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
    )
    return tokenized_datasets
