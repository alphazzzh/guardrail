SAFETY_LABELS = ["安全", "争议", "不安全"]

UNSAFE_LABELS = [
    "违反社会主义核心价值观",
    "歧视性内容",
    "商业违法违规",
    "侵犯他人合法权益",
    "无法满足特定服务类型的安全需求",
]

UNSAFE_LABEL_INDEX = SAFETY_LABELS.index("不安全")  # 不安全在三分类中的索引


def get_label_config():
    """导出标签配置，供训练保存与推理加载。"""
    return {
        "safety_labels": SAFETY_LABELS,  # 三分类标签
        "unsafe_labels": UNSAFE_LABELS,  # 子类标签
        "unsafe_label_index": UNSAFE_LABEL_INDEX,  # 三分类中“不安全”索引
    }
