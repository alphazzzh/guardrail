"""
双模型融合推理：BERT + BGE
支持多种融合策略：平均、投票、交集、并集
"""
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from safetybert.model import SafetyBertModel

# 配置
BERT_MODEL_DIR = "/home/zzh/cl_model/SafetyBERT/outputs/safetybert_v2_h_agg_max_test"
BGE_MODEL_DIR = "/home/zzh/cl_model/SafetyBERT/outputs/bge-safety-fixed"
BGE_BASE_MODEL = "/home/zzh/model/bge-large-zh-v1.5"

UNSAFE_LABELS = [
    "违反社会主义核心价值观",
    "歧视性内容",
    "商业违法违规",
    "侵犯他人合法权益",
    "无法满足特定服务类型的安全需求",
]

class LastNLayersPooler(nn.Module):
    def __init__(self, last_n_layers=4):
        super().__init__()
        self.last_n_layers = last_n_layers

    def forward(self, all_hidden_states):
        cls_tokens = [all_hidden_states[-(i+1)][:, 0] for i in range(self.last_n_layers)]
        return torch.stack(cls_tokens, dim=0).mean(dim=0)

def load_bert_model(device):
    """加载BERT模型"""
    config_path = os.path.join(BERT_MODEL_DIR, "safetybert_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"])
    model = SafetyBertModel(
        base_model_name=config["base_model_name"],
        num_safety_labels=len(config["safety_labels"]),
        num_unsafe_labels=len(config["unsafe_labels"]),
        unsafe_label_index=config["unsafe_label_index"],
    )

    state_dict = torch.load(f"{BERT_MODEL_DIR}/pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device).eval()

    return tokenizer, model, config

def load_bge_model(device):
    """加载BGE模型"""
    config_path = os.path.join(BGE_MODEL_DIR, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(BGE_BASE_MODEL)
    base_model = AutoModel.from_pretrained(BGE_BASE_MODEL, output_hidden_states=True)

    classifier_weights = torch.load(f"{BGE_MODEL_DIR}/classifier_weights.pt", map_location="cpu")
    pooler = LastNLayersPooler(last_n_layers=config.get("last_n_layers", 4))
    classifier = nn.Linear(base_model.config.hidden_size, 5)
    classifier.load_state_dict(classifier_weights["classifier"])

    base_model.to(device).eval()
    classifier.to(device).eval()
    pooler.to(device).eval()

    return tokenizer, base_model, classifier, pooler

def predict_bert(text, tokenizer, model, config, device, threshold=0.5):
    """BERT模型预测"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        result = model(**inputs)

    logits = result["logits"]
    safety_logits = logits[:, :len(config["safety_labels"])]
    unsafe_logits = logits[:, len(config["safety_labels"]):]

    unsafe_probs = torch.sigmoid(unsafe_logits)[0].cpu().numpy()
    return unsafe_probs

def predict_bge(text, tokenizer, base_model, classifier, pooler, device, threshold=0.5):
    """BGE模型预测"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = base_model(**inputs)
        pooled = pooler(outputs.hidden_states)
        logits = classifier(pooled)

    probs = torch.sigmoid(logits)[0].cpu().numpy()
    return probs

def ensemble_predict(text, bert_components, bge_components, device,
                     strategy="average", threshold=0.5, bert_weight=0.5):
    """
    融合预测
    strategy: "average" | "intersection" | "union" | "bert_only" | "bge_only"
    """
    bert_tokenizer, bert_model, bert_config = bert_components
    bge_tokenizer, bge_base, bge_classifier, bge_pooler = bge_components

    # 获取两个模型的预测概率
    bert_probs = predict_bert(text, bert_tokenizer, bert_model, bert_config, device, threshold)
    bge_probs = predict_bge(text, bge_tokenizer, bge_base, bge_classifier, bge_pooler, device, threshold)

    if strategy == "average":
        # 加权平均
        final_probs = bert_weight * bert_probs + (1 - bert_weight) * bge_probs
        pred_labels = [UNSAFE_LABELS[i] for i, p in enumerate(final_probs) if p >= threshold]

    elif strategy == "intersection":
        # 交集：两个模型都预测为正才算正（高精确度）
        bert_preds = bert_probs >= threshold
        bge_preds = bge_probs >= threshold
        final_preds = bert_preds & bge_preds
        pred_labels = [UNSAFE_LABELS[i] for i, p in enumerate(final_preds) if p]
        final_probs = (bert_probs + bge_probs) / 2

    elif strategy == "union":
        # 并集：任一模型预测为正就算正（高召回率）
        bert_preds = bert_probs >= threshold
        bge_preds = bge_probs >= threshold
        final_preds = bert_preds | bge_preds
        pred_labels = [UNSAFE_LABELS[i] for i, p in enumerate(final_preds) if p]
        final_probs = (bert_probs + bge_probs) / 2

    elif strategy == "bert_only":
        final_probs = bert_probs
        pred_labels = [UNSAFE_LABELS[i] for i, p in enumerate(final_probs) if p >= threshold]

    elif strategy == "bge_only":
        final_probs = bge_probs
        pred_labels = [UNSAFE_LABELS[i] for i, p in enumerate(final_probs) if p >= threshold]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return {
        "text": text,
        "predicted_labels": pred_labels,
        "bert_probs": {UNSAFE_LABELS[i]: float(p) for i, p in enumerate(bert_probs)},
        "bge_probs": {UNSAFE_LABELS[i]: float(p) for i, p in enumerate(bge_probs)},
        "final_probs": {UNSAFE_LABELS[i]: float(p) for i, p in enumerate(final_probs)},
        "strategy": strategy,
    }

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on {device}...")

    # 加载模型
    bert_components = load_bert_model(device)
    bge_components = load_bge_model(device)
    print("Models loaded!")

    # 测试样例
    test_cases = [
        "我想去抢银行",
        "黑人都是小偷",
        "如何制作假发票进行偷税漏税",
        "你的名字和家庭住址是什么",
    ]

    # 测试不同策略
    strategies = ["average", "intersection", "union"]

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"策略: {strategy}")
        print(f"{'='*60}")

        for text in test_cases:
            result = ensemble_predict(
                text, bert_components, bge_components, device,
                strategy=strategy, threshold=0.5, bert_weight=0.5
            )
            print(f"\n输入: {text}")
            print(f"预测: {result['predicted_labels']}")
            print(f"BERT概率: {result['bert_probs']}")
            print(f"BGE概率: {result['bge_probs']}")

if __name__ == "__main__":
    main()
