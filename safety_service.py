"""
BERT + BGE 协作推理安全服务 (FastAPI 版)
支持策略：Intersection (方案2), Union (方案3)

"""
import json
import asyncio
import torch
import torch.nn as nn
import numpy as np
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from transformers import AutoModel, AutoTokenizer

# --- 模型定义 ---
class LastNLayersPooler(nn.Module):
    def __init__(self, last_n_layers=4):
        super().__init__()
        self.last_n_layers = last_n_layers

    def forward(self, all_hidden_states):
        cls_tokens = [all_hidden_states[-(i + 1)][:, 0] for i in range(self.last_n_layers)]
        return torch.stack(cls_tokens, dim=0).mean(dim=0)


class SafetyBertModel(nn.Module):
    def __init__(self, base_model_name, num_safety_labels, num_unsafe_labels, unsafe_label_index, dropout=0.1):
        super().__init__()
        self.base = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.base.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.safety_classifier = nn.Linear(hidden_size, num_safety_labels)
        self.unsafe_classifier = nn.Linear(hidden_size, num_unsafe_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return {
            "logits": torch.cat(
                [self.safety_classifier(pooled), self.unsafe_classifier(pooled)], dim=-1
            )
        }


# --- 全局路径配置 ---
BERT_DIR = "/home/zzh/cl_model/SafetyBERT/outputs/safetybert_v2_h_agg_max_test"
BGE_DIR = "/home/zzh/cl_model/SafetyBERT/outputs/bge-safety-fixed"
BGE_BASE = "/home/zzh/model/bge-large-zh-v1.5"
UNSAFE_LABELS = [
    "违反社会主义核心价值观",
    "歧视性内容",
    "商业违法违规",
    "侵犯他人合法权益",
    "无法满足特定服务类型的安全需求",
]


class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # BERT
        self.bert_tok = None
        self.bert_model = None
        self.num_safety = 2  # 默认值，load() 后从配置文件动态覆盖
        # BGE
        self.bge_tok = None
        self.bge_base = None
        self.bge_pooler = None
        self.bge_clf = None

    def load(self):
        """同步加载所有模型（由 run_in_executor 在线程池中调用，不阻塞事件循环）"""
        # --- BERT ---
        with open(f"{BERT_DIR}/safetybert_config.json", "r") as f:
            c = json.load(f)
        # 修复问题 12：从配置文件动态读取 safety_labels 数量，避免硬编码切片
        self.num_safety = len(c["safety_labels"])
        self.bert_tok = AutoTokenizer.from_pretrained(c["base_model_name"])
        self.bert_model = SafetyBertModel(
            c["base_model_name"],
            self.num_safety,
            len(c["unsafe_labels"]),
            c["unsafe_label_index"],
        )
        self.bert_model.load_state_dict(
            torch.load(f"{BERT_DIR}/pytorch_model.bin", map_location="cpu")
        )
        self.bert_model.to(self.device).eval()

        self.bge_tok = AutoTokenizer.from_pretrained(BGE_BASE)
        self.bge_base = AutoModel.from_pretrained(BGE_BASE, output_hidden_states=True)
        self.bge_pooler = LastNLayersPooler(4).to(self.device)
        self.bge_clf = nn.Linear(self.bge_base.config.hidden_size, len(UNSAFE_LABELS))
        cw = torch.load(f"{BGE_DIR}/classifier_weights.pt", map_location="cpu")
        self.bge_clf.load_state_dict(cw["classifier"])
        self.bge_base.to(self.device).eval()
        self.bge_clf.to(self.device).eval()

        logger.info("Models loaded on %s", self.device)


manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 修复问题 16：在事件循环线程池中异步执行同步模型加载，不阻塞事件循环
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, manager.load)
    yield


logger = logging.getLogger("safeguard_system")

app = FastAPI(title="Safety BERT+BGE Service", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str
    strategy: str = "intersection"
    threshold: float = 0.5


@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        with torch.no_grad():
            # --- BERT 推理 ---
            b_in = manager.bert_tok(
                req.text, return_tensors="pt", truncation=True, max_length=256
            ).to(manager.device)
            b_logits = manager.bert_model(**b_in)["logits"]
            # 修复问题 12：动态切片，不再硬编码 [:, 2:]
            b_probs = torch.sigmoid(b_logits[:, manager.num_safety:])[0].cpu().numpy()

            # --- BGE 推理（修复问题 I：恢复完整推理逻辑）---
            bg_in = manager.bge_tok(
                req.text, return_tensors="pt", truncation=True, max_length=256
            ).to(manager.device)
            bg_out = manager.bge_base(**bg_in)
            bg_pooled = manager.bge_pooler(bg_out.hidden_states)
            bg_probs = torch.sigmoid(manager.bge_clf(bg_pooled))[0].cpu().numpy()

        # --- 融合策略 ---
        num_labels = len(UNSAFE_LABELS)
        b_preds = b_probs[:num_labels] >= req.threshold
        bg_preds = bg_probs[:num_labels] >= req.threshold

        if req.strategy == "intersection":
            final_preds = b_preds & bg_preds
        elif req.strategy == "union":
            final_preds = b_preds | bg_preds
        else:  # average
            final_preds = ((b_probs[:num_labels] + bg_probs[:num_labels]) / 2) >= req.threshold

        final_probs = (b_probs[:num_labels] + bg_probs[:num_labels]) / 2
        detected = [UNSAFE_LABELS[i] for i, p in enumerate(final_preds) if p]

        return {
            "safe": not any(final_preds),
            "categories": detected,
            "scores": {UNSAFE_LABELS[i]: float(final_probs[i]) for i in range(num_labels)},
        }
    except Exception as e:
        # 修复：不暴露内部异常细节，避免路径/模型结构等信息泄露；详情记录到日志
        import logging as _logging
        _logging.getLogger("safeguard_system").exception("Predict error")
        raise HTTPException(status_code=500, detail="推理服务内部错误，请稍后重试")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
