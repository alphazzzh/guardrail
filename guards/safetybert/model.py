import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


class SafetyBertModel(nn.Module):
    def __init__(
        self,
        base_model_name,
        num_safety_labels,
        num_unsafe_labels,
        unsafe_label_index,
        dropout=0.1,
        unsafe_loss_weight=1.0,
        hierarchy_loss_weight=0.1, # SOTA: 默认 0.1
        config=None,
        **kwargs,
    ):
        super().__init__()
        # 优先使用传入的 config 初始化
        if config is not None:
            self.base = AutoModel.from_config(config)
        else:
            self.base = AutoModel.from_pretrained(base_model_name)
            
        hidden_size = self.base.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        
        # 纯正的 h_agg_max 结构：无 Projection, 无 Gate
        # 直接连接到 768 维 hidden_size
        self.safety_classifier = nn.Linear(hidden_size, num_safety_labels)
        self.unsafe_classifier = nn.Linear(hidden_size, num_unsafe_labels)
        
        self.unsafe_label_index = unsafe_label_index
        self.unsafe_loss_weight = unsafe_loss_weight
        self.hierarchy_loss_weight = hierarchy_loss_weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        unsafe_labels=None,
        **kwargs,
    ):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        if outputs.pooler_output is None:
            pooled = outputs.last_hidden_state[:, 0]
        else:
            pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        
        safety_logits = self.safety_classifier(pooled)
        unsafe_logits = self.unsafe_classifier(pooled)

        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(safety_logits, labels)
            loss = ce_loss
            
            # --- Hierarchy Loss (Max Aggregation) ---
            # 这是 h_agg_max 的核心
            if self.hierarchy_loss_weight > 0:
                safety_probs = torch.softmax(safety_logits, dim=-1)
                prob_unsafe_in_safety = safety_probs[:, self.unsafe_label_index]
                
                unsafe_probs_sub = torch.sigmoid(unsafe_logits)
                agg_prob_unsafe_sub = torch.max(unsafe_probs_sub, dim=-1).values
                
                h_loss = F.mse_loss(prob_unsafe_in_safety, agg_prob_unsafe_sub)
                loss = loss + self.hierarchy_loss_weight * h_loss
            # ----------------------------------------

            if unsafe_labels is not None:
                unsafe_labels = unsafe_labels.float()
                unsafe_mask = (labels == self.unsafe_label_index).float().unsqueeze(-1)
                unsafe_count = unsafe_mask.sum()
                if unsafe_count.item() > 0:
                    bce = F.binary_cross_entropy_with_logits(
                        unsafe_logits, unsafe_labels, reduction="none"
                    )
                    bce = (bce * unsafe_mask).sum() / unsafe_count
                    loss = loss + self.unsafe_loss_weight * bce

        combined_logits = torch.cat([safety_logits, unsafe_logits], dim=-1)
        return {"loss": loss, "logits": combined_logits}
