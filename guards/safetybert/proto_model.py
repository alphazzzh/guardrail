"""
BGE + 类别原型匹配模型
核心思想：
- 使用BGE模型获取文本embedding
- 为每个类别维护一个原型向量（训练时计算）
- 预测时计算输入与原型的相似度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class PrototypeMatcher:
    """类别原型匹配器"""
    
    def __init__(self, num_safety_labels=3, num_unsafe_labels=5):
        self.num_safety_labels = num_safety_labels
        self.num_unsafe_labels = num_unsafe_labels
        
        # 原型向量 (在训练过程中更新)
        self.safety_prototypes = None  # [num_safety_labels, hidden_size]
        self.unsafe_prototypes = None  # [num_unsafe_labels, hidden_size]
        
        # 用于在线更新原型 (指数移动平均)
        self.registered_safety = torch.zeros(num_safety_labels, dtype=torch.bool)
        self.registered_unsafe = torch.zeros(num_unsafe_labels, dtype=torch.bool)
    
    def update_prototypes(self, safety_prototypes, unsafe_prototypes):
        """直接设置原型向量"""
        self.safety_prototypes = safety_prototypes
        self.unsafe_prototypes = unsafe_prototypes
        self.registered_safety = torch.ones(self.num_safety_labels, dtype=torch.bool)
        self.registered_unsafe = torch.ones(self.num_unsafe_labels, dtype=torch.bool)
    
    def update_prototype_ema(self, label_idx, embedding, is_safety=True, alpha=0.1):
        """使用指数移动平均更新单个原型"""
        embedding = embedding.detach()
        if is_safety:
            if self.safety_prototypes is None:
                self.safety_prototypes = torch.zeros(self.num_safety_labels, embedding.shape[-1])
            if not self.registered_safety[label_idx]:
                self.safety_prototypes[label_idx] = embedding
                self.registered_safety[label_idx] = True
            else:
                self.safety_prototypes[label_idx] = (
                    alpha * embedding + (1 - alpha) * self.safety_prototypes[label_idx]
                )
        else:
            if self.unsafe_prototypes is None:
                self.unsafe_prototypes = torch.zeros(self.num_unsafe_labels, embedding.shape[-1])
            if not self.registered_unsafe[label_idx]:
                self.unsafe_prototypes[label_idx] = embedding
                self.registered_unsafe[label_idx] = True
            else:
                self.unsafe_prototypes[label_idx] = (
                    alpha * embedding + (1 - alpha) * self.unsafe_prototypes[label_idx]
                )
    
    def compute_similarity(self, embedding, prototypes):
        """计算embedding与原型的余弦相似度"""
        if prototypes is None:
            return None
        # embedding: [batch, hidden_size], prototypes: [num_classes, hidden_size]
        embedding = F.normalize(embedding, p=2, dim=-1)
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        return torch.mm(embedding, prototypes.t())  # [batch, num_classes]
    
    def predict_safety(self, embedding):
        """预测安全类别"""
        sim = self.compute_similarity(embedding, self.safety_prototypes)
        if sim is None:
            return None, None
        probs = torch.softmax(sim * 10, dim=-1)  # temperature scaling
        return probs, sim
    
    def predict_unsafe(self, embedding):
        """预测不安全子类别"""
        sim = self.compute_similarity(embedding, self.unsafe_prototypes)
        if sim is None:
            return None, None
        # 多标签预测：每个子类别独立判断
        probs = torch.sigmoid(sim * 5)  # temperature scaling
        return probs, sim


class BGEPrototypeModel(nn.Module):
    """BGE + 原型匹配模型"""
    
    # BGE模型的指令前缀
    INSTRUCTION = "为这个句子生成表示以用于检索相似文章："
    
    def __init__(
        self,
        model_name_or_path="BAAI/bge-base-zh-v1.5",
        num_safety_labels=3,
        num_unsafe_labels=5,
        unsafe_label_index=1,
        projection_dim=None,  # 可选：降维投影
        use_instruction=True,  # 是否使用BGE指令前缀
    ):
        super().__init__()
        
        self.base = AutoModel.from_pretrained(model_name_or_path)
        self.hidden_size = self.base.config.hidden_size
        self.use_instruction = use_instruction
        self.unsafe_label_index = unsafe_label_index
        
        # 可选投影层
        self.projection = None
        if projection_dim and projection_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, projection_dim)
            self.output_dim = projection_dim
        else:
            self.output_dim = self.hidden_size
        
        # 原型匹配器
        self.prototype_matcher = PrototypeMatcher(num_safety_labels, num_unsafe_labels)
        
        # 可学习的温度参数
        self.safety_temperature = nn.Parameter(torch.tensor(10.0))
        self.unsafe_temperature = nn.Parameter(torch.tensor(5.0))
        
        # 投影到标签空间的可学习矩阵（用于训练时的对比学习）
        self.safety_projection = nn.Linear(self.output_dim, num_safety_labels)
        self.unsafe_projection = nn.Linear(self.output_dim, num_unsafe_labels)
    
    def get_embedding(self, input_ids, attention_mask):
        """获取BGE embedding"""
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # BGE使用[CLS]位置的向量
        embedding = outputs.last_hidden_state[:, 0]
        
        # 可选投影
        if self.projection is not None:
            embedding = self.projection(embedding)
        
        return embedding
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        unsafe_labels=None,
        **kwargs,
    ):
        """前向传播：计算embedding和损失"""
        embedding = self.get_embedding(input_ids, attention_mask)
        
        # 投影到标签空间
        safety_logits = self.safety_projection(embedding)
        unsafe_logits = self.unsafe_projection(embedding)
        
        loss = None
        if labels is not None:
            # 安全分类损失
            ce_loss = F.cross_entropy(safety_logits, labels)
            loss = ce_loss
            
            # 对比学习损失：拉近同类样本
            if unsafe_labels is not None and (labels == self.unsafe_label_index).any():
                unsafe_labels = unsafe_labels.float()
                unsafe_mask = (labels == self.unsafe_label_index).float().unsqueeze(-1)
                unsafe_count = unsafe_mask.sum()
                
                if unsafe_count.item() > 0:
                    # 多标签分类损失
                    bce = F.binary_cross_entropy_with_logits(
                        unsafe_logits, unsafe_labels, reduction="none"
                    )
                    bce = (bce * unsafe_mask.squeeze(-1)).sum() / unsafe_count
                    loss = loss + bce
        
        return {
            "loss": loss,
            "logits": torch.cat([safety_logits, unsafe_logits], dim=-1),
            "embedding": embedding,
        }
    
    def compute_prototypes_from_dataloader(self, dataloader, device="cuda"):
        """从数据加载器计算原型向量"""
        self.eval()
        
        # 收集每个类别的embedding
        safety_embeddings = {i: [] for i in range(self.prototype_matcher.num_safety_labels)}
        unsafe_embeddings = {i: [] for i in range(self.prototype_matcher.num_unsafe_labels)}
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                unsafe_labels = batch.get("unsafe_labels")
                
                embedding = self.get_embedding(input_ids, attention_mask)
                
                # 收集安全类别embedding
                for i in range(len(labels)):
                    label = labels[i].item()
                    safety_embeddings[label].append(embedding[i].cpu())
                
                # 收集不安全子类别embedding
                if unsafe_labels is not None:
                    unsafe_mask = (labels == self.unsafe_label_index)
                    for i in range(len(labels)):
                        if unsafe_mask[i]:
                            for j, has_label in enumerate(unsafe_labels[i]):
                                if has_label.item() == 1:
                                    unsafe_embeddings[j].append(embedding[i].cpu())
        
        # 计算平均原型
        safety_prototypes = []
        for i in range(self.prototype_matcher.num_safety_labels):
            if safety_embeddings[i]:
                proto = torch.stack(safety_embeddings[i]).mean(dim=0)
            else:
                proto = torch.zeros(self.output_dim)
            safety_prototypes.append(proto)
        safety_prototypes = torch.stack(safety_prototypes)
        
        unsafe_prototypes = []
        for i in range(self.prototype_matcher.num_unsafe_labels):
            if unsafe_embeddings[i]:
                proto = torch.stack(unsafe_embeddings[i]).mean(dim=0)
            else:
                proto = torch.zeros(self.output_dim)
            unsafe_prototypes.append(proto)
        unsafe_prototypes = torch.stack(unsafe_prototypes)
        
        # 更新原型
        self.prototype_matcher.update_prototypes(safety_prototypes, unsafe_prototypes)
        
        return safety_prototypes, unsafe_prototypes
    
    def predict(self, input_ids, attention_mask, threshold=0.5):
        """使用原型匹配进行预测"""
        embedding = self.get_embedding(input_ids, attention_mask)
        
        # 安全类别预测
        safety_probs, safety_sim = self.prototype_matcher.predict_safety(embedding)
        safety_pred = torch.argmax(safety_probs, dim=-1)
        
        # 不安全子类别预测
        unsafe_probs, unsafe_sim = self.prototype_matcher.predict_unsafe(embedding)
        unsafe_pred = (unsafe_probs > threshold)
        
        return {
            "safety_probs": safety_probs,
            "safety_pred": safety_pred,
            "unsafe_probs": unsafe_probs,
            "unsafe_pred": unsafe_pred,
            "embedding": embedding,
        }


def get_bge_tokenizer(model_name_or_path="BAAI/bge-base-zh-v1.5"):
    """获取BGE tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer


def preprocess_text_for_bge(text, instruction=None):
    """为BGE模型预处理文本"""
    if instruction is None:
        instruction = BGEPrototypeModel.INSTRUCTION
    return f"{instruction}{text}"
