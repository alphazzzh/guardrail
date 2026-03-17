import numpy as np


def _sigmoid(x):
    """数值稳定的 sigmoid。"""
    x = np.asarray(x)
    out = np.empty_like(x)
    pos_mask = x >= 0
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[~pos_mask])
    out[~pos_mask] = exp_x / (1.0 + exp_x)
    return out


def _precision_recall_f1(tp, fp, fn):
    """根据 TP/FP/FN 计算精确率、召回率与 F1。"""
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def _micro_scores(y_true, y_pred):
    """计算多标签 micro 指标。"""
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    return _precision_recall_f1(tp, fp, fn)


def _macro_f1(y_true, y_pred):
    """计算多标签 macro F1（逐类平均）。"""
    f1s = []
    for i in range(y_true.shape[1]):
        tp = ((y_true[:, i] == 1) & (y_pred[:, i] == 1)).sum()
        fp = ((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum()
        fn = ((y_true[:, i] == 1) & (y_pred[:, i] == 0)).sum()
        _, _, f1 = _precision_recall_f1(tp, fp, fn)
        f1s.append(f1)
    if not f1s:
        return 0.0
    return float(np.mean(f1s))


def build_compute_metrics(num_safety_labels, num_unsafe_labels, unsafe_label_index=None, threshold=0.5):
    """构建 Trainer 需要的 compute_metrics 回调。"""
    def compute_metrics(eval_pred):
        """同时计算三分类准确率与不安全多标签指标。"""
        if hasattr(eval_pred, "predictions"):
            logits = eval_pred.predictions
            label_ids = eval_pred.label_ids
        else:
            logits, label_ids = eval_pred

        if isinstance(label_ids, (tuple, list)) and len(label_ids) >= 2:
            safety_labels = label_ids[0]
            unsafe_labels = label_ids[1]
        else:
            safety_labels = label_ids
            unsafe_labels = None

        safety_logits = logits[:, :num_safety_labels]
        unsafe_logits = logits[:, num_safety_labels : num_safety_labels + num_unsafe_labels]

        # 三分类准确率
        safety_preds = np.argmax(safety_logits, axis=-1)
        safety_acc = float((safety_preds == safety_labels).mean())
        metrics = {"safety_acc": safety_acc}

        if unsafe_labels is None:
            return metrics

        # 多标签阈值化并计算 micro/macro
        unsafe_probs = _sigmoid(unsafe_logits)
        unsafe_preds = (unsafe_probs >= threshold).astype(int)
        unsafe_labels = np.array(unsafe_labels)

        # 1. 计算全局多标签指标 (所有样本)
        precision, recall, f1 = _micro_scores(unsafe_labels, unsafe_preds)
        metrics.update(
            {
                "unsafe_precision_micro": float(precision),
                "unsafe_recall_micro": float(recall),
                "unsafe_f1_micro": float(f1),
                "unsafe_f1_macro": _macro_f1(unsafe_labels, unsafe_preds),
            }
        )

        # 2. 如果提供了 unsafe_label_index，计算仅在 ground-truth unsafe 样本上的指标 (Conditional Metrics)
        if unsafe_label_index is not None:
            mask = (safety_labels == unsafe_label_index)
            if mask.any():
                cond_labels = unsafe_labels[mask]
                cond_preds = unsafe_preds[mask]
                c_prec, c_rec, c_f1 = _micro_scores(cond_labels, cond_preds)
                metrics.update({
                    "cond_unsafe_precision_micro": float(c_prec),
                    "cond_unsafe_recall_micro": float(c_rec),
                    "cond_unsafe_f1_micro": float(c_f1),
                })

        return metrics

    return compute_metrics
