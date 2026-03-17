"""
Presidio PII 检测评估脚本（中文增强版）

安装依赖：
    pip install presidio-analyzer presidio-anonymizer
    pip install spacy
    python -m spacy download zh_core_web_sm   # 中文 NER 模型（约 40MB）

运行：
    python tests/test_presidio_pii.py
    python tests/test_presidio_pii.py --mode demo    # 只跑演示
    python test_presidio_pii.py --mode bench   # 只跑基准测试
"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ──────────────────────────────────────────────────────────────
# 1. 中国本地 & 通用 PII Recognizer 定义 (全量自写，拒绝官方内置)
# ──────────────────────────────────────────────────────────────
from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerRegistry,
    PatternRecognizer,
    Pattern,
    RecognizerResult,
    EntityRecognizer,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# ---------- 中国身份证 ----------
class ChineseIDCardRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="ID_CARD",
            supported_language="zh",
            name="ChineseIDCardGB",
            patterns=[Pattern("id_card_18", r"(?<!\d)[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[012])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)", score=0.95)]
        )

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        """执行 GB 11643-1999 校验位算法"""
        if len(pattern_text) != 18:
            return False
            
        # 加权因子
        weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        # 余数与校验位的映射字典
        checksum_map = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']
        
        try:
            # 计算前 17 位乘积之和
            total = sum(int(pattern_text[i]) * weights[i] for i in range(17))
            # 对 11 取模，匹配最后一位
            return pattern_text[17].upper() == checksum_map[total % 11]
        except ValueError:
            return False

id_card_recognizer = ChineseIDCardRecognizer()

# ---------- 中国手机号与固定电话 ----------
phone_cn_recognizer = PatternRecognizer(
    supported_language="zh",
    supported_entity="PHONE_NUMBER",
    name="ChinesePhoneRecognizer",
    patterns=[
        Pattern(name="mobile_cn", regex=r"(?<!\d)1[3-9]\d{9}(?!\d)", score=0.85),
        Pattern(name="landline_cn", regex=r"(?<!\d)0\d{2,3}-?\d{7,8}(?!\d)", score=0.85)
    ],
)

# ---------- 银行卡号 (降低基础分，依赖上下文防误报) ----------
class ChineseBankCardRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="BANK_CARD",
            supported_language="zh",
            name="ChineseBankCardLuhn",
            # 基础分直接给 0.95，因为有底层的严格数学校验兜底，不需要再依赖上下文
            patterns=[Pattern("bank_card", r"(?<!\d)[3-9]\d{14,18}(?!\d)", score=0.95)]
        )

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        """执行 Luhn (模10) 算法校验"""
        digits = [int(x) for x in pattern_text if x.isdigit()]
        if len(digits) < 15:
            return False
            
        # 从最后一位（校验位）开始倒序计算
        odd_sum = sum(digits[-1::-2])
        # 偶数位乘 2，若大于 9 则将个位与十位相加
        even_sum = sum(sum(divmod(2 * d, 10)) for d in digits[-2::-2])
        
        # 结果能被 10 整除即为合法银行卡
        return (odd_sum + even_sum) % 10 == 0

bank_card_recognizer = ChineseBankCardRecognizer()

# ---------- 车牌号 ----------
license_plate_recognizer = PatternRecognizer(
    supported_language="zh",
    supported_entity="LICENSE_PLATE",
    name="ChineseLicensePlateRecognizer",
    patterns=[Pattern(name="license_plate_cn", regex=r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁夏][A-HJ-NP-Z][·•]?[A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9挂学警港澳]", score=0.90)],
)

# ---------- 护照 / 港澳通行证 ----------
passport_recognizer = PatternRecognizer(
    supported_language="zh",
    supported_entity="PASSPORT",
    name="ChinesePassportRecognizer",
    patterns=[Pattern(
        name="passport_cn", 
        regex=r"(?<![A-Za-z0-9])[EeGgDdSsPpHh]\d{8}(?![A-Za-z0-9])", 
        score=0.85
    )],
)

# ---------- 统一社会信用代码 (严格边界 + 上下文) ----------
class ChineseUSCCRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="USCC",
            supported_language="zh",
            name="ChineseUSCCGB",
            patterns=[Pattern("uscc", r"(?<![A-Za-z0-9])[0-9A-HJ-NP-RT-UW-Y]{2}\d{6}[0-9A-HJ-NP-RT-UW-Y]{10}(?![A-Za-z0-9])", score=0.95)]
        )

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        """执行 GB 32100-2015 校验位算法"""
        if len(pattern_text) != 18:
            return False
            
        # 特殊的 31 进制字母表
        alphabet = "0123456789ABCDEFGHJKLMNPQRTUWXY"
        # 加权因子
        weights = [1, 3, 9, 27, 19, 26, 16, 17, 20, 29, 25, 13, 8, 24, 10, 30, 28]
        
        try:
            # 计算前 17 位的加权总和
            total = sum(alphabet.index(pattern_text[i]) * weights[i] for i in range(17))
            # 计算校验位索引
            checksum_index = (31 - (total % 31)) % 31
            # 匹配最后一位
            return pattern_text[17].upper() == alphabet[checksum_index]
        except ValueError:
            return False

uscc_recognizer = ChineseUSCCRecognizer()

# ---------- 【新增】邮箱地址 ----------
email_recognizer = PatternRecognizer(
    supported_language="zh",
    supported_entity="EMAIL_ADDRESS",
    name="EmailRecognizer",
    patterns=[Pattern(name="email", regex=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", score=1.0)],
)

# ---------- IP 地址 (IPv4 + IPv6) ----------
ip_recognizer = PatternRecognizer(
    supported_language="zh",
    supported_entity="IP_ADDRESS",
    name="IpRecognizer",
    patterns=[
        # 严谨的 IPv4
        Pattern(
            name="ipv4", 
            regex=r"(?<![\d\.])(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?![\d\.])", 
            score=0.95
        ),
        # 标准的 IPv6 (简写支持)
        Pattern(
            name="ipv6", 
            regex=r"(?<![A-Fa-f0-9:])(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}(?![A-Fa-f0-9:])", 
            score=0.95
        )
    ],
    context=["IP", "服务器", "地址", "网关", "路由", "ipv4", "ipv6"]
)

# ---------- 【新增】MAC 地址 ----------
mac_recognizer = PatternRecognizer(
    supported_language="zh",
    supported_entity="MAC_ADDRESS",
    name="MacRecognizer",
    patterns=[Pattern(name="mac", regex=r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b", score=1.0)],
)

# ---------- 微信账号 (强依赖上下文) ----------
wechat_recognizer = PatternRecognizer(
    supported_language="zh",
    supported_entity="WECHAT_ID",
    name="WechatRecognizer",
    patterns=[Pattern(
        name="wechat_id", 
        # 严格边界：以字母开头，6-20位，允许包含 - 和 _
        regex=r"(?<![A-Za-z0-9_-])[a-zA-Z][a-zA-Z0-9_-]{5,19}(?![A-Za-z0-9_-])", 
        # 基础分 0.4（不及格），哪怕误命中普通的英文单词或代码，也会被丢弃
        score=0.4 
    )],
    # 只有周围出现以下关键词，才会加分使其越过及格线
    context=["微信号", "微信", "wechat", "wx", "加我", "v我", "薇信"]
)

# ---------- QQ 号码 (强依赖上下文) ----------
qq_recognizer = PatternRecognizer(
    supported_language="zh",
    supported_entity="QQ_NUMBER",
    name="QQRecognizer",
    patterns=[Pattern(
        name="qq_num", 
        # 严格边界：首位不为 0 的 5-11 位纯数字
        regex=r"(?<!\d)[1-9]\d{4,10}(?!\d)", 
        # 基础分同样设为 0.4（不及格），防止误杀邮编、验证码或价格
        score=0.4 
    )],
    # 只有周围出现以下关键词，才会确认为 QQ 号
    context=["QQ", "qq", "企鹅", "扣扣", "群号", "q号", "加群"]
)

# ---------- 内部保密词汇黑名单 ----------
# 只要文本中出现 deny_list 里的词，直接判定为敏感信息，无需正则
deny_list_recognizer = PatternRecognizer(
    supported_language="zh",
    supported_entity="INTERNAL_SECRET",
    name="InternalSecretRecognizer",
    # 业务人员可以随时在这里增删黑名单词汇
    deny_list=["阿波罗计划", "X09引擎", "海神系统", "Q3未发布财报"],
    # 黑名单匹配非常确凿，直接给满分
    deny_list_score=1.0 
)

# ──────────────────────────────────────────────────────────────
# 2. 构建 AnalyzerEngine (真正的纯血中文版)
# ──────────────────────────────────────────────────────────────
def build_analyzer(use_spacy_zh: bool = False) -> AnalyzerEngine:
    # RecognizerRegistry 默认 supported_languages=["en"]，必须显式声明，
    # 否则 AnalyzerEngine 校验 analyzer.supported_languages ⊆ registry.supported_languages 时失败。
    registry = RecognizerRegistry(supported_languages=["zh", "en"])
    # 注册自定义中文识别器（Email、IP、MAC 均已覆盖，无需官方内置）
    for r in [
        id_card_recognizer,
        phone_cn_recognizer,
        bank_card_recognizer,
        license_plate_recognizer,
        passport_recognizer,
        uscc_recognizer,
        email_recognizer,
        ip_recognizer,
        mac_recognizer
    ]:
        registry.add_recognizer(r)

    if use_spacy_zh:
        try:
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "zh", "model_name": "zh_core_web_sm"}],
            }
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()

            # Presidio 在构建 NLP 引擎时会向 registry 内部注入一个
            # supported_language="en" 的 SpacyRecognizer（内部行为，不受外部控制）。
            # 引擎初始化会校验 registry 中所有语言 ⊆ supported_languages，
            # 若只声明 ["zh"] 则 "en" 不合法，报 Misconfigured engine。
            # 解决：supported_languages 同时声明 ["zh", "en"]，但 analyze() 时
            # 传 language="zh"，这样只有 supported_language="zh" 的识别器会激活，
            # 内部那个 "en" SpacyRecognizer 不参与推理。
            return AnalyzerEngine(
                registry=registry,
                nlp_engine=nlp_engine,
                supported_languages=["zh", "en"],
            )
        except Exception as e:
            print(f"[WARN] spaCy 中文模型加载失败，退回纯规则模式: {e}")

    return AnalyzerEngine(registry=registry, supported_languages=["zh", "en"])


# ──────────────────────────────────────────────────────────────
# 3. 脱敏工具函数
# ──────────────────────────────────────────────────────────────
def redact(text: str, analyzer: AnalyzerEngine, language: str = "zh") -> Tuple[str, List]:
    """分析并脱敏，返回 (脱敏后文本, 检测结果列表)"""
    anonymizer = AnonymizerEngine()
    results = analyzer.analyze(text=text, language=language)

    if not results:
        return text, []

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={
            "ID_CARD":       OperatorConfig("replace", {"new_value": "[身份证号]"}),
            "PHONE_NUMBER":  OperatorConfig("replace", {"new_value": "[手机号]"}),
            "BANK_CARD":     OperatorConfig("replace", {"new_value": "[银行卡号]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[邮箱]"}),
            "IP_ADDRESS":    OperatorConfig("replace", {"new_value": "[IP地址]"}),
            "LICENSE_PLATE": OperatorConfig("replace", {"new_value": "[车牌号]"}),
            "PASSPORT":      OperatorConfig("replace", {"new_value": "[证件号]"}),
            "USCC":          OperatorConfig("replace", {"new_value": "[社会信用代码]"}),
            "LOCATION":      OperatorConfig("replace", {"new_value": "[地址]"}),
            "PERSON":        OperatorConfig("replace", {"new_value": "[姓名]"}),
        },
    )
    return anonymized.text, results


# ──────────────────────────────────────────────────────────────
# 4. Demo：交互式演示
# ──────────────────────────────────────────────────────────────
DEMO_TEXTS = [
    "我的手机号是13812345678，身份证号110101199001011234，请保密",
    "银行卡号6222021234567890123，请转账到此账户",
    "服务器IP 192.168.1.100，邮箱 admin@company.com 请勿外传",
    "车牌沪A·88888，护照号E12345678",
    "统一社会信用代码：91310000MA1FL6X829",
    "今天天气不错，明天我们去公园玩吧",  # 无 PII 对照
    "固定电话010-62345678，手机13912345678",
]


def run_demo(analyzer: AnalyzerEngine):
    print("\n" + "=" * 60)
    print("【演示模式】PII 检测与脱敏效果")
    print("=" * 60)

    for text in DEMO_TEXTS:
        redacted, results = redact(text, analyzer)
        print(f"\n原文: {text}")
        if results:
            detected = [f"{r.entity_type}({r.score:.2f})" for r in results]
            print(f"检测: {', '.join(detected)}")
        else:
            print("检测: (无 PII)")
        print(f"脱敏: {redacted}")


# ──────────────────────────────────────────────────────────────
# 5. 基准测试：计算 P / R / F1
# ──────────────────────────────────────────────────────────────
def _overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    return s1 < e2 and s2 < e1


def evaluate(analyzer: AnalyzerEngine, benchmark_path: str) -> Dict:
    """
    评估指标：
      - Precision = TP / (TP + FP)
      - Recall    = TP / (TP + FN)
      - F1        = 2 * P * R / (P + R)
    匹配规则：实体类型相同 + 位置有重叠即视为 TP（宽松匹配，适合中文字符边界不稳定的场景）
    """
    tp = fp = fn = 0
    type_stats: Dict[str, Dict[str, int]] = {}

    samples = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    for sample in samples:
        text = sample["text"]
        gold = sample["entities"]  # ground truth

        results = analyzer.analyze(text=text, language="zh")
        preds = [{"type": r.entity_type, "start": r.start, "end": r.end} for r in results]

        matched_gold = set()
        matched_pred = set()

        for pi, pred in enumerate(preds):
            for gi, g in enumerate(gold):
                if (
                    pred["type"] == g["type"]
                    and _overlap(pred["start"], pred["end"], g["start"], g["end"])
                    and gi not in matched_gold
                ):
                    tp += 1
                    matched_gold.add(gi)
                    matched_pred.add(pi)
                    # 按类型统计
                    t = g["type"]
                    type_stats.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
                    type_stats[t]["tp"] += 1
                    break

        for pi in range(len(preds)):
            if pi not in matched_pred:
                fp += 1
                t = preds[pi]["type"]
                type_stats.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
                type_stats[t]["fp"] += 1

        for gi in range(len(gold)):
            if gi not in matched_gold:
                fn += 1
                t = gold[gi]["type"]
                type_stats.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
                type_stats[t]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "overall": {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn},
        "by_type": type_stats,
        "total_samples": len(samples),
    }


def run_benchmark(analyzer: AnalyzerEngine, benchmark_path: str):
    print("\n" + "=" * 60)
    print("【基准测试】评估指标")
    print("=" * 60)

    metrics = evaluate(analyzer, benchmark_path)
    ov = metrics["overall"]

    print(f"\n样本数: {metrics['total_samples']}")
    print(f"TP={ov['tp']}  FP={ov['fp']}  FN={ov['fn']}")
    print(f"\n{'指标':<12} {'值':>8}")
    print("-" * 22)
    print(f"{'Precision':<12} {ov['precision']:>7.1%}")
    print(f"{'Recall':<12} {ov['recall']:>7.1%}")
    print(f"{'F1':<12} {ov['f1']:>7.1%}")

    print(f"\n{'类型':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 62)
    for t, s in sorted(metrics["by_type"].items()):
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0.0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"{t:<20} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} {p:>9.1%} {r:>7.1%} {f:>7.1%}")

    return metrics


# ──────────────────────────────────────────────────────────────
# 6. 主入口
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Presidio 中文 PII 评估")
    parser.add_argument("--mode", choices=["demo", "bench", "all"], default="all", help="运行模式")
    parser.add_argument("--spacy", action="store_true", help="启用 spaCy 中文 NER（需安装 zh_core_web_sm）")
    parser.add_argument(
        "--benchmark",
        default=str(Path(__file__).parent / "pii_benchmark.jsonl"),
        help="基准测试数据集路径",
    )
    args = parser.parse_args()

    print(f"初始化 AnalyzerEngine (spaCy={args.spacy}) ...")
    analyzer = build_analyzer(use_spacy_zh=args.spacy)
    print("初始化完成")

    if args.mode in ("demo", "all"):
        run_demo(analyzer)

    if args.mode in ("bench", "all"):
        run_benchmark(analyzer, args.benchmark)
