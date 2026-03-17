"""
PII 脱敏模块（基于 Microsoft Presidio）

支持的中国本地实体：
  身份证号 (ID_CARD)、手机/固话 (PHONE_NUMBER)、银行卡号 (BANK_CARD)
  车牌号 (LICENSE_PLATE)、护照/港澳通行证 (PASSPORT)
  统一社会信用代码 (USCC)、邮箱 (EMAIL_ADDRESS)、IP 地址 (IP_ADDRESS)
  MAC 地址 (MAC_ADDRESS)、自定义黑名单词汇 (INTERNAL_SECRET)

用户自定义黑名单：
  编辑 config/pii_deny_list.yaml，重启服务生效

环境变量：
  PII_ENABLED=true/false        是否开启 PII 脱敏（默认 true）
  PII_USE_SPACY_ZH=false/true   是否启用 spaCy 中文 NER（默认 false）
  PII_DENY_LIST_PATH=<path>     自定义黑名单 YAML 路径
"""

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("safeguard_system")

# 是否成功加载 presidio
available: bool = False

# 对外暴露的唯一接口
def redact(text: str) -> str:
    """对 LLM 输出文本进行 PII 脱敏。presidio 不可用时原文返回。"""
    return text  # 被下方成功初始化后覆盖


# ──────────────────────────────────────────────────────────────────────────────
# 内部初始化（try/except 包裹，presidio 未安装时静默降级）
# ──────────────────────────────────────────────────────────────────────────────
try:
    from presidio_analyzer import (
        AnalyzerEngine,
        RecognizerRegistry,
        PatternRecognizer,
        Pattern,
    )
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    # ── 识别器定义 ──────────────────────────────────────────────────────────

    class _ChineseIDCardRecognizer(PatternRecognizer):
        """中国居民身份证（GB 11643-1999 校验）"""
        def __init__(self):
            super().__init__(
                supported_entity="ID_CARD",
                supported_language="zh",
                name="ChineseIDCardGB",
                patterns=[Pattern(
                    "id_card_18",
                    r"(?<!\d)[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[012])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)",
                    score=0.95,
                )],
            )

        def validate_result(self, pattern_text: str) -> Optional[bool]:
            if len(pattern_text) != 18:
                return False
            weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
            checksum_map = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']
            try:
                total = sum(int(pattern_text[i]) * weights[i] for i in range(17))
                return pattern_text[17].upper() == checksum_map[total % 11]
            except ValueError:
                return False

    class _ChineseBankCardRecognizer(PatternRecognizer):
        """银行卡号（Luhn 算法校验）"""
        def __init__(self):
            super().__init__(
                supported_entity="BANK_CARD",
                supported_language="zh",
                name="ChineseBankCardLuhn",
                patterns=[Pattern("bank_card", r"(?<!\d)[3-9]\d{14,18}(?!\d)", score=0.95)],
            )

        def validate_result(self, pattern_text: str) -> Optional[bool]:
            digits = [int(x) for x in pattern_text if x.isdigit()]
            if len(digits) < 15:
                return False
            odd_sum = sum(digits[-1::-2])
            even_sum = sum(sum(divmod(2 * d, 10)) for d in digits[-2::-2])
            return (odd_sum + even_sum) % 10 == 0

    class _ChineseUSCCRecognizer(PatternRecognizer):
        """统一社会信用代码（GB 32100-2015 校验）"""
        def __init__(self):
            super().__init__(
                supported_entity="USCC",
                supported_language="zh",
                name="ChineseUSCCGB",
                patterns=[Pattern(
                    "uscc",
                    r"(?<![A-Za-z0-9])[0-9A-HJ-NP-RT-UW-Y]{2}\d{6}[0-9A-HJ-NP-RT-UW-Y]{10}(?![A-Za-z0-9])",
                    score=0.95,
                )],
            )

        def validate_result(self, pattern_text: str) -> Optional[bool]:
            if len(pattern_text) != 18:
                return False
            alphabet = "0123456789ABCDEFGHJKLMNPQRTUWXY"
            weights = [1, 3, 9, 27, 19, 26, 16, 17, 20, 29, 25, 13, 8, 24, 10, 30, 28]
            try:
                total = sum(alphabet.index(pattern_text[i]) * weights[i] for i in range(17))
                checksum_index = (31 - (total % 31)) % 31
                return pattern_text[17].upper() == alphabet[checksum_index]
            except ValueError:
                return False

    # 无需校验算法的纯正则识别器
    _phone_recognizer = PatternRecognizer(
        supported_language="zh",
        supported_entity="PHONE_NUMBER",
        name="ChinesePhoneRecognizer",
        patterns=[
            Pattern("mobile_cn", r"(?<!\d)1[3-9]\d{9}(?!\d)", score=0.85),
            Pattern("landline_cn", r"(?<!\d)0\d{2,3}-?\d{7,8}(?!\d)", score=0.85),
        ],
    )
    _license_plate_recognizer = PatternRecognizer(
        supported_language="zh",
        supported_entity="LICENSE_PLATE",
        name="ChineseLicensePlateRecognizer",
        patterns=[Pattern(
            "license_plate_cn",
            r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁夏][A-HJ-NP-Z][·•]?[A-HJ-NP-Z0-9]{4}[A-HJ-NP-Z0-9挂学警港澳]",
            score=0.90,
        )],
    )
    _passport_recognizer = PatternRecognizer(
        supported_language="zh",
        supported_entity="PASSPORT",
        name="ChinesePassportRecognizer",
        patterns=[Pattern(
            "passport_cn",
            r"(?<![A-Za-z0-9])[EeGgDdSsPpHh]\d{8}(?![A-Za-z0-9])",
            score=0.85,
        )],
    )
    _email_recognizer = PatternRecognizer(
        supported_language="zh",
        supported_entity="EMAIL_ADDRESS",
        name="EmailRecognizer",
        patterns=[Pattern("email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", score=1.0)],
    )
    _ip_recognizer = PatternRecognizer(
        supported_language="zh",
        supported_entity="IP_ADDRESS",
        name="IpRecognizer",
        patterns=[
            Pattern(
                "ipv4",
                r"(?<![\d\.])(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?![\d\.])",
                score=0.95,
            ),
            Pattern(
                "ipv6",
                r"(?<![A-Fa-f0-9:])(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}(?![A-Fa-f0-9:])",
                score=0.95,
            ),
        ],
        context=["IP", "服务器", "地址", "网关", "路由", "ipv4", "ipv6"],
    )
    _mac_recognizer = PatternRecognizer(
        supported_language="zh",
        supported_entity="MAC_ADDRESS",
        name="MacRecognizer",
        patterns=[Pattern("mac", r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b", score=1.0)],
    )

    # 脱敏替换规则
    _OPERATORS = {
        "ID_CARD":       OperatorConfig("replace", {"new_value": "[身份证号]"}),
        "PHONE_NUMBER":  OperatorConfig("replace", {"new_value": "[手机号]"}),
        "BANK_CARD":     OperatorConfig("replace", {"new_value": "[银行卡号]"}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[邮箱]"}),
        "IP_ADDRESS":    OperatorConfig("replace", {"new_value": "[IP地址]"}),
        "MAC_ADDRESS":   OperatorConfig("replace", {"new_value": "[MAC地址]"}),
        "LICENSE_PLATE": OperatorConfig("replace", {"new_value": "[车牌号]"}),
        "PASSPORT":      OperatorConfig("replace", {"new_value": "[证件号]"}),
        "USCC":          OperatorConfig("replace", {"new_value": "[社会信用代码]"}),
        "INTERNAL_SECRET": OperatorConfig("replace", {"new_value": "[内部机密]"}),
        "LOCATION":      OperatorConfig("replace", {"new_value": "[地址]"}),
        "PERSON":        OperatorConfig("replace", {"new_value": "[姓名]"}),
    }

    # ── 黑名单加载 ──────────────────────────────────────────────────────────

    def _load_deny_list(path: str) -> List[str]:
        """从 YAML 文件加载黑名单词汇列表。文件不存在或格式错误时返回空列表。"""
        p = Path(path)
        if not p.exists():
            logger.debug("PII 黑名单文件不存在，跳过: %s", path)
            return []
        try:
            import yaml  # type: ignore
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            terms = data.get("terms", []) if isinstance(data, dict) else []
            terms = [str(t).strip() for t in terms if str(t).strip()]
            logger.info("PII 黑名单加载完成，共 %d 条词汇，来源: %s", len(terms), path)
            return terms
        except ImportError:
            # pyyaml 未安装，回退到逐行读取
            logger.warning("pyyaml 未安装，尝试按纯文本逐行读取黑名单: %s", path)
            lines = [ln.strip().lstrip("- ").strip() for ln in p.read_text("utf-8").splitlines()]
            return [ln for ln in lines if ln and not ln.startswith("#")]
        except Exception as e:
            logger.error("PII 黑名单加载失败: %s", e)
            return []

    # ── AnalyzerEngine 构建 ─────────────────────────────────────────────────

    def _build_analyzer(deny_list: List[str], use_spacy_zh: bool) -> AnalyzerEngine:
        # RecognizerRegistry 必须显式声明 supported_languages，
        # 否则 AnalyzerEngine 校验 analyzer.sl ⊆ registry.sl 时失败（默认 ["en"]）
        registry = RecognizerRegistry(supported_languages=["zh", "en"])

        base_recognizers = [
            _ChineseIDCardRecognizer(),
            _phone_recognizer,
            _ChineseBankCardRecognizer(),
            _license_plate_recognizer,
            _passport_recognizer,
            _ChineseUSCCRecognizer(),
            _email_recognizer,
            _ip_recognizer,
            _mac_recognizer,
        ]
        if deny_list:
            base_recognizers.append(PatternRecognizer(
                supported_language="zh",
                supported_entity="INTERNAL_SECRET",
                name="InternalSecretRecognizer",
                deny_list=deny_list,
                deny_list_score=1.0,
            ))

        for r in base_recognizers:
            registry.add_recognizer(r)

        if use_spacy_zh:
            try:
                configuration = {
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "zh", "model_name": "zh_core_web_sm"}],
                }
                provider = NlpEngineProvider(nlp_configuration=configuration)
                nlp_engine = provider.create_engine()
                return AnalyzerEngine(
                    registry=registry,
                    nlp_engine=nlp_engine,
                    supported_languages=["zh", "en"],
                )
            except Exception as e:
                logger.warning("spaCy 中文模型加载失败，退回纯规则模式: %s", e)

        return AnalyzerEngine(registry=registry, supported_languages=["zh", "en"])

    # ── 模块初始化 ───────────────────────────────────────────────────────────

    from config.settings import PII_DENY_LIST_PATH, PII_USE_SPACY_ZH

    _deny_list: List[str] = _load_deny_list(PII_DENY_LIST_PATH)
    _analyzer: AnalyzerEngine = _build_analyzer(_deny_list, PII_USE_SPACY_ZH)
    _anonymizer: AnonymizerEngine = AnonymizerEngine()

    # ── 覆盖对外接口 ─────────────────────────────────────────────────────────

    def redact(text: str) -> str:
        """对文本进行 PII 脱敏，返回替换后的文本。异常时原文返回。"""
        if not text:
            return text
        try:
            results = _analyzer.analyze(text=text, language="zh")
            if not results:
                return text
            anonymized = _anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=_OPERATORS,
            )
            return anonymized.text
        except Exception as e:
            logger.warning("PII 脱敏异常，原文返回: %s", e)
            return text

    available = True
    logger.info("PII 脱敏模块初始化完成（spaCy=%s，黑名单=%d 条）", PII_USE_SPACY_ZH, len(_deny_list))

except ImportError as _e:
    logger.warning("presidio 未安装，PII 脱敏功能不可用: %s", _e)
except Exception as _e:
    logger.error("PII 脱敏模块初始化失败: %s", _e, exc_info=True)
