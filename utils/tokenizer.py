"""Tokenizer 管理"""
import logging
import os
import threading
from pathlib import Path

from transformers import AutoTokenizer

from config import TOKENIZER_PATH

logger = logging.getLogger("safeguard_system")

_tokenizer = None
_tokenizer_lock = threading.Lock()


def get_tokenizer():
    """获取全局 tokenizer 实例（单例模式，线程安全）"""
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:  # 双重检查锁定
                try:
                    _tokenizer = AutoTokenizer.from_pretrained(
                        TOKENIZER_PATH, use_fast=True, trust_remote_code=True
                    )
                    logger.info("成功加载 Tokenizer: %s", TOKENIZER_PATH)
                except Exception as e:
                    raise SystemExit(f"Tokenizer 加载失败，请检查路径: {TOKENIZER_PATH}, Error: {e}")
    return _tokenizer
