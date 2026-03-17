"""敏感词扫描模块（基于 AC 自动机）"""
import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ahocorasick

logger = logging.getLogger("safeguard_system")


class SensitiveScanner:
    """
    AC 自动机热更新：reload() 构建新 automaton 后一次性替换 state（scan 无锁，热路径轻量）
    """

    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths
        self._state: Tuple[Optional[ahocorasick.Automaton], Dict[str, List[str]]] = (None, {})
        self.word_count = 0
        self._reload_lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        """重新加载敏感词库"""
        new_words: set[str] = set()
        new_sources: Dict[str, set[str]] = defaultdict(set)

        for path in self.file_paths:
            p = Path(path)
            if not p.exists():
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip()
                        if w:
                            new_words.add(w)
                            new_sources[w].add(p.name)
            except Exception as e:
                logger.warning("加载词库 %s 失败: %s", p.name, e)

        if not new_words:
            with self._reload_lock:
                self._state = (None, {})
                self.word_count = 0
            logger.warning("敏感词库为空：未加载到任何词条")
            return

        # 稳定排序，避免 set 导致 automaton 构建不确定
        words_sorted = sorted(new_words, key=lambda x: (len(x), x))
        automaton = ahocorasick.Automaton()
        for i, word in enumerate(words_sorted):
            automaton.add_word(word, (i, word))
        automaton.make_automaton()

        with self._reload_lock:
            self._state = (automaton, {k: sorted(list(v)) for k, v in new_sources.items()})
            self.word_count = len(words_sorted)

        logger.info("敏感词库已更新，总计 %d 条词汇", len(words_sorted))

    def scan(self, text: str) -> List[Dict[str, Any]]:
        """扫描文本中的敏感词"""
        if not text:
            return []
        automaton, sources = self._state
        if automaton is None:
            return []
        hits: List[Dict[str, Any]] = []
        seen = set()
        for _, (_, word) in automaton.iter(text):
            if word in seen:
                continue
            seen.add(word)
            hits.append({"word": word, "sources": sources.get(word, [])})
        return hits
