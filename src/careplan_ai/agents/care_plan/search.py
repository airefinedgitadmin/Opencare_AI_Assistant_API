"""Keyword search (BM25) over ONE resident's records - no model, no external
service, deterministic (ties broken by evidence order).

Used to build each requirement's evidence pack and as the care plan agent's
`search_records` tool. Words are stemmed the same way as the code checks
(validation.py), and numbers / dates are kept whole so "250ml" or
"02/03/2025" match exactly.
"""

import math
import re
from collections import Counter

from careplan_ai.agents.care_plan.schemas import EvidenceItem
from careplan_ai.agents.care_plan.validation import is_content_stem, stem

_TOKEN = re.compile(r"\d+(?:[.:/,]\d+)*[a-z]*|[a-z][a-z'’-]*", re.IGNORECASE)
_K1 = 1.5
_B = 0.75


def tokens(text: str) -> list[str]:
    out: list[str] = []
    for raw in _TOKEN.findall(text):
        if raw[0].isdigit():
            out.append(raw.lower())
        else:
            s = stem(raw)
            if is_content_stem(s):
                out.append(s)
    return out


class RecordIndex:
    """BM25 over the label and text of each evidence item."""

    def __init__(self, items: list[EvidenceItem]) -> None:
        self.items = items
        self._docs = [Counter(tokens(f"{e.label} {e.text}")) for e in items]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg = (sum(self._lengths) / len(self._lengths)) if items else 0.0
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(items)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(
        self, query: str, *, k: int, exclude: set[str] | None = None
    ) -> list[tuple[EvidenceItem, float]]:
        terms = set(tokens(query))
        if not terms or not self.items:
            return []
        exclude = exclude or set()
        scored: list[tuple[float, int]] = []
        for index, (doc, length) in enumerate(zip(self._docs, self._lengths, strict=True)):
            if self.items[index].id in exclude:
                continue
            score = 0.0
            for term in terms & doc.keys():
                tf = doc[term]
                norm = tf + _K1 * (1 - _B + _B * length / (self._avg or 1))
                score += self._idf[term] * tf * (_K1 + 1) / norm
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [(self.items[i], s) for s, i in scored[:k]]
