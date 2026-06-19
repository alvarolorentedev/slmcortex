from __future__ import annotations

from collections import Counter

from repo_brain.models import SymbolRecord


def symbol_counts(symbols: list[SymbolRecord]) -> dict[str, int]:
    return dict(sorted(Counter(symbol.kind for symbol in symbols).items()))

