"""Границы предложений внутри блока.

Кандидаты даёт регулярное выражение, а отсеиваются они по слову перед точкой:
сокращение, инициал и номер пункта заканчиваются точкой, но предложение на них
не кончается. Разрез в таком месте оставил бы «согласно ст.» отдельным чанком.
"""

from __future__ import annotations

from document_worker.domain.chunking.structure_rules import (
    ABBREVIATIONS,
    RE_INITIAL,
    RE_SENTENCE_END,
)


class SentenceSplitter:
    """Ищет начала предложений в тексте."""

    def boundaries(self, text: str) -> tuple[int, ...]:
        """Смещения начал предложений, кроме нулевого."""
        return tuple(
            match.end()
            for match in RE_SENTENCE_END.finditer(text)
            if not _is_false_boundary(text, match.start())
        )


def _is_false_boundary(text: str, at: int) -> bool:
    # Слово перед границей есть всегда: кандидат требует знака препинания слева.
    last = text[:at].rstrip().rsplit(maxsplit=1)[-1]
    return (
        last.lower() in ABBREVIATIONS
        or RE_INITIAL.match(last) is not None
        or last.rstrip(".").isdigit()
    )
