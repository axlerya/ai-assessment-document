r"""Разметка строк страницы.

Порядок правил фиксирован, побеждает первое сработавшее. Два места в нём не
вкусовые. Реквизиты проверяются раньше капс-заголовков: строка «ИНН 7701234567
КПП 770101001» строчных букв не содержит, и при обратном порядке подписной блок
договора распадается на десяток односстрочных секций. Реквизиты проверяются
позже пунктов: иначе «1. ИНН заказчика указывается в разделе 12» перестаёт быть
пунктом и выпадает из дерева секций.

Строки режутся явным проходом по `\n`, а не `str.splitlines()`: последний
дополнительно режет по `\v`, `\f` и разделителям абзаца Unicode, ломая
соответствие «строка → смещение в тексте страницы».
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from document_worker.domain.chunking.structure_rules import (
    MAX_HEADING_ILLEGIBLE_RATIO,
    MAX_SIGNATURE_ROLE_CHARS,
    RE_APPENDIX,
    RE_ARTICLE,
    RE_BANK_ACCOUNT,
    RE_BULLET,
    RE_CASED_LETTER,
    RE_CITY_LINE,
    RE_CLAUSE,
    RE_DOC_NUMBER,
    RE_INNER_SENTENCE_BREAK,
    RE_LONG_DIGIT_RUN,
    RE_LOWERCASE,
    RE_MONEY_AMOUNT,
    RE_PAGE_ARTIFACT,
    RE_REQUISITE_CODE,
    RE_RU_DATE,
    RE_SECTION,
    RE_SIGNATURE_LINE,
    RE_SIGNATURE_ROLE,
    RE_STAMP,
    RE_SUBCLAUSE_LETTER,
    RE_SUBCLAUSE_PAREN,
    RE_TABLE_GAP,
    RE_TABLE_GAP_ROW,
    RE_TABLE_PIPE_ROW,
    RE_TABLE_SEPARATOR,
    RE_TERMINAL_PUNCT,
    RE_UPPER_HEADING,
    TAB_WIDTH,
    TABLE_MIN_ROWS,
    UPPER_HEADING_PAGE_RATIO_CUTOFF,
    YEAR_MAX,
    YEAR_MIN,
)
from document_worker.domain.constants import MAX_PAGES
from document_worker.domain.value_objects.text import TextSpan

if TYPE_CHECKING:
    import re
    from collections.abc import Sequence

    from document_worker.domain.entities.document_page import DocumentPage
    from document_worker.domain.value_objects.paging import PageNumber

# Минимальная длина строки, на которой имеет смысл судить о её регистре.
MIN_CASED_LINE_CHARS = 4
# Столько точек с пробелом внутри строки делают её абзацем, а не заголовком.
MAX_HEADING_SENTENCE_BREAKS = 1
# Разброс числа колонок, при котором строки ещё считаются одной таблицей.
MAX_COLUMN_DRIFT = 1

_YEAR_DIGITS = 4


class LineKind(StrEnum):
    """Что представляет собой строка страницы."""

    BLANK = "blank"
    PAGE_ARTIFACT = "page_artifact"
    TABLE_SEPARATOR = "table_separator"
    TABLE_ROW = "table_row"
    APPENDIX = "appendix"
    HEADING_PART = "heading_part"
    HEADING_ARTICLE = "heading_article"
    HEADING_UPPER = "heading_upper"
    CLAUSE = "clause"
    SUBCLAUSE_PAREN = "subclause_paren"
    SUBCLAUSE_LETTER = "subclause_letter"
    BULLET = "bullet"
    REQUISITES = "requisites"
    SIGNATURE = "signature"
    TEXT = "text"

    @property
    def opens_section(self) -> bool:
        """Открывает ли строка новую секцию документа."""
        return self in _SECTION_OPENING_KINDS


@dataclass(frozen=True, slots=True)
class LayoutLine:
    """Одна строка страницы вместе со своей разметкой."""

    span: TextSpan
    text: str
    kind: LineKind
    page_number: PageNumber
    indent: int
    illegible_ratio: float
    number: str | None = None
    title: str | None = None
    keyword: str | None = None


@dataclass(frozen=True, slots=True)
class _Mark:
    """Итог одного правила: вид строки и извлечённые из неё части."""

    kind: LineKind
    number: str | None = None
    title: str | None = None
    keyword: str | None = None


class LineClassifier:
    """Размечает строки страницы по правилам русского юридического документа."""

    def classify_page(self, page: DocumentPage) -> tuple[LayoutLine, ...]:
        """Размечает все строки страницы; смещения — относительно её текста."""
        content = page.text.content
        spans = _line_spans(content)
        texts = [span.slice_of(content) for span in spans]
        ratios = [_illegible_ratio(span, page) for span in spans]
        allow_upper = _uppercase_detection_allowed(texts)
        marks = [
            self._mark(text, ratio=ratio, allow_upper=allow_upper)
            for text, ratio in zip(texts, ratios, strict=True)
        ]
        return tuple(
            LayoutLine(
                span=span,
                text=text,
                kind=mark.kind,
                page_number=page.number,
                indent=_indent_of(text),
                illegible_ratio=ratio,
                number=mark.number,
                title=mark.title,
                keyword=mark.keyword,
            )
            for span, text, ratio, mark in zip(
                spans, texts, ratios, _resolve_gap_rows(marks, texts), strict=True
            )
        )

    def _mark(self, text: str, *, ratio: float, allow_upper: bool) -> _Mark | None:
        """Вид строки; None — кандидат в строку таблицы, решает второй проход."""
        stripped = text.strip()
        if not stripped:
            return _Mark(LineKind.BLANK)
        return (
            _page_artifact(stripped)
            or _table(stripped)
            or _heading(stripped)
            or _signature(stripped)
            or _numbered(text, stripped)
            or _requisites(stripped)
            or _uppercase_heading(stripped, ratio=ratio, allowed=allow_upper)
            or _gap_row_candidate(text)
        )


def _line_spans(content: str) -> tuple[TextSpan, ...]:
    spans: list[TextSpan] = []
    start = 0
    while True:
        break_at = content.find("\n", start)
        if break_at < 0:
            spans.append(TextSpan(start, len(content)))
            return tuple(spans)
        spans.append(TextSpan(start, break_at))
        start = break_at + 1


def _illegible_ratio(span: TextSpan, page: DocumentPage) -> float:
    """Доля символов строки, покрытых структурными диапазонами страницы.

    Считается только по ним: маркеров в каноническом тексте нет по построению,
    и поиск маркера регуляркой дал бы двойной учёт одного дефекта.
    """
    if span.is_empty:
        return 0.0
    covered = sum(
        max(0, min(span.end, other.end) - max(span.start, other.start))
        for other in (illegible.span for illegible in page.illegible_spans)
    )
    return covered / span.length


def _indent_of(text: str) -> int:
    indent = 0
    for character in text:
        if character == " ":
            indent += 1
        elif character == "\t":
            indent += TAB_WIDTH
        else:
            break
    return indent


def _uppercase_detection_allowed(texts: Sequence[str]) -> bool:
    """Выключает капс-заголовки на странице, набранной капсом целиком.

    Без этого типовой устав ООО даёт три-пять десятков «секций» на первой
    странице.
    """
    candidates = [
        stripped
        for stripped in (text.strip() for text in texts)
        if len(stripped) >= MIN_CASED_LINE_CHARS
    ]
    if not candidates:
        return True
    uppercase = sum(1 for text in candidates if RE_LOWERCASE.search(text) is None)
    return uppercase / len(candidates) <= UPPER_HEADING_PAGE_RATIO_CUTOFF


def _page_artifact(stripped: str) -> _Mark | None:
    match = RE_PAGE_ARTIFACT.match(stripped)
    # Число выше предела страниц — это год или сумма, а не колонтитул.
    if match is None or int(match["num"]) > MAX_PAGES:
        return None
    return _Mark(LineKind.PAGE_ARTIFACT)


def _table(stripped: str) -> _Mark | None:
    if RE_TABLE_SEPARATOR.match(stripped) is not None:
        return _Mark(LineKind.TABLE_SEPARATOR)
    if RE_TABLE_PIPE_ROW.match(stripped) is not None:
        return _Mark(LineKind.TABLE_ROW)
    return None


def _heading(stripped: str) -> _Mark | None:
    appendix = RE_APPENDIX.match(stripped)
    if appendix is not None:
        return _Mark(LineKind.APPENDIX, number=appendix["num"], keyword=appendix["kw"])
    section = RE_SECTION.match(stripped)
    if section is not None:
        return _mark_of(LineKind.HEADING_PART, section)
    article = RE_ARTICLE.match(stripped)
    if article is not None:
        return _mark_of(LineKind.HEADING_ARTICLE, article)
    return None


def _mark_of(kind: LineKind, match: re.Match[str]) -> _Mark:
    return _Mark(
        kind,
        number=match["num"],
        title=match["title"] or None,
        keyword=match["kw"],
    )


def _signature(stripped: str) -> _Mark | None:
    matched = (
        RE_STAMP.match(stripped) is not None
        or RE_SIGNATURE_LINE.search(stripped) is not None
        or _is_signature_role(stripped)
    )
    return _Mark(LineKind.SIGNATURE) if matched else None


def _is_signature_role(stripped: str) -> bool:
    """Роль стороны в подписном блоке, а не предложение о той же стороне.

    Без оговорки о длине и пунктуации «Исполнитель обязуется поставить товар
    в срок.» становится подписью и вместе с ней неделимым блоком.
    """
    if RE_SIGNATURE_ROLE.match(stripped) is None:
        return False
    return (
        len(stripped) <= MAX_SIGNATURE_ROLE_CHARS
        and RE_TERMINAL_PUNCT.search(stripped) is None
    )


def _numbered(text: str, stripped: str) -> _Mark | None:
    clause = RE_CLAUSE.match(text)
    if clause is not None and not _is_false_clause(clause["num"], stripped):
        return _Mark(LineKind.CLAUSE, number=clause["num"])
    paren = RE_SUBCLAUSE_PAREN.match(text)
    if paren is not None:
        return _Mark(LineKind.SUBCLAUSE_PAREN, number=paren["num"])
    letter = RE_SUBCLAUSE_LETTER.match(text)
    if letter is not None:
        return _Mark(LineKind.SUBCLAUSE_LETTER, number=letter["num"])
    if RE_BULLET.match(text) is not None:
        return _Mark(LineKind.BULLET)
    return None


def _is_false_clause(number: str, stripped: str) -> bool:
    """Год и денежная сумма в начале строки пунктом не являются."""
    if (
        "." not in number
        and len(number) == _YEAR_DIGITS
        and YEAR_MIN <= int(number) <= YEAR_MAX
    ):
        return True
    money = RE_MONEY_AMOUNT.match(stripped)
    return money is not None and money.end() == len(stripped)


def _requisites(stripped: str) -> _Mark | None:
    matched = any(
        pattern.search(stripped) is not None
        for pattern in (
            RE_REQUISITE_CODE,
            RE_BANK_ACCOUNT,
            RE_LONG_DIGIT_RUN,
            RE_RU_DATE,
            RE_CITY_LINE,
            RE_DOC_NUMBER,
        )
    )
    return _Mark(LineKind.REQUISITES) if matched else None


def _uppercase_heading(stripped: str, *, ratio: float, allowed: bool) -> _Mark | None:
    """Капс-заголовок с оговорками: длина, число предложений, неразборчивость."""
    if not allowed or not _looks_like_uppercase_heading(stripped):
        return None
    if len(RE_INNER_SENTENCE_BREAK.findall(stripped)) > MAX_HEADING_SENTENCE_BREAKS:
        return None
    # OCR-мусор капсом не имеет права открыть секцию.
    if ratio > MAX_HEADING_ILLEGIBLE_RATIO:
        return None
    return _Mark(LineKind.HEADING_UPPER, title=stripped)


def _looks_like_uppercase_heading(stripped: str) -> bool:
    if RE_UPPER_HEADING.match(stripped) is None:
        return False
    # Голый номер счёта или год букв не содержат и заголовком не являются.
    return RE_CASED_LETTER.search(stripped) is not None


def _gap_row_candidate(text: str) -> _Mark | None:
    """None — строку решает второй проход: одиночная такая строка не таблица."""
    if RE_TABLE_GAP_ROW.match(text) is not None:
        return None
    return _Mark(LineKind.TEXT)


def _resolve_gap_rows(
    marks: Sequence[_Mark | None],
    texts: Sequence[str],
) -> tuple[_Mark, ...]:
    """Повышает до строк таблицы только серии из нескольких строк с колонками.

    Одиночная строка с широкими пробелами — это подписной блок «Заказчик —
    Исполнитель», и таблицей она не является.
    """
    resolved: list[_Mark] = [mark or _Mark(LineKind.TEXT) for mark in marks]
    for start, end in _candidate_runs(marks, texts):
        if end - start >= TABLE_MIN_ROWS:
            for index in range(start, end):
                resolved[index] = _Mark(LineKind.TABLE_ROW)
    return tuple(resolved)


def _candidate_runs(
    marks: Sequence[_Mark | None],
    texts: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    columns = 0
    for index, mark in enumerate(marks):
        if mark is not None:
            if start is not None:
                runs.append((start, index))
                start = None
            continue
        current = _column_count(texts[index])
        if start is None:
            start, columns = index, current
        elif abs(current - columns) > MAX_COLUMN_DRIFT:
            runs.append((start, index))
            start, columns = index, current
    if start is not None:
        runs.append((start, len(marks)))
    return tuple(runs)


def _column_count(text: str) -> int:
    return len(RE_TABLE_GAP.split(text.strip()))


_SECTION_OPENING_KINDS = frozenset(
    {
        LineKind.APPENDIX,
        LineKind.HEADING_PART,
        LineKind.HEADING_ARTICLE,
        LineKind.HEADING_UPPER,
        LineKind.CLAUSE,
        LineKind.SUBCLAUSE_PAREN,
        LineKind.SUBCLAUSE_LETTER,
    }
)
