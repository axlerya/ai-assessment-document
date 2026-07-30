"""Правила нормализации текста страницы.

Каждый символ результата произведён ровно из одного символа исходника по
закрытой таблице, не зависящей от содержимого. Дописать текст правило не может:
для этого пришлось бы расширить перечисление действий.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from document_worker.domain.normalization.offsets import (
    OffsetMap,
    OffsetMapBuilder,
    RuleAction,
)

if TYPE_CHECKING:
    from collections.abc import Callable

NEWLINE = "\n"
SPACE = " "
HYPHEN = "-"

# Типографские лигатуры. Буквы алфавитов (æ, œ, ß) сюда не входят.
LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}

INVISIBLE_CHARS = frozenset(
    chr(code)
    for code in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x200E, 0x200F, 0x00AD)
)

# En dash и em dash не трогаем: в «п. 1–5» и «истец — ООО» они несут смысл.
HYPHEN_LIKE_DASHES = frozenset(chr(code) for code in (0x2010, 0x2011, 0x2212))

SPACE_LIKE = frozenset(
    chr(code)
    for code in (
        0x00A0,
        0x2000,
        0x2001,
        0x2002,
        0x2003,
        0x2004,
        0x2005,
        0x2006,
        0x2007,
        0x2008,
        0x2009,
        0x200A,
        0x202F,
        0x205F,
        0x3000,
    )
)


@dataclass(frozen=True, slots=True)
class NormalizationRule:
    """Одно правило нормализации."""

    name: str
    actions: frozenset[RuleAction]
    transform: Callable[[str], tuple[str, OffsetMap]] = field(compare=False)
    applies_to_ocr_only: bool = False

    def apply(self, text: str) -> tuple[str, OffsetMap]:
        """Применяет правило и возвращает результат вместе с картой смещений."""
        if not text:
            return "", OffsetMap.identity(0)
        return self.transform(text)


def _nfc(text: str) -> tuple[str, OffsetMap]:
    # NFC, а не NFKC: совместимостные преобразования превращают № в No,
    # ½ в 1/2 и меняют юридически значимую запись.
    builder = OffsetMapBuilder()
    parts: list[str] = []
    index = 0
    while index < len(text):
        end = index + 1
        while end < len(text) and unicodedata.combining(text[end]):
            end += 1
        cluster = text[index:end]
        composed = unicodedata.normalize("NFC", cluster)
        if len(composed) > len(cluster):
            composed = cluster
        parts.append(composed)
        builder.add(
            source=len(cluster),
            target=len(composed),
            action=_cluster_action(cluster, composed),
        )
        index = end
    return "".join(parts), builder.build()


def _cluster_action(cluster: str, composed: str) -> RuleAction:
    if cluster == composed:
        return RuleAction.KEEP
    if len(composed) < len(cluster):
        return RuleAction.COLLAPSE
    return RuleAction.MAP


def _fold_ligatures(text: str) -> tuple[str, OffsetMap]:
    builder = OffsetMapBuilder()
    parts: list[str] = []
    for char in text:
        expanded = LIGATURES.get(char)
        parts.append(expanded or char)
        builder.add(
            source=1,
            target=len(expanded) if expanded else 1,
            action=RuleAction.UNFOLD if expanded else RuleAction.KEEP,
        )
    return "".join(parts), builder.build()


def _remove_invisible(text: str) -> tuple[str, OffsetMap]:
    builder = OffsetMapBuilder()
    parts: list[str] = []
    for char in text:
        dropped = char in INVISIBLE_CHARS
        if not dropped:
            parts.append(char)
        builder.add(
            source=1,
            target=0 if dropped else 1,
            action=RuleAction.DROP if dropped else RuleAction.KEEP,
        )
    return "".join(parts), builder.build()


def _strip_control(text: str) -> tuple[str, OffsetMap]:
    builder = OffsetMapBuilder()
    parts: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\r":
            consumed = 2 if text[index + 1 : index + 2] == NEWLINE else 1
            parts.append(NEWLINE)
            builder.add(
                source=consumed,
                target=1,
                action=RuleAction.COLLAPSE if consumed > 1 else RuleAction.MAP,
            )
            index += consumed
            continue
        if char == "\t":
            parts.append(SPACE)
            builder.add(source=1, target=1, action=RuleAction.MAP)
        elif char != NEWLINE and _is_control(char):
            builder.add(source=1, target=0, action=RuleAction.DROP)
        else:
            parts.append(char)
            builder.add(source=1, target=1, action=RuleAction.KEEP)
        index += 1
    return "".join(parts), builder.build()


def _is_control(char: str) -> bool:
    return unicodedata.category(char) == "Cc"


def _unify_dashes(text: str) -> tuple[str, OffsetMap]:
    builder = OffsetMapBuilder()
    parts: list[str] = []
    for char in text:
        replaced = char in HYPHEN_LIKE_DASHES
        parts.append(HYPHEN if replaced else char)
        builder.add(
            source=1,
            target=1,
            action=RuleAction.MAP if replaced else RuleAction.KEEP,
        )
    return "".join(parts), builder.build()


def _collapse_spaces(text: str) -> tuple[str, OffsetMap]:
    builder = OffsetMapBuilder()
    parts: list[str] = []
    index = 0
    at_line_start = True
    while index < len(text):
        char = text[index]
        if char == NEWLINE:
            parts.append(char)
            builder.add(source=1, target=1, action=RuleAction.KEEP)
            at_line_start = True
            index += 1
            continue
        if not _is_space_like(char):
            parts.append(char)
            builder.add(source=1, target=1, action=RuleAction.KEEP)
            at_line_start = False
            index += 1
            continue

        run_end = index
        while run_end < len(text) and _is_space_like(text[run_end]):
            run_end += 1
        run = run_end - index
        # Отступ в начале строки сохраняется: его читает детектор структуры.
        if at_line_start:
            parts.append(SPACE * run)
            builder.add(source=run, target=run, action=RuleAction.MAP)
        elif run_end == len(text) or text[run_end] == NEWLINE:
            builder.add(source=run, target=0, action=RuleAction.DROP)
        else:
            parts.append(SPACE)
            builder.add(
                source=run,
                target=1,
                action=RuleAction.COLLAPSE if run > 1 else RuleAction.MAP,
            )
        index = run_end
    return "".join(parts), builder.build()


def _is_space_like(char: str) -> bool:
    return char == SPACE or char in SPACE_LIKE


NFC_NORMALIZE = NormalizationRule(
    name="nfc_normalize",
    actions=frozenset({RuleAction.MAP, RuleAction.COLLAPSE}),
    transform=_nfc,
)

FOLD_LIGATURES = NormalizationRule(
    name="fold_ligatures",
    actions=frozenset({RuleAction.UNFOLD}),
    transform=_fold_ligatures,
)

REMOVE_INVISIBLE = NormalizationRule(
    name="remove_invisible",
    actions=frozenset({RuleAction.DROP}),
    transform=_remove_invisible,
)

STRIP_CONTROL = NormalizationRule(
    name="strip_control",
    actions=frozenset({RuleAction.DROP, RuleAction.MAP, RuleAction.COLLAPSE}),
    transform=_strip_control,
)

UNIFY_DASHES = NormalizationRule(
    name="unify_dashes",
    actions=frozenset({RuleAction.MAP}),
    transform=_unify_dashes,
)

COLLAPSE_SPACES = NormalizationRule(
    name="collapse_spaces",
    actions=frozenset({RuleAction.MAP, RuleAction.COLLAPSE, RuleAction.DROP}),
    transform=_collapse_spaces,
)


# Буквы, неотличимые в типичном шрифте.
_LATIN_TO_CYRILLIC = {
    "a": "а",
    "c": "с",
    "e": "е",
    "o": "о",
    "p": "р",
    "x": "х",
    "y": "у",
    "A": "А",
    "B": "В",
    "C": "С",
    "E": "Е",
    "H": "Н",
    "K": "К",
    "M": "М",
    "O": "О",
    "P": "Р",
    "T": "Т",
    "X": "Х",
}
_CYRILLIC_TO_LATIN = {value: key for key, value in _LATIN_TO_CYRILLIC.items()}

_LETTER = r"[^\W\d_]"
_TOKEN_RE = re.compile(rf"{_LETTER}+")
_HYPHENATED_BREAK_RE = re.compile(rf"({_LETTER}+)-\n({_LETTER}+)")

# Дефис здесь часть слова, а не перенос. Таблица закрытая: определять это
# по смыслу значило бы угадывать.
_HYPHEN_KEEPING_PREFIXES = frozenset({"в", "во", "из", "кое", "по", "пол"})
_HYPHEN_KEEPING_SUFFIXES = frozenset({"то", "либо", "нибудь", "таки"})
_LIST_MARKER_RE = re.compile(r"^\s*(?:[—•\-]\s|\d+(?:\.\d+)*[.)]\s|[а-яa-z][.)]\s)")
_SENTENCE_END = ".!?:;"
_CYRILLIC_FIRST = "Ѐ"
_CYRILLIC_LAST = "ӿ"

_Replacement = tuple[int, int, str, RuleAction]


def _is_cyrillic(char: str) -> bool:
    return _CYRILLIC_FIRST <= char <= _CYRILLIC_LAST


def _is_latin(char: str) -> bool:
    return ("a" <= char <= "z") or ("A" <= char <= "Z")


def _replacements_to_map(
    text: str,
    replacements: list[_Replacement],
) -> tuple[str, OffsetMap]:
    """Собирает результат и карту по списку замен; остальное переносится как есть."""
    builder = OffsetMapBuilder()
    parts: list[str] = []
    cursor = 0
    for start, end, target, action in replacements:
        if start > cursor:
            parts.append(text[cursor:start])
            builder.add(
                source=start - cursor, target=start - cursor, action=RuleAction.KEEP
            )
        parts.append(target)
        builder.add(source=end - start, target=len(target), action=action)
        cursor = end
    if cursor < len(text):
        parts.append(text[cursor:])
        builder.add(
            source=len(text) - cursor,
            target=len(text) - cursor,
            action=RuleAction.KEEP,
        )
    return "".join(parts), builder.build()


def _dehyphenate(text: str) -> tuple[str, OffsetMap]:
    """Снимает перенос слова, оставляя дефис там, где он часть слова."""
    replacements: list[_Replacement] = []
    for match in _HYPHENATED_BREAK_RE.finditer(text):
        head, tail = match.group(1), match.group(2)
        hyphen_position = match.start() + len(head)
        start = (
            hyphen_position + 1
            if _hyphen_belongs_to_word(head, tail)
            else hyphen_position
        )
        replacements.append((start, hyphen_position + 2, "", RuleAction.DROP))
    return _replacements_to_map(text, replacements)


def _hyphen_belongs_to_word(head: str, tail: str) -> bool:
    # Заглавная после переноса это составное имя собственное: Санкт-Петербург.
    return (
        tail[0].isupper()
        or head.lower() in _HYPHEN_KEEPING_PREFIXES
        or tail.lower() in _HYPHEN_KEEPING_SUFFIXES
    )


def _join_soft_lines(text: str) -> tuple[str, OffsetMap]:
    replacements: list[_Replacement] = []
    for match in re.finditer(NEWLINE, text):
        start = match.start()
        if start == 0 or text[start + 1 : start + 2] == NEWLINE:
            continue
        if text[start - 1] in _SENTENCE_END or text[start - 1] == NEWLINE:
            continue
        # Маркер списка и нумерация это структура договора, её читает чанкование.
        if _LIST_MARKER_RE.match(text[start + 1 :]):
            continue
        replacements.append((start, start + 1, SPACE, RuleAction.MAP))
    return _replacements_to_map(text, replacements)


def _fold_homoglyphs(text: str) -> tuple[str, OffsetMap]:
    replacements: list[_Replacement] = []
    for match in _TOKEN_RE.finditer(text):
        table = _token_table(match.group())
        if table is None:
            continue
        for offset, char in enumerate(match.group()):
            replacement = table.get(char)
            if replacement is not None:
                position = match.start() + offset
                replacements.append(
                    (position, position + 1, replacement, RuleAction.MAP)
                )
    return _replacements_to_map(text, replacements)


def _token_table(token: str) -> dict[str, str] | None:
    """Таблица замен, если алфавит токена определяется однозначно.

    Токен целиком из омоглифов алфавита не имеет и остаётся как есть.
    """
    cyrillic = sum(
        1 for char in token if _is_cyrillic(char) and char not in _CYRILLIC_TO_LATIN
    )
    latin = sum(
        1 for char in token if _is_latin(char) and char not in _LATIN_TO_CYRILLIC
    )
    if cyrillic and not latin:
        return _LATIN_TO_CYRILLIC
    if latin and not cyrillic:
        return _CYRILLIC_TO_LATIN
    return None


DEHYPHENATE_LINE_BREAK = NormalizationRule(
    name="dehyphenate_line_break",
    actions=frozenset({RuleAction.DROP}),
    transform=_dehyphenate,
)

JOIN_SOFT_LINES = NormalizationRule(
    name="join_soft_lines",
    actions=frozenset({RuleAction.MAP}),
    transform=_join_soft_lines,
)

FOLD_HOMOGLYPHS = NormalizationRule(
    name="fold_homoglyphs",
    actions=frozenset({RuleAction.MAP}),
    transform=_fold_homoglyphs,
    applies_to_ocr_only=True,
)
