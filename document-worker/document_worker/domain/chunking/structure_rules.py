"""Правила разметки строк русского юридического документа.

Регулярные выражения компилируются на импорте: около сорока шаблонов — это
единицы миллисекунд, что допустимо даже при переимпорте в каждом рабочем
процессе. Исключения из правил живут в классификаторе, а не внутри шаблонов:
регулярка с семью вшитыми оговорками нечитаема, а её правка меняет хэш
параметров целиком.
"""

from __future__ import annotations

import re
from typing import Final

RULES_VERSION: Final[str] = "ru-legal-1"

MAX_HEADING_CHARS: Final[int] = 160
MAX_HEADING_ILLEGIBLE_RATIO: Final[float] = 0.20
UPPER_HEADING_PAGE_RATIO_CUTOFF: Final[float] = 0.60
TABLE_MIN_ROWS: Final[int] = 2
YEAR_MIN: Final[int] = 1900
YEAR_MAX: Final[int] = 2100
MAX_HEADING_PATH_DEPTH: Final[int] = 8

# Ширина табуляции при подсчёте отступа: пункт с продолжением различается
# именно отступом, а таб и четыре пробела в PDF неразличимы на глаз.
TAB_WIDTH: Final[int] = 4

RE_SECTION: Final = re.compile(
    r"^\s*(?P<kw>РАЗДЕЛ|ПОДРАЗДЕЛ|ГЛАВА|ЧАСТЬ|Раздел|Подраздел|Глава|Часть)"
    r"\s+(?P<num>[IVXLCDM]{1,7}|\d{1,3}(?:\.\d{1,3})*)"
    r"\s*[.:)]?\s*(?P<title>.*)$"
)

RE_ARTICLE: Final = re.compile(
    r"^\s*(?P<kw>Статья|СТАТЬЯ|Ст\.)\s*"
    r"(?P<num>\d{1,4}(?:\.\d{1,3})*)\s*[.:)]?\s*(?P<title>.*)$"
)

# До пяти уровней: один плюс не более четырёх повторов.
RE_CLAUSE: Final = re.compile(
    r"^(?P<indent>[ \t]*)(?P<num>\d{1,4}(?:\.\d{1,3}){0,4})\.?"
    r"[ \t]+(?=[«\"(\[А-ЯЁA-Z\d])(?P<body>\S.*)$"
)

RE_SUBCLAUSE_PAREN: Final = re.compile(
    r"^(?P<indent>[ \t]*)(?P<num>\d{1,3})\)[ \t]+(?=\S)(?P<body>.*)$"
)

RE_SUBCLAUSE_LETTER: Final = re.compile(
    r"^(?P<indent>[ \t]*)(?P<num>[а-яё])\)[ \t]+(?=\S)(?P<body>.*)$"
)

RE_BULLET: Final = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>[-–—•●▪*])[ \t]+(?=\S)(?P<body>.*)$"
)

RE_UPPER_HEADING: Final = re.compile(rf"^[^a-zа-яё]{{4,{MAX_HEADING_CHARS}}}$")

# Две и больше точек с пробелом — это абзац капсом, а не заголовок.
RE_INNER_SENTENCE_BREAK: Final = re.compile(r"\.\s")

RE_CASED_LETTER: Final = re.compile(r"[^\W\d_]")
RE_LOWERCASE: Final = re.compile(r"[a-zа-яё]")

RE_APPENDIX: Final = re.compile(
    r"^\s*(?P<kw>ПРИЛОЖЕНИЕ|Приложение)\s*(?:№\s*)?"
    r"(?P<num>\d{1,3}|[IVXLCDM]{1,5})?\s*(?:к\s+(?P<parent>.+?))?\s*$"
)

RE_REQUISITE_CODE: Final = re.compile(
    r"\b(?P<code>ИНН|КПП|ОГРНИП|ОГРН|ОКПО|ОКТМО|ОКВЭД|БИК|СНИЛС)\b"
    r"[\s:№]*(?P<value>\d[\d\s\-]{6,20}\d)"
)

RE_BANK_ACCOUNT: Final = re.compile(
    r"(?:р/с|р\.\s?с\.|к/с|к\.\s?с\.|расч[её]тный\s+сч[её]т|"
    r"корреспондентский\s+сч[её]т)[\s:№]*(?P<value>\d[\d\s]{17,25}\d)",
    re.IGNORECASE,
)

# Счёт, набранный без подписи «р/с».
RE_LONG_DIGIT_RUN: Final = re.compile(r"\d[\d\s]{13,}\d")

RE_DOC_NUMBER: Final = re.compile(
    r"№\s*(?P<value>[0-9A-Za-zА-Яа-яЁё][0-9A-Za-zА-Яа-яЁё\-/._]{0,31})"
)

RE_RU_DATE: Final = re.compile(
    r"[«\"]?(?P<day>\d{1,2})[»\"]?\s*(?P<month>январ\w*|феврал\w*|март\w*|апрел\w*|"
    r"ма[йя]\w*|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)"
    r"\s*(?P<year>\d{4})\s*(?:г\.?|года)?",
    re.IGNORECASE,
)

RE_CITY_LINE: Final = re.compile(
    r"^\s*(?:г\.|гор\.|город)\s*(?P<city>[А-ЯЁ][а-яё\-]{2,30})\s*$"
)

RE_MONEY_AMOUNT: Final = re.compile(
    r"^\s*\d{1,3}(?:[\s ]\d{3})*(?:[.,]\d{1,2})?\s*"
    r"(?:\((?P<words>[^)]{3,160})\)\s*)?"
    r"(?:руб(?:лей|\.)?|₽|коп(?:еек|\.)?|тыс\.|млн\.?|млрд\.?|USD|EUR|\$)",
    re.IGNORECASE,
)

RE_SIGNATURE_ROLE: Final = re.compile(
    r"^\s*(?:Генеральный\s+директор|Исполнительный\s+директор|Директор|Президент|"
    r"Главный\s+бухгалтер|Руководитель|Управляющий|Представитель|Заказчик|"
    r"Исполнитель|Поставщик|Покупатель|Продавец|Подрядчик|Арендодатель|Арендатор|"
    r"Займодавец|Заемщик|Сторона\s*[12]|"
    r"От\s+(?:Заказчика|Исполнителя|Поставщика|Покупателя))\b",
    re.IGNORECASE,
)

RE_SIGNATURE_LINE: Final = re.compile(
    r"_{3,}\s*/?\s*(?P<fio>[А-ЯЁ][а-яё\-]+\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.|"
    r"[А-ЯЁ]\.\s?[А-ЯЁ]\.\s?[А-ЯЁ][а-яё\-]+)?"
)

RE_STAMP: Final = re.compile(r"^\s*М\.?\s?П\.?\s*$")

RE_TABLE_PIPE_ROW: Final = re.compile(r"^\s*\|.*\|\s*$")
RE_TABLE_GAP_ROW: Final = re.compile(r"^\s*\S.*?(?:[ \t]{3,}\S+){2,}\s*$")
RE_TABLE_SEPARATOR: Final = re.compile(r"^[\s|+\-=_]{5,}$")
RE_TABLE_GAP: Final = re.compile(r"[ \t]{3,}")

RE_PAGE_ARTIFACT: Final = re.compile(
    r"^\s*(?:[-–—]\s*)?(?:Стр(?:аница)?\.?\s*)?(?P<num>\d{1,3})"
    r"(?:\s*(?:из|/)\s*\d{1,3})?\s*(?:[-–—])?\s*$",
    re.IGNORECASE,
)

RE_SENTENCE_END: Final = re.compile(r"(?<=[.!?…])[»\"')\]]*\s+(?=[«\"(\[А-ЯЁA-Z\d])")

RE_INITIAL: Final = re.compile(r"^[А-ЯЁA-Z]\.$")

ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        "т.е.", "т.к.", "т.д.", "т.п.", "и.о.", "г.", "гг.", "ст.", "п.", "пп.",
        "ч.", "гл.", "абз.", "подп.", "разд.", "прил.", "руб.", "коп.", "тыс.",
        "млн.", "млрд.", "им.", "др.", "проч.", "см.", "рис.", "табл.", "стр.",
        "эт.", "корп.", "д.", "кв.", "ул.", "пр.", "обл.", "р-н", "с/х", "н/д",
    }
)  # fmt: skip
