"""Тексты корпуса: вымышленные, со структурой настоящего документа.

Содержание выдумано целиком — корпус измерительный, и реальных данных в нём
быть не должно. Структура настоящая: заголовок, преамбула, нумерованные
пункты, таблица реквизитов, подписи. Именно её разбирает сборка чанков, и
именно по ней считаются границы секций.

Генерация детерминирована: один и тот же seed даёт тот же текст до символа.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

BlockKind = Literal["heading", "paragraph", "clause", "requisites", "signature"]

KINDS_OF_CONTRACT: Final[tuple[str, ...]] = (
    "ПОСТАВКИ",
    "ОКАЗАНИЯ УСЛУГ",
    "АРЕНДЫ НЕЖИЛОГО ПОМЕЩЕНИЯ",
    "ПОДРЯДА",
)
CITIES: Final[tuple[str, ...]] = ("Москва", "Казань", "Новосибирск", "Екатеринбург")
COMPANIES: Final[tuple[str, ...]] = (
    "Общество с ограниченной ответственностью «Северная звезда»",
    "Акционерное общество «Промышленный альянс»",
    "Общество с ограниченной ответственностью «Восток-Логистика»",
    "Акционерное общество «Гранит-Инвест»",
)
OFFICERS: Final[tuple[str, ...]] = (
    "Ковалёв Игорь Петрович",
    "Смирнова Анна Владимировна",
    "Дементьев Сергей Аркадьевич",
    "Юрченко Марина Олеговна",
)
SECTIONS: Final[tuple[str, ...]] = (
    "ПРЕДМЕТ ДОГОВОРА",
    "ПРАВА И ОБЯЗАННОСТИ СТОРОН",
    "ЦЕНА ДОГОВОРА И ПОРЯДОК РАСЧЁТОВ",
    "ОТВЕТСТВЕННОСТЬ СТОРОН",
    "ФОРС-МАЖОР",
    "ПОРЯДОК РАЗРЕШЕНИЯ СПОРОВ",
    "СРОК ДЕЙСТВИЯ ДОГОВОРА",
    "ЗАКЛЮЧИТЕЛЬНЫЕ ПОЛОЖЕНИЯ",
)
CLAUSES: Final[tuple[str, ...]] = (
    (
        "Поставщик обязуется передать Покупателю товар в количестве и"
        " ассортименте, определённых в спецификации, являющейся неотъемлемой"
        " частью настоящего договора."
    ),
    (
        "Покупатель обязуется принять товар и оплатить его в порядке и сроки,"
        " предусмотренные разделом о расчётах."
    ),
    (
        "Право собственности на товар переходит к Покупателю с момента подписания"
        " товарной накладной уполномоченными представителями сторон."
    ),
    (
        "Стоимость единицы товара указывается в спецификации и включает налог на"
        " добавленную стоимость по ставке, установленной законодательством."
    ),
    (
        "Оплата производится в безналичном порядке платёжным поручением в течение"
        " десяти банковских дней с даты получения счёта."
    ),
    (
        "За нарушение срока поставки Поставщик уплачивает пеню в размере ноль целых"
        " одна десятая процента от стоимости непоставленного товара за каждый день"
        " просрочки."
    ),
    (
        "Сторона, не исполнившая обязательство вследствие обстоятельств"
        " непреодолимой силы, освобождается от ответственности при условии"
        " письменного уведомления другой стороны в течение пяти рабочих дней."
    ),
    (
        "Все споры и разногласия стороны разрешают путём переговоров, а при"
        " недостижении согласия — в арбитражном суде по месту нахождения ответчика."
    ),
    (
        "Договор вступает в силу с момента подписания обеими сторонами и действует"
        " до полного исполнения принятых на себя обязательств."
    ),
    (
        "Настоящий договор составлен в двух экземплярах, имеющих равную юридическую"
        " силу, по одному для каждой из сторон."
    ),
    (
        "Изменения и дополнения к настоящему договору действительны при условии их"
        " совершения в письменной форме и подписания уполномоченными лицами."
    ),
    (
        "Стороны обязуются не разглашать сведения, ставшие им известными в ходе"
        " исполнения договора, без предварительного письменного согласия."
    ),
)


@dataclass(frozen=True, slots=True)
class Block:
    """Смысловой кусок страницы вместе с его строками."""

    kind: BlockKind
    lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PageContent:
    """Одна страница: блоки, её текст и границы секций в нём."""

    blocks: tuple[Block, ...]

    @property
    def text(self) -> str:
        """Эталонный текст страницы."""
        return "\n\n".join("\n".join(block.lines) for block in self.blocks)

    @property
    def section_boundaries(self) -> tuple[int, ...]:
        """Смещения, с которых начинаются секции страницы."""
        offsets: list[int] = []
        position = 0
        for index, block in enumerate(self.blocks):
            if block.kind == "heading":
                offsets.append(position)
            position += len("\n".join(block.lines))
            if index + 1 < len(self.blocks):
                position += 2
        return tuple(offsets)


def build_document(*, seed: int, pages: int) -> tuple[PageContent, ...]:
    """Собирает документ из вымышленных, но связных блоков."""
    rng = random.Random(seed)  # noqa: S311 — корпус, а не криптография
    number = rng.randrange(100, 999)
    kind = rng.choice(KINDS_OF_CONTRACT)
    city = rng.choice(CITIES)
    parties = rng.sample(COMPANIES, 2)
    officers = rng.sample(OFFICERS, 2)
    sections = _sections_for(rng, pages)

    built = [
        _first_page(
            number=number,
            kind=kind,
            city=city,
            parties=parties,
            section=sections[0],
            rng=rng,
        )
    ]
    built.extend(
        _body_page(title, rng, index + 2) for index, title in enumerate(sections[1:])
    )
    built.append(_last_page(parties, officers))
    return tuple(built[:pages])


def _sections_for(rng: random.Random, pages: int) -> list[str]:
    # Секций ровно столько, сколько страниц без последней: последняя занята
    # реквизитами и подписями.
    count = max(1, pages - 1)
    return list(SECTIONS[:count]) if count <= len(SECTIONS) else _repeated(rng, count)


def _repeated(rng: random.Random, count: int) -> list[str]:
    return [rng.choice(SECTIONS) for _ in range(count)]


def _first_page(  # noqa: PLR0913 — первая страница собирается из всех этих частей
    *,
    number: int,
    kind: str,
    city: str,
    parties: Sequence[str],
    section: str,
    rng: random.Random,
) -> PageContent:
    preamble = (
        f"{parties[0]}, именуемое в дальнейшем «Поставщик», с одной стороны,"
        f" и {parties[1]}, именуемое в дальнейшем «Покупатель», с другой"
        " стороны, заключили настоящий договор о нижеследующем."
    )
    return PageContent(
        blocks=(
            Block("heading", (f"ДОГОВОР {kind} № {number}",)),
            Block("paragraph", (f"г. {city}", "«12» марта 2026 г.")),
            Block("paragraph", (preamble,)),
            Block("heading", (f"1. {section}",)),
            Block("clause", _clauses(rng, section_number=1, count=4)),
        )
    )


def _body_page(title: str, rng: random.Random, section_number: int) -> PageContent:
    return PageContent(
        blocks=(
            Block("heading", (f"{section_number}. {title}",)),
            Block("clause", _clauses(rng, section_number=section_number, count=5)),
        )
    )


def _last_page(parties: Sequence[str], officers: Sequence[str]) -> PageContent:
    requisites = (
        "РЕКВИЗИТЫ СТОРОН",
        f"Поставщик: {parties[0]}",
        "ИНН 7701234567, КПП 770101001, ОГРН 1157746000000",
        "Расчётный счёт 40702810900000001234 в АО «Первый банк»",
        f"Покупатель: {parties[1]}",
        "ИНН 5024098765, КПП 502401001, ОГРН 1095024000000",
        "Расчётный счёт 40702810100000004321 в ПАО «Второй банк»",
    )
    signatures = (
        f"От Поставщика: генеральный директор {officers[0]} ______________",
        f"От Покупателя: генеральный директор {officers[1]} ______________",
    )
    return PageContent(
        blocks=(
            Block("heading", (requisites[0],)),
            Block("requisites", requisites[1:]),
            Block("signature", signatures),
        )
    )


def _clauses(rng: random.Random, *, section_number: int, count: int) -> tuple[str, ...]:
    chosen = rng.sample(CLAUSES, count)
    return tuple(
        f"{section_number}.{index + 1}. {clause}" for index, clause in enumerate(chosen)
    )
