"""Распознанная страница не обязана хранить свой рендер.

Ограничение требовало у страницы `ocr`/`hybrid` ссылку на изображение, то есть
выгрузку рендера в объектное хранилище. Выгрузки нет и не планируется: рендер
детерминированно воспроизводится из исходного PDF, который в хранилище и так
лежит, а триста страниц по восемь мегабайт на документ не имеют в v1 ни одного
потребителя.

Воспроизвести рендер можно, только зная разрешение, поэтому `render_dpi`
остаётся обязательным — теперь он и есть содержательная часть ограничения.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-02
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD = "ck__document_pages__ocr_has_image_ref"
_NEW = "ck__document_pages__ocr_has_render_dpi"

_OLD_CONDITION = (
    "extraction_method NOT IN ('ocr','hybrid')"
    " OR (image_key IS NOT NULL AND render_dpi IS NOT NULL)"
)
_NEW_CONDITION = "extraction_method NOT IN ('ocr','hybrid') OR render_dpi IS NOT NULL"


def upgrade() -> None:
    """Меняет требование ссылки на изображение требованием разрешения."""
    op.execute(f"ALTER TABLE document_pages DROP CONSTRAINT {_OLD}")
    op.execute(
        f"ALTER TABLE document_pages ADD CONSTRAINT {_NEW} CHECK ({_NEW_CONDITION})"
    )


def downgrade() -> None:
    """Возвращает прежнее требование.

    Строки без ссылки на изображение придётся удалить вручную: придумать им
    ссылку нельзя, а тихо подставить пустую — значит сохранить неправду.
    """
    op.execute(f"ALTER TABLE document_pages DROP CONSTRAINT {_NEW}")
    op.execute(
        f"ALTER TABLE document_pages ADD CONSTRAINT {_OLD} CHECK ({_OLD_CONDITION})"
    )
