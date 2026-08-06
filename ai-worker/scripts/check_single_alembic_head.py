"""Проверка, что у миграций ровно одна голова.

Две головы появляются после параллельных веток и ломают `upgrade head`.

    python scripts/check_single_alembic_head.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

SERVICE_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Возвращает 0, если голова одна, иначе 1."""
    config = Config(str(SERVICE_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(SERVICE_ROOT / "migrations"))
    heads = ScriptDirectory.from_config(config).get_heads()
    if len(heads) == 1:
        return 0
    print(f"у миграций должна быть одна голова, найдено: {heads}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
