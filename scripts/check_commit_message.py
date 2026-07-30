"""Хук commit-msg: тема коммита должна быть вида `тип: описание`.

Служебные сообщения git (Merge, Revert) пропускаются — их текст формирует сам git.

    python scripts/check_commit_message.py .git/COMMIT_EDITMSG
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWED_TYPES = ("feature", "fix", "docs")
SUBJECT_RE = re.compile(rf"^(?:{'|'.join(ALLOWED_TYPES)}): \S.*$")
GIT_GENERATED_PREFIXES = ("Merge ", "Revert ")

USAGE_ERROR = "нужен ровно один аргумент — путь к файлу с сообщением коммита"
EMPTY_MESSAGE_ERROR = "сообщение коммита пустое"
FORMAT_ERROR_TEMPLATE = (
    "неверный формат сообщения коммита:\n"
    "  {subject}\n\n"
    "Ожидается «тип: описание на русском языке», допустимые типы: {types}.\n"
    "Примеры:\n"
    "  feature: прикрутил отправку сообщений в тг\n"
    "  fix: починил авторизацию на сайте\n"
    "  docs: обновил правила для коммитов"
)


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _extract_subject(raw_message: str) -> str | None:
    """Первая строка сообщения, не пустая и не комментарий git."""
    for line in raw_message.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return None


def main(argv: list[str]) -> int:
    """Возвращает 0, если формат темы корректен, иначе 1."""
    if len(argv) != 1:
        return _fail(USAGE_ERROR)

    message_file = Path(argv[0])
    try:
        raw_message = message_file.read_text(encoding="utf-8")
    except OSError as error:
        return _fail(f"не удалось прочитать {message_file}: {error}")

    subject = _extract_subject(raw_message)
    if subject is None:
        return _fail(EMPTY_MESSAGE_ERROR)

    if subject.startswith(GIT_GENERATED_PREFIXES) or SUBJECT_RE.match(subject):
        return 0

    return _fail(
        FORMAT_ERROR_TEMPLATE.format(subject=subject, types=", ".join(ALLOWED_TYPES))
    )


if __name__ == "__main__":
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main(sys.argv[1:]))
