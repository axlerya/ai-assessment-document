"""Укладывает модель в образ на сборке.

Запускается один раз при сборке и в подготовке медленных тестов. В рантайме
сети нет: первый же документ иначе полез бы качать два гигабайта в середине
обработки.
"""

from __future__ import annotations

from ai_worker.infrastructure.embedding.model_registry import (
    download_missing,
    model_dir_from_env,
    verify,
)


def main() -> None:
    """Докачивает недостающие файлы модели и сверяет суммы."""
    model_dir = model_dir_from_env()
    downloaded = download_missing(model_dir)
    verify(model_dir)
    print(f"модель в {model_dir}: докачано {len(downloaded)} файлов")


if __name__ == "__main__":
    main()
