"""Собранный образ.

Проверяется то, что нельзя проверить внутри процесса: что пакет установлен
копией, а не ссылкой на каталог сборки, что процесс работает не от root и что
образ не разросся незаметно.

Тесты помечены `slow`: сборка образа занимает минуты и в обычный прогон не
входит. Запуск — `pytest -m slow`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

IMAGE = "document-worker:test"
SERVICE_ROOT = Path(__file__).resolve().parents[3]
MAX_IMAGE_SIZE_MB = 400
BYTES_IN_MB = 1024 * 1024
NON_ROOT_UID = 10001


def _docker(*args: str) -> str:
    executable = shutil.which("docker")
    if executable is None:  # pragma: no cover — на машине без docker тесты пропущены
        pytest.skip("docker недоступен")
    result = subprocess.run(  # noqa: S603 — аргументы фиксированы в этом модуле
        [executable, *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=900,
    )
    return result.stdout.strip()


@pytest.fixture(scope="module")
def image() -> str:
    """Собирает образ сервиса."""
    _docker("build", "-t", IMAGE, str(SERVICE_ROOT))
    return IMAGE


def test_image_imports_the_package(image: str) -> None:
    # Установка ссылкой оставила бы образ зависимым от каталога сборки.
    output = _docker(
        "run", "--rm", image, "python", "-c", "import document_worker; print('ok')"
    )

    assert output == "ok"


def test_image_runs_as_non_root(image: str) -> None:
    # Под случайным пользователем тома монтируются с чужими правами.
    output = _docker("run", "--rm", "--entrypoint", "id", image, "-u")

    assert int(output) == NON_ROOT_UID


def test_image_starts_without_network(image: str) -> None:
    # Скрытый сетевой вызов на старте — модель, шрифт, словарь — превращает
    # выкатку в лотерею и запрещён уставом.
    output = _docker(
        "run",
        "--rm",
        "--network",
        "none",
        image,
        "python",
        "-c",
        "from document_worker.bootstrap import composition; print('ok')",
    )

    assert output == "ok"


def test_image_size_stays_within_the_limit(image: str) -> None:
    size = json.loads(_docker("image", "inspect", image))[0]["Size"]

    assert size / BYTES_IN_MB < MAX_IMAGE_SIZE_MB


def test_token_counter_works_without_network(image: str) -> None:
    # Словарь BPE укладывается в образ на сборке: без прогрева первый же
    # документ упёрся бы в недоступный интернет посреди обработки.
    output = _docker(
        "run",
        "--rm",
        "--network",
        "none",
        image,
        "python",
        "-c",
        "from document_worker.domain.chunking.policy import "
        "DEFAULT_CHUNKING_POLICY as p;"
        "from document_worker.infrastructure.tokenization.tiktoken_counter import "
        "TiktokenTokenCounter;"
        "print(TiktokenTokenCounter(p.encoding).count('договор поставки'))",
    )

    assert int(output) > 0
