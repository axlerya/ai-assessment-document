"""Порты: спецификация границ сервиса.

Порт пишется раньше своей реализации, потому что он и есть договор. Проверки
здесь про форму договора, а не про поведение: параметр, специфичный для одного
провайдера, делает абстракцию ложной — заменить его будет нечем.
"""

from __future__ import annotations

import inspect

import pytest

from ai_worker.application.ports import (
    embedding,
    health,
    llm,
    publishing,
    reading,
    repositories,
    rerank,
    search,
    system,
    unit_of_work,
)

pytestmark = pytest.mark.unit

PORT_MODULES = (
    embedding,
    rerank,
    llm,
    reading,
    search,
    publishing,
    repositories,
    unit_of_work,
    system,
    health,
)

# Имена, выдающие конкретного поставщика. Параметр порта существует, только
# если его умеют все запланированные реализации: иначе абстракция ложная и
# заменить поставщика нельзя, не переписав вызывающий код.
VENDOR_WORDS = (
    "openai",
    "anthropic",
    "deepinfra",
    "onnx",
    "torch",
    "transformers",
    "pgvector",
    "postgres",
    "rabbit",
    "amqp",
    "hnsw",
    "ef_search",
    "temperature",
    "top_p",
)


def _is_protocol(candidate: type) -> bool:
    """Отличает объявленный протокол от обычного класса рядом с ним."""
    return bool(getattr(candidate, "_is_protocol", False))


def _protocols() -> list[type]:
    found: list[type] = []
    for module in PORT_MODULES:
        found.extend(
            value
            for value in vars(module).values()
            if isinstance(value, type)
            and value.__module__ == module.__name__
            and _is_protocol(value)
        )
    return found


def test_every_port_is_declared_as_a_protocol() -> None:
    # Протокол, а не абстрактный класс: реализация не наследуется от порта, и
    # инфраструктура не становится зависимостью прикладного слоя.
    assert _protocols(), "в модулях портов нет ни одного протокола"


def test_every_port_is_runtime_checkable() -> None:
    # Композиционный корень проверяет сборку на старте, а не на первом
    # сообщении: без `runtime_checkable` такая проверка невозможна.
    not_checkable = [
        port.__name__
        for port in _protocols()
        if not getattr(port, "_is_runtime_protocol", False)
    ]

    assert not not_checkable, f"порты без runtime_checkable: {not_checkable}"


def test_every_port_method_is_annotated() -> None:
    # Аннотации читаются как строки, а не разрешаются: имена типов живут в
    # блоке TYPE_CHECKING, и попытка их разрешить требовала бы тащить
    # инфраструктуру в рантайм ради проверки.
    unannotated = [
        f"{port.__name__}.{name}"
        for port in _protocols()
        for name, member in vars(port).items()
        if not name.startswith("_")
        and callable(member)
        and "return" not in getattr(member, "__annotations__", {})
    ]

    assert not unannotated, f"методы портов без аннотации результата: {unannotated}"


def test_ports_do_not_leak_provider_specific_parameters() -> None:
    offenders: list[str] = []
    for port in _protocols():
        for name, member in vars(port).items():
            if name.startswith("_") or not callable(member):
                continue
            signature = inspect.signature(member)
            offenders.extend(
                f"{port.__name__}.{name}({parameter})"
                for parameter in signature.parameters
                if any(word in parameter.lower() for word in VENDOR_WORDS)
            )

    assert not offenders, f"порт зашивает поставщика: {offenders}"


def test_ports_do_not_import_infrastructure() -> None:
    # Порт, знающий про SQLAlchemy или брокер, перестаёт быть границей.
    forbidden = ("sqlalchemy", "faststream", "openai", "pgvector", "torch")
    offenders = [
        f"{module.__name__}: {name}"
        for module in PORT_MODULES
        for name in vars(module)
        if any(word in str(vars(module).get(name)).lower() for word in forbidden)
    ]

    assert not offenders, f"порт зависит от инфраструктуры: {offenders}"


def test_every_port_carries_a_docstring() -> None:
    # Порт — договор: без объяснения, что он обещает, реализовать его можно
    # только угадыванием.
    silent = [port.__name__ for port in _protocols() if not port.__doc__]

    assert not silent


def test_ports_are_only_declared_in_their_own_package() -> None:
    for module in PORT_MODULES:
        assert module.__name__.startswith("ai_worker.application.ports")


def test_dataclasses_beside_ports_are_not_mistaken_for_them() -> None:
    # Страховка от вырождения: рядом с портами лежат обычные значения вроде
    # `HealthStatus`, и если бы `_protocols` собирал их тоже, проверки выше
    # ничего не значили бы.
    names = {port.__name__ for port in _protocols()}

    assert "HealthStatus" not in names
    assert "LlmCompletion" not in names
    assert "SearchHit" not in names
