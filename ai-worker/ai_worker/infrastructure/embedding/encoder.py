"""Прогон bge-m3 внутри рабочего процесса.

Модель грузится один раз на процесс и живёт в его памяти: перезагрузка
XLM-RoBERTa-large на каждый вызов дороже самого вызова (ADR-0016).

`torch` и `transformers` импортируются внутри функций, а не сверху. Модуль
импортируется и в родительском процессе — он должен уметь сослаться на функцию,
которую отдаёт пулу, — а тянуть туда полтора гигабайта весов и полсекунды
импорта незачем: там инференса не бывает.

Нормировка здесь не делается. Ею распоряжается политика версий, и если бы
рабочий процесс нормировал всегда, флаг политики был бы декорацией.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from ai_worker.infrastructure.embedding.model_registry import OFFLINE_ENVIRONMENT
from ai_worker.infrastructure.embedding.raw import RawEmbedding

if TYPE_CHECKING:
    import torch

SPARSE_HEAD_FILE: Final[str] = "sparse_linear.pt"


@dataclass(frozen=True, slots=True)
class _Loaded:
    """Всё, что нужно для прогона, загруженное один раз."""

    tokenizer: Any
    model: Any
    sparse_weight: torch.Tensor
    sparse_bias: torch.Tensor
    ignored_tokens: frozenset[int]
    max_input_tokens: int


_LOADED: _Loaded | None = None


def load(model_dir: str, max_input_tokens: int) -> None:
    """Готовит модель в этом процессе.

    Вызывается на старте сервиса: первая же загрузка занимает секунды, и
    платить их первым сообщением значило бы съесть его таймаут.
    """
    _prepared(model_dir, max_input_tokens)


def encode_batch(
    model_dir: str,
    max_input_tokens: int,
    texts: tuple[str, ...],
) -> tuple[RawEmbedding, ...]:
    """Считает оба представления для пачки текстов одним проходом модели."""
    import torch  # noqa: PLC0415 — импорт торча в родительском процессе не нужен

    state = _prepared(model_dir, max_input_tokens)
    with torch.inference_mode():
        encoded = state.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=state.max_input_tokens,
            return_tensors="pt",
        )
        hidden = state.model(**encoded).last_hidden_state
        # Плотное представление bge-m3 — состояние первого токена; разреженное
        # считает отдельная голова по всем состояниям сразу.
        dense = hidden[:, 0]
        token_weights = torch.relu(
            hidden @ state.sparse_weight.T + state.sparse_bias
        ).squeeze(-1)

    return tuple(
        RawEmbedding(
            dense=tuple(dense[row].tolist()),
            sparse=_token_weights(
                encoded["input_ids"][row].tolist(),
                encoded["attention_mask"][row].tolist(),
                token_weights[row].tolist(),
                state.ignored_tokens,
            ),
        )
        for row in range(len(texts))
    )


def _token_weights(
    token_ids: list[int],
    attention: list[int],
    weights: list[float],
    ignored: frozenset[int],
) -> dict[int, float]:
    """Вес каждого токена чанка — наибольший из его вхождений.

    Повторы одного токена дают несколько весов, а разреженный вектор хранит
    один на индекс: берётся сильнейший, иначе вес зависел бы от порядка обхода.
    """
    result: dict[int, float] = {}
    for token_id, present, weight in zip(token_ids, attention, weights, strict=True):
        if not present or token_id in ignored or weight <= 0:
            continue
        if weight > result.get(token_id, 0.0):
            result[token_id] = weight
    return result


def _prepared(model_dir: str, max_input_tokens: int) -> _Loaded:
    global _LOADED  # noqa: PLW0603 — кэш на процесс: ради него пул и заведён
    if _LOADED is None or _LOADED.max_input_tokens != max_input_tokens:
        _LOADED = _load(Path(model_dir), max_input_tokens)
    return _LOADED


def _load(model_dir: Path, max_input_tokens: int) -> _Loaded:
    # Запрет на сеть выставляется до импорта: после него переменные уже
    # прочитаны, и промах по кэшу увёл бы процесс качать веса.
    os.environ.update(OFFLINE_ENVIRONMENT)
    import torch  # noqa: PLC0415 — см. комментарий модуля
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    # Точность задаётся здесь, а не конфигурацией модели: она меняет сами
    # значения вектора, а реестр версий сверяет только параметры политики и
    # такую подмену не заметил бы. Половинная точность на процессоре к тому же
    # не ускоряет, а замедляет.
    model = AutoModel.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32
    )
    model.eval()
    head = torch.load(
        model_dir / SPARSE_HEAD_FILE, map_location="cpu", weights_only=True
    )
    return _Loaded(
        tokenizer=tokenizer,
        model=model,
        # Голова разреженных весов выложена в половинной точности, а состояния
        # модели считаются в одинарной: без приведения матрицы не перемножаются.
        sparse_weight=head["weight"].to(torch.float32),
        sparse_bias=head["bias"].to(torch.float32),
        # Служебные токены не несут содержания чанка, и их вес засорял бы
        # разреженный вектор одинаковыми индексами у всех документов.
        ignored_tokens=frozenset(
            token_id
            for token_id in (
                tokenizer.cls_token_id,
                tokenizer.eos_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
                tokenizer.unk_token_id,
            )
            if token_id is not None
        ),
        max_input_tokens=max_input_tokens,
    )
