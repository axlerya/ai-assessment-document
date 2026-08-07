"""Параметры построения векторов и хэш, которым они закреплены за версией.

Полей со значениями по умолчанию у политики нет: пропуск поля обязан быть
`TypeError` конструктора, видимым mypy, а не тихим значением, разъезжающимся с
настройками сервиса.

Хэш параметров сверяется с реестром на старте. Без этой сверки смена модели,
нормализации, префикса или предела входа через `.env` сложила бы векторы разной
геометрии в один namespace `embedding_version` — незаметно и необратимо: дублей
нет, ошибок нет, а половина корпуса ищется в другом пространстве (ADR-0004).

Размер пачки сюда не входит намеренно: он влияет на скорость, но не на вектор,
и обратное проверяется тестом.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NoReturn

from ai_worker.domain.constants import DENSE_DIMENSIONS, SPARSE_TOP_K
from ai_worker.domain.errors import InvalidEmbeddingPolicy
from ai_worker.domain.value_objects.embedding_identity import EmbeddingIdentity
from ai_worker.domain.value_objects.versioning import EmbeddingVersion

if TYPE_CHECKING:
    from collections.abc import Mapping

EMBEDDING_VERSION: Final[EmbeddingVersion] = EmbeddingVersion(1, 0, 0)


@dataclass(frozen=True, slots=True)
class EmbeddingPolicy:
    """Чем и как считаются оба представления чанка."""

    version: EmbeddingVersion
    model_name: str
    dimensions: int
    normalize: bool
    max_input_tokens: int
    sparse_top_k: int
    query_prefix: str
    passage_prefix: str

    def __post_init__(self) -> None:
        """Сверяет параметры с границами хранилища и с самими собой.

        Raises:
            InvalidEmbeddingPolicy: Параметр делает вектор непригодным для
                колонки или для индекса.
        """
        if not self.model_name.strip():
            self._reject("имя модели не задано")
        if self.dimensions != DENSE_DIMENSIONS:
            self._reject(f"ширина плотного вектора не {DENSE_DIMENSIONS}")
        if not 0 < self.sparse_top_k <= SPARSE_TOP_K:
            # Выше предела HNSW индекс отказывается строиться, и падает это не
            # на записи вектора, а на миграции.
            self._reject(f"предел разреженного вектора вне 1..{SPARSE_TOP_K}")
        if self.max_input_tokens <= 0:
            self._reject("предел входа в токенах не положителен")

    @staticmethod
    def _reject(reason: str) -> NoReturn:
        raise InvalidEmbeddingPolicy(
            f"параметры эмбеддингов непригодны: {reason}",
            context={"reason": reason},
        )

    @property
    def identity(self) -> EmbeddingIdentity:
        """Версия вместе с моделью: тем и объясняется происхождение вектора."""
        return EmbeddingIdentity(version=self.version, model_name=self.model_name)

    def params_hash(self) -> str:
        """sha256 канонического представления всех параметров."""
        payload = json.dumps(
            {
                "embedding_version": str(self.version),
                "model_name": self.model_name,
                "dimensions": self.dimensions,
                "normalize": self.normalize,
                "max_input_tokens": self.max_input_tokens,
                "sparse_top_k": self.sparse_top_k,
                "query_prefix": self.query_prefix,
                "passage_prefix": self.passage_prefix,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def ensure_registered(self) -> None:
        """Требует, чтобы параметрам соответствовала объявленная версия.

        Raises:
            InvalidEmbeddingPolicy: Версия неизвестна либо параметры под ней
                другие — векторы разной геометрии иначе попадут в один
                namespace.
        """
        recorded = KNOWN_EMBEDDING_HASHES.get(str(self.version))
        if recorded is None:
            raise InvalidEmbeddingPolicy(
                "версия эмбеддингов не объявлена в реестре параметров",
                context={"version": str(self.version)},
            )
        actual = self.params_hash()
        if actual != recorded:
            raise InvalidEmbeddingPolicy(
                "параметры эмбеддингов не соответствуют своей версии",
                context={
                    "version": str(self.version),
                    "expected": recorded,
                    "actual": actual,
                },
            )


# bge-m3 обучена без служебных префиксов, в отличие от моделей семейства e5:
# добавленный сюда текст ушёл бы в вектор как часть содержания.
DEFAULT_EMBEDDING_POLICY: Final[EmbeddingPolicy] = EmbeddingPolicy(
    version=EMBEDDING_VERSION,
    model_name="BAAI/bge-m3",
    dimensions=DENSE_DIMENSIONS,
    normalize=True,
    max_input_tokens=1024,
    sparse_top_k=SPARSE_TOP_K,
    query_prefix="",
    passage_prefix="",
)

# Хэш каждой выпущенной версии эмбеддингов. Запись сюда — единственный способ
# изменить геометрию векторов, и она требует осознанного инкремента версии.
KNOWN_EMBEDDING_HASHES: Final[Mapping[str, str]] = {
    "1.0.0": "1c35d2350e6326d2a091f8a44ff5a9d7fb2113ac0b13415546a0d7c9945d825c",
}
