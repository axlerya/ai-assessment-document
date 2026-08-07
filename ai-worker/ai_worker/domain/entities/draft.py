"""Черновик, его утверждения и их цитаты.

Главное правило собрано здесь по построению: тело черновика склеивается только
из подтверждённых утверждений. Отклонённые не выбрасываются — они хранятся с
кодом причины, потому что попытка додумать факт это ровно то, что нужно уметь
разбирать, — но в текст, который увидит оператор, они не попадают никогда.

Цитата строится срезом текста чанка, а не тем, что вернула модель. Совпадение
проверяется здесь же: несовпавшая цитата — не предупреждение, а отказ, иначе
уверенно звучащий текст на неподтверждённом основании доедет до оператора.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, override

from ai_worker.domain.errors import FabricatedQuote, InvariantViolation
from ai_worker.domain.value_objects.enums import ClaimSection, DraftStatus
from ai_worker.domain.value_objects.identifiers import CitationId, DraftId
from ai_worker.domain.value_objects.scores import Ratio

if TYPE_CHECKING:
    from ai_worker.domain.entities.source_chunk import ChunkRef, SourceChunk
    from ai_worker.domain.value_objects.enums import DraftType, RejectCode
    from ai_worker.domain.value_objects.identifiers import (
        ClaimId,
        CorrelationId,
        DocumentId,
        RequestId,
    )
    from ai_worker.domain.value_objects.scores import Score
    from ai_worker.domain.value_objects.text import QuoteSpan
    from ai_worker.domain.value_objects.versioning import (
        ChunkingVersion,
        EmbeddingVersion,
        PromptVersion,
    )

SECTION_TITLES: dict[ClaimSection, str] = {
    ClaimSection.PARTIES: "Стороны и роли",
    ClaimSection.DOCUMENTS: "Документы и реквизиты",
    ClaimSection.DATES: "Ключевые даты и сроки",
    ClaimSection.AMOUNTS: "Суммы и обязательства",
    ClaimSection.OPEN_QUESTIONS: "Открытые вопросы",
}

NOTHING_FOUND = "Подтверждённых сведений по этому разделу в документах не найдено."


@dataclass(frozen=True, slots=True)
class Citation:
    """Ссылка утверждения на точный фрагмент текста чанка."""

    id: CitationId
    claim_id: ClaimId
    ref: ChunkRef
    quote: str
    span: QuoteSpan
    retrieval_score: Score
    rerank_score: Score
    reliable: bool

    def __post_init__(self) -> None:
        """Сверяет длину цитаты с её диапазоном.

        Raises:
            InvariantViolation: Длина текста не совпадает с длиной диапазона —
                цитата указывает не на тот фрагмент.
        """
        if len(self.quote) != self.span.length:
            raise InvariantViolation(
                "длина цитаты не совпадает с её диапазоном",
                context={"quote": len(self.quote), "span": self.span.length},
            )

    @classmethod
    def for_quote(  # noqa: PLR0913 — цитата описывается всеми этими значениями
        cls,
        *,
        claim_id: ClaimId,
        chunk: SourceChunk,
        span: QuoteSpan,
        quote: str,
        retrieval_score: Score,
        rerank_score: Score,
        reliable: bool,
    ) -> Self:
        """Строит цитату, сверив её с текстом источника.

        Raises:
            FabricatedQuote: Текст цитаты не является срезом текста чанка по
                указанному диапазону — то есть подтверждения не существует.
        """
        if not span.matches(chunk.text, quote=quote):
            raise FabricatedQuote(
                "цитата не является срезом текста своего чанка",
                context={
                    "chunk_id": str(chunk.ref.chunk_id),
                    "start": span.start,
                    "end": span.end,
                },
            )
        return cls(
            id=CitationId.deterministic(
                claim_id=claim_id, chunk_id=chunk.ref.chunk_id, quote_start=span.start
            ),
            claim_id=claim_id,
            ref=chunk.ref,
            quote=quote,
            span=span,
            retrieval_score=retrieval_score,
            rerank_score=rerank_score,
            reliable=reliable,
        )


@dataclass(frozen=True, slots=True, eq=False)
class Claim:
    """Одно проверяемое фактическое высказывание черновика."""

    id: ClaimId
    index: int
    section: ClaimSection
    text: str
    citations: tuple[Citation, ...]
    supported: bool
    reject_code: RejectCode | None = None

    def __post_init__(self) -> None:
        """Сверяет утверждение с его подтверждениями.

        Raises:
            InvariantViolation: Текст пуст, порядковый номер отрицателен, код
                отказа противоречит признаку подтверждённости, подтверждённое
                утверждение осталось без цитат либо цитата принадлежит другому
                утверждению.
        """
        if not self.text.strip():
            raise InvariantViolation(
                "пустое утверждение нечего подтверждать",
                context={"index": self.index},
            )
        if self.index < 0:
            raise InvariantViolation(
                "порядковый номер утверждения отрицателен",
                context={"index": self.index},
            )
        if self.supported == (self.reject_code is not None):
            raise InvariantViolation(
                "код причины существует ровно у отклонённого утверждения",
                context={"supported": self.supported, "reject": self.reject_code},
            )
        if self.supported and not self.citations:
            raise InvariantViolation(
                "подтверждённое утверждение без цитат непредставимо",
                context={"index": self.index},
            )
        foreign = [
            str(citation.id)
            for citation in self.citations
            if citation.claim_id != self.id
        ]
        if foreign:
            raise InvariantViolation(
                "цитата принадлежит другому утверждению",
                context={"citations": foreign},
            )

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Claim):
            return NotImplemented
        return self.id == other.id

    @override
    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True, slots=True, eq=False)
class Draft:
    """Сводка фактов по делу вместе со своим происхождением."""

    id: DraftId
    request_id: RequestId
    document_id: DocumentId
    draft_type: DraftType
    query: str
    status: DraftStatus
    claims: tuple[Claim, ...]
    model_name: str
    prompt_version: PromptVersion
    retrieval_profile: str
    embedding_version: EmbeddingVersion
    chunking_version: ChunkingVersion
    evidence_total: int
    correlation_id: CorrelationId | None = None

    def __post_init__(self) -> None:
        """Проверяет запрос и порядковые номера утверждений.

        Raises:
            InvariantViolation: Запрос пуст либо порядковые номера утверждений
                повторяются — тогда их ключи совпали бы, и часть утверждений
                молча потерялась бы при сохранении.
        """
        if not self.query.strip():
            raise InvariantViolation(
                "черновик без запроса не воспроизводим",
                context={"request_id": str(self.request_id)},
            )
        indexes = [claim.index for claim in self.claims]
        if len(set(indexes)) != len(indexes):
            raise InvariantViolation(
                "порядковые номера утверждений повторяются",
                context={"indexes": indexes},
            )
        # Черновик отвечает по одному документу, и цитата на чужой чанк — это
        # выдумка со ссылкой на настоящий документ: самый правдоподобный вид
        # неподтверждённого утверждения.
        foreign = [
            str(citation.ref.chunk_id)
            for claim in self.claims
            for citation in claim.citations
            if citation.ref.document_id != self.document_id
        ]
        if foreign:
            raise InvariantViolation(
                "цитата указывает на чанк другого документа",
                context={"document_id": str(self.document_id), "chunks": foreign},
            )

    @classmethod
    def assembled(  # noqa: PLR0913 — черновик описывается всеми этими значениями
        cls,
        *,
        request_id: RequestId,
        document_id: DocumentId,
        draft_type: DraftType,
        query: str,
        claims: tuple[Claim, ...],
        model_name: str,
        prompt_version: PromptVersion,
        retrieval_profile: str,
        embedding_version: EmbeddingVersion,
        chunking_version: ChunkingVersion,
        evidence_total: int,
        correlation_id: CorrelationId | None = None,
    ) -> Self:
        """Собирает черновик и выводит его исход из состава утверждений.

        Отсутствие подтверждённых утверждений — не отказ: сервис корректно
        отказался выдумывать, и это отдельный штатный исход.
        """
        supported = any(claim.supported for claim in claims)
        return cls(
            id=DraftId.deterministic(
                request_id=request_id, prompt_version=prompt_version
            ),
            request_id=request_id,
            document_id=document_id,
            draft_type=draft_type,
            query=query,
            status=(
                DraftStatus.GENERATED
                if supported
                else DraftStatus.INSUFFICIENT_EVIDENCE
            ),
            claims=claims,
            model_name=model_name,
            prompt_version=prompt_version,
            retrieval_profile=retrieval_profile,
            embedding_version=embedding_version,
            chunking_version=chunking_version,
            evidence_total=evidence_total,
            correlation_id=correlation_id,
        )

    @property
    def supported_claims(self) -> tuple[Claim, ...]:
        """Утверждения, прошедшие проверку обоснованности."""
        return tuple(claim for claim in self.claims if claim.supported)

    @property
    def claims_total(self) -> int:
        """Сколько утверждений вернула модель."""
        return len(self.claims)

    @property
    def claims_grounded(self) -> int:
        """Сколько утверждений подтверждено источниками."""
        return len(self.supported_claims)

    @property
    def claims_unsupported(self) -> int:
        """Сколько утверждений отклонено."""
        return self.claims_total - self.claims_grounded

    @property
    def citations_total(self) -> int:
        """Сколько ссылок на источники несут подтверждённые утверждения."""
        return sum(len(claim.citations) for claim in self.supported_claims)

    @property
    def groundedness(self) -> Ratio:
        """Доля подтверждённых утверждений."""
        return Ratio.of(part=self.claims_grounded, whole=self.claims_total)

    @property
    def body(self) -> str:
        """Текст черновика: только подтверждённые утверждения, по разделам.

        Раздел открытых вопросов присутствует всегда, даже пустым: молчание о
        недостатке данных читалось бы как его отсутствие.
        """
        supported = self.supported_claims
        lines: list[str] = []
        for section in ClaimSection:
            claims = sorted(
                (claim for claim in supported if claim.section is section),
                key=lambda claim: claim.index,
            )
            if not claims and section is not ClaimSection.OPEN_QUESTIONS:
                continue
            lines.append(f"## {SECTION_TITLES[section]}")
            lines.append("")
            lines.extend(f"- {claim.text}" for claim in claims)
            if not claims:
                lines.append(f"- {NOTHING_FOUND}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Draft):
            return NotImplemented
        return self.id == other.id

    @override
    def __hash__(self) -> int:
        return hash(self.id)
