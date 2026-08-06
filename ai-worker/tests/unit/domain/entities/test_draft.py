"""Цитата, утверждение и черновик: тело собирается только из подтверждённого."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ai_worker.domain.entities.draft import Citation, Claim, Draft
from ai_worker.domain.errors import FabricatedQuote, InvariantViolation
from ai_worker.domain.value_objects.enums import (
    ClaimSection,
    DraftStatus,
    DraftType,
    RejectCode,
)
from ai_worker.domain.value_objects.identifiers import ClaimId, DraftId, RequestId
from ai_worker.domain.value_objects.scores import Ratio, Score
from ai_worker.domain.value_objects.text import QuoteSpan
from ai_worker.domain.value_objects.versioning import (
    ChunkingVersion,
    EmbeddingVersion,
    PromptVersion,
)
from tests.factories import CHUNK_TEXT, make_chunk

if TYPE_CHECKING:
    from ai_worker.domain.entities.source_chunk import SourceChunk

pytestmark = pytest.mark.unit

PROMPT_VERSION = PromptVersion(1, 0, 0)
QUOTE = "Договор поставки № 12/АБ"
QUOTE_SPAN = QuoteSpan(start=0, end=len(QUOTE))


def _citation(
    *,
    claim_id: ClaimId | None = None,
    chunk: SourceChunk | None = None,
    quote: str = QUOTE,
    span: QuoteSpan | None = None,
    reliable: bool = True,
) -> Citation:
    return Citation.for_quote(
        claim_id=claim_id or ClaimId.generate(),
        chunk=chunk or make_chunk(),
        span=span or QUOTE_SPAN,
        quote=quote,
        retrieval_score=Score(0.9),
        rerank_score=Score(2.1),
        reliable=reliable,
    )


def _claim(
    *,
    index: int = 0,
    section: ClaimSection = ClaimSection.DOCUMENTS,
    text: str = "Стороны заключили договор поставки № 12/АБ.",
    citations: tuple[Citation, ...] | None = None,
    supported: bool = True,
    reject_code: RejectCode | None = None,
) -> Claim:
    draft_id = DraftId.generate()
    claim_id = ClaimId.deterministic(draft_id=draft_id, claim_index=index)
    return Claim(
        id=claim_id,
        index=index,
        section=section,
        text=text,
        citations=citations
        if citations is not None
        else (_citation(claim_id=claim_id),),
        supported=supported,
        reject_code=reject_code,
    )


def _draft(claims: tuple[Claim, ...]) -> Draft:
    return Draft.assembled(
        request_id=RequestId.generate(),
        document_id=make_chunk().ref.document_id,
        draft_type=DraftType.CASE_FACT_SUMMARY,
        query="Собери сводку фактов по делу",
        claims=claims,
        model_name="deepseek-ai/DeepSeek-V4-Flash",
        prompt_version=PROMPT_VERSION,
        retrieval_profile="hybrid-rrf-v1",
        embedding_version=EmbeddingVersion(1, 0, 0),
        chunking_version=ChunkingVersion(1, 0, 0),
        evidence_total=7,
    )


def test_citation_quotes_exactly_what_the_chunk_says() -> None:
    citation = _citation()

    assert citation.quote == QUOTE
    assert citation.span.slice_of(CHUNK_TEXT) == QUOTE


def test_citation_rejects_quote_that_is_not_a_slice_of_the_chunk() -> None:
    # Ключевая защита: модель вправе вернуть что угодно, но подтверждением
    # считается только дословный срез текста источника.
    with pytest.raises(FabricatedQuote):
        _citation(quote="Договор аренды № 99/ЯЯ")


def test_citation_rejects_a_span_outside_the_chunk() -> None:
    with pytest.raises(FabricatedQuote):
        _citation(span=QuoteSpan(start=0, end=len(CHUNK_TEXT) + 20))


def test_citation_key_is_determined_by_claim_chunk_and_offset() -> None:
    claim_id = ClaimId.generate()
    chunk = make_chunk()

    first = _citation(claim_id=claim_id, chunk=chunk)
    second = _citation(claim_id=claim_id, chunk=chunk)

    assert first.id == second.id


def test_citations_to_different_chunks_do_not_share_a_key() -> None:
    claim_id = ClaimId.generate()

    first = _citation(claim_id=claim_id, chunk=make_chunk())
    second = _citation(claim_id=claim_id, chunk=make_chunk())

    assert first.id != second.id


def test_citation_keeps_the_coordinates_needed_to_check_it() -> None:
    citation = _citation()

    assert citation.ref.page_number == 3
    assert citation.retrieval_score == Score(0.9)
    assert citation.rerank_score == Score(2.1)


def test_claim_text_cannot_be_blank() -> None:
    with pytest.raises(InvariantViolation):
        _claim(text="   ")


def test_supported_claim_has_no_reject_code() -> None:
    with pytest.raises(InvariantViolation):
        _claim(supported=True, reject_code=RejectCode.NO_CITATION)


def test_rejected_claim_must_explain_why() -> None:
    # Без кода причины разбор попытки додумать превращается в чтение логов.
    with pytest.raises(InvariantViolation):
        _claim(supported=False, reject_code=None)


def test_supported_claim_without_citations_is_impossible() -> None:
    with pytest.raises(InvariantViolation):
        _claim(supported=True, citations=())


def test_rejected_claim_may_have_no_citations() -> None:
    claim = _claim(supported=False, citations=(), reject_code=RejectCode.NO_CITATION)

    assert not claim.supported


def test_citations_belong_to_their_claim() -> None:
    # Цитата чужого утверждения означает перепутанный ключ и ссылку, которая
    # ничего не подтверждает.
    with pytest.raises(InvariantViolation):
        _claim(citations=(_citation(claim_id=ClaimId.generate()),))


def test_draft_body_contains_only_supported_claims() -> None:
    supported = _claim(index=0, text="Договор № 12/АБ заключён 3 марта 2024 года.")
    rejected = _claim(
        index=1,
        text="Стороны договорились о неустойке в размере 5 000 000 рублей.",
        citations=(),
        supported=False,
        reject_code=RejectCode.NO_CITATION,
    )

    draft = _draft((supported, rejected))

    assert "12/АБ" in draft.body
    assert "5 000 000" not in draft.body


def test_rejected_claims_are_kept_for_triage() -> None:
    rejected = _claim(
        index=0, citations=(), supported=False, reject_code=RejectCode.QUOTE_NOT_FOUND
    )

    draft = _draft((rejected,))

    assert draft.claims_unsupported == 1
    assert draft.claims[0].reject_code is RejectCode.QUOTE_NOT_FOUND


def test_open_questions_section_is_always_present() -> None:
    # Единственное место, где сервис говорит о недостатке данных. Пустым оно
    # не бывает: молчание читалось бы как «вопросов нет».
    draft = _draft((_claim(section=ClaimSection.PARTIES),))

    assert "Открытые вопросы" in draft.body


def test_sections_follow_the_declared_order() -> None:
    dates = _claim(index=0, section=ClaimSection.DATES, text="Срок поставки — 30 дней.")
    parties = _claim(
        index=1, section=ClaimSection.PARTIES, text="Поставщик — «Вектор»."
    )

    body = _draft((dates, parties)).body

    assert body.index("Поставщик") < body.index("Срок поставки")


def test_draft_without_supported_claims_reports_insufficient_evidence() -> None:
    rejected = _claim(
        index=0, citations=(), supported=False, reject_code=RejectCode.NO_CITATION
    )

    draft = _draft((rejected,))

    assert draft.status is DraftStatus.INSUFFICIENT_EVIDENCE


def test_draft_with_at_least_one_supported_claim_is_generated() -> None:
    assert _draft((_claim(),)).status is DraftStatus.GENERATED


def test_groundedness_is_the_share_of_supported_claims() -> None:
    supported = _claim(index=0)
    rejected = _claim(
        index=1, citations=(), supported=False, reject_code=RejectCode.NO_CITATION
    )

    draft = _draft((supported, rejected))

    assert draft.groundedness == Ratio(0.5)


def test_counters_add_up() -> None:
    draft = _draft(
        (
            _claim(index=0),
            _claim(
                index=1,
                citations=(),
                supported=False,
                reject_code=RejectCode.NO_CITATION,
            ),
        )
    )

    assert draft.claims_total == draft.claims_grounded + draft.claims_unsupported


def test_draft_key_is_determined_by_request_and_prompt_version() -> None:
    request_id = RequestId.generate()

    draft = Draft.assembled(
        request_id=request_id,
        document_id=make_chunk().ref.document_id,
        draft_type=DraftType.CASE_FACT_SUMMARY,
        query="Собери сводку фактов по делу",
        claims=(_claim(),),
        model_name="deepseek-ai/DeepSeek-V4-Flash",
        prompt_version=PROMPT_VERSION,
        retrieval_profile="hybrid-rrf-v1",
        embedding_version=EmbeddingVersion(1, 0, 0),
        chunking_version=ChunkingVersion(1, 0, 0),
        evidence_total=7,
    )

    assert draft.id == DraftId.deterministic(
        request_id=request_id, prompt_version=PROMPT_VERSION
    )


def test_claim_indexes_must_not_repeat() -> None:
    with pytest.raises(InvariantViolation):
        _draft((_claim(index=0), _claim(index=0)))


def test_empty_query_is_refused() -> None:
    with pytest.raises(InvariantViolation):
        Draft.assembled(
            request_id=RequestId.generate(),
            document_id=make_chunk().ref.document_id,
            draft_type=DraftType.CASE_FACT_SUMMARY,
            query="  ",
            claims=(_claim(),),
            model_name="deepseek-ai/DeepSeek-V4-Flash",
            prompt_version=PROMPT_VERSION,
            retrieval_profile="hybrid-rrf-v1",
            embedding_version=EmbeddingVersion(1, 0, 0),
            chunking_version=ChunkingVersion(1, 0, 0),
            evidence_total=7,
        )
