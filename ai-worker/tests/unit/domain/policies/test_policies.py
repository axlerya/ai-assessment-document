"""Политики домена: чистые функции, возвращающие вердикт."""

from __future__ import annotations

import inspect

import pytest

from ai_worker.domain import policies
from ai_worker.domain.entities.draft import Citation
from ai_worker.domain.policies.claim_grounding import (
    ClaimGroundingPolicy,
    ClaimVerdict,
)
from ai_worker.domain.policies.draft_completeness import DraftCompletenessPolicy
from ai_worker.domain.policies.evidence_reliability import EvidenceReliabilityPolicy
from ai_worker.domain.value_objects.enums import (
    ClaimSection,
    ExtractionMethod,
    RejectCode,
)
from ai_worker.domain.value_objects.identifiers import ChunkId, ClaimId
from ai_worker.domain.value_objects.scores import Score
from ai_worker.domain.value_objects.text import QuoteSpan
from tests.factories import make_chunk, make_quality

pytestmark = pytest.mark.unit

QUOTE = "Договор поставки № 12/АБ"
QUOTE_SPAN = QuoteSpan(start=0, end=len(QUOTE))
RELIABILITY = EvidenceReliabilityPolicy(min_citable_confidence=0.60)
GROUNDING = ClaimGroundingPolicy()


def _citation(*, claim_id: ClaimId, reliable: bool = True) -> Citation:
    return Citation.for_quote(
        claim_id=claim_id,
        chunk=make_chunk(),
        span=QUOTE_SPAN,
        quote=QUOTE,
        retrieval_score=Score(0.9),
        rerank_score=Score(2.1),
        reliable=reliable,
    )


def test_policies_return_verdicts_and_do_not_construct_entities() -> None:
    # Политика, порождающая сущность, рано или поздно порождает такую, которую
    # сама сущность обязана отвергнуть. Вердикт этого класса ошибок не знает.
    forbidden = {"Claim", "Citation", "Draft", "Evidence", "ChunkEmbedding"}

    for module in policies.ALL:
        source = inspect.getsource(module)
        built = {name for name in forbidden if f"{name}(" in source}
        assert not built, f"{module.__name__} строит сущности: {sorted(built)}"


def test_text_layer_chunk_is_reliable() -> None:
    verdict = RELIABILITY.judge(make_quality())

    assert verdict.reliable
    assert verdict.reason is None


def test_reliability_policy_marks_chunk_with_illegible_spans() -> None:
    # Устав запрещает выдавать нераспознанное за распознанное: такой фрагмент
    # доходит до оператора, но опереть на него утверждение нельзя.
    quality = make_quality(
        method=ExtractionMethod.OCR, confidence=0.95, illegible_span_count=1
    )

    verdict = RELIABILITY.judge(quality)

    assert not verdict.reliable
    assert verdict.reason == "illegible_spans"


def test_low_confidence_chunk_is_not_citable() -> None:
    quality = make_quality(method=ExtractionMethod.OCR, confidence=0.31)

    verdict = RELIABILITY.judge(quality)

    assert not verdict.reliable
    assert verdict.reason == "low_confidence"


def test_confidence_exactly_at_the_threshold_is_citable() -> None:
    # Порог включающий: иначе калибровка сдвигала бы границу дважды.
    quality = make_quality(method=ExtractionMethod.OCR, confidence=0.60)

    assert RELIABILITY.judge(quality).reliable


def test_threshold_outside_zero_to_one_is_refused() -> None:
    with pytest.raises(ValueError, match="порог"):
        EvidenceReliabilityPolicy(min_citable_confidence=1.5)


def test_claim_without_citations_is_not_grounded() -> None:
    verdict = GROUNDING.judge(citations=(), context_chunk_ids=frozenset())

    assert verdict == ClaimVerdict(supported=False, reject_code=RejectCode.NO_CITATION)


def test_claim_with_only_unreliable_citations_is_not_grounded() -> None:
    claim_id = ClaimId.generate()
    citation = _citation(claim_id=claim_id, reliable=False)

    verdict = GROUNDING.judge(
        citations=(citation,), context_chunk_ids=frozenset({citation.ref.chunk_id})
    )

    assert not verdict.supported
    assert verdict.reject_code is RejectCode.UNRELIABLE_EVIDENCE_ONLY


def test_claim_citing_a_chunk_outside_the_context_is_not_grounded() -> None:
    # Ссылка на чанк, которого модель не видела, — выдумка со ссылкой на
    # существующий документ: самый правдоподобный и потому опасный случай.
    claim_id = ClaimId.generate()
    citation = _citation(claim_id=claim_id)

    verdict = GROUNDING.judge(
        citations=(citation,), context_chunk_ids=frozenset({ChunkId.generate()})
    )

    assert not verdict.supported
    assert verdict.reject_code is RejectCode.CHUNK_NOT_IN_CONTEXT


def test_claim_with_one_reliable_citation_is_grounded() -> None:
    claim_id = ClaimId.generate()
    reliable = _citation(claim_id=claim_id)
    unreliable = _citation(claim_id=claim_id, reliable=False)

    verdict = GROUNDING.judge(
        citations=(reliable, unreliable),
        context_chunk_ids=frozenset({reliable.ref.chunk_id, unreliable.ref.chunk_id}),
    )

    assert verdict == ClaimVerdict(supported=True, reject_code=None)


def test_verdict_of_a_supported_claim_carries_no_reason() -> None:
    claim_id = ClaimId.generate()
    citation = _citation(claim_id=claim_id)

    verdict = GROUNDING.judge(
        citations=(citation,), context_chunk_ids=frozenset({citation.ref.chunk_id})
    )

    assert verdict.reject_code is None


def test_completeness_policy_requires_the_open_questions_section() -> None:
    verdict = DraftCompletenessPolicy().judge(
        sections=frozenset({ClaimSection.PARTIES, ClaimSection.DATES})
    )

    assert not verdict.complete
    assert ClaimSection.OPEN_QUESTIONS in verdict.missing


def test_draft_with_open_questions_is_complete() -> None:
    verdict = DraftCompletenessPolicy().judge(
        sections=frozenset({ClaimSection.OPEN_QUESTIONS})
    )

    assert verdict.complete
    assert verdict.missing == ()
