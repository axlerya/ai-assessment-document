"""Конвейер чанкования документа.

Обработка идёт страница за страницей в порядке номеров, сквозной «поток текста
документа» не материализуется: чанк не пересекает границу страницы, поэтому
конкатенация дала бы только второй набор смещений, который пришлось бы
переводить обратно в page-relative.

Через границу страницы переносится ровно одно состояние — стек секций.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from document_worker.domain.chunking.chunk_assembler import ChunkAssembler
from document_worker.domain.chunking.line_classifier import LineClassifier
from document_worker.domain.chunking.paragraph_splitter import ParagraphSplitter
from document_worker.domain.chunking.quality import only_chunk_of_document
from document_worker.domain.chunking.sentence_splitter import SentenceSplitter
from document_worker.domain.chunking.structure_detector import StructureDetector

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_worker.domain.chunking.chunk_assembler import ChunkDraft
    from document_worker.domain.chunking.line_classifier import LayoutLine
    from document_worker.domain.chunking.policy import ChunkingPolicy
    from document_worker.domain.chunking.structure_detector import SectionNode
    from document_worker.domain.entities.document_page import DocumentPage
    from document_worker.domain.interfaces.token_counter import TokenCounter
    from document_worker.domain.value_objects.versioning import ChunkingVersion

    _PageLines = tuple[DocumentPage, tuple[LayoutLine, ...], tuple[SectionNode, ...]]


@dataclass(frozen=True, slots=True)
class ChunkingPipeline:
    """Превращает сохранённые страницы в черновики чанков."""

    policy: ChunkingPolicy
    classifier: LineClassifier
    detector: StructureDetector
    paragraphs: ParagraphSplitter
    assembler: ChunkAssembler

    @property
    def version(self) -> ChunkingVersion:
        """Версия чанкования, которой получен результат."""
        return self.policy.version

    @property
    def params_hash(self) -> str:
        """Хэш параметров: по нему в логах видно, чем получен набор чанков."""
        return self.policy.params_hash()

    def run(self, pages: Sequence[DocumentPage]) -> tuple[ChunkDraft, ...]:
        """Разбивает страницы на чанки.

        Страницы без текста чанков не порождают: пустой чанк запрещён и
        сущностью, и ограничением схемы. Пустой результат — не ошибка.
        """
        usable = sorted(
            (page for page in pages if page.text.content.strip()),
            key=lambda page: int(page.number),
        )
        per_page = [self.classifier.classify_page(page) for page in usable]
        drafts: list[ChunkDraft] = []
        for page, lines, sections in self._with_sections(usable, per_page):
            blocks = self.paragraphs.split(lines, sections, content=page.text.content)
            drafts.extend(self.assembler.assemble(page, blocks))
        return _with_single_chunk_exception(drafts)

    def _with_sections(
        self,
        pages: Sequence[DocumentPage],
        per_page: Sequence[tuple[LayoutLine, ...]],
    ) -> list[_PageLines]:
        """Раскладывает сквозное дерево секций обратно по страницам."""
        sections = self.detector.detect([line for lines in per_page for line in lines])
        result: list[_PageLines] = []
        offset = 0
        for page, lines in zip(pages, per_page, strict=True):
            result.append((page, lines, sections[offset : offset + len(lines)]))
            offset += len(lines)
        return result


def build_pipeline(
    policy: ChunkingPolicy,
    token_counter: TokenCounter,
) -> ChunkingPipeline:
    """Собирает конвейер со всеми его частями."""
    return ChunkingPipeline(
        policy=policy,
        classifier=LineClassifier(),
        detector=StructureDetector(),
        paragraphs=ParagraphSplitter(),
        assembler=ChunkAssembler(
            policy=policy,
            token_counter=token_counter,
            sentences=SentenceSplitter(),
        ),
    )


def _with_single_chunk_exception(drafts: list[ChunkDraft]) -> tuple[ChunkDraft, ...]:
    """Единственный чанк документа индексируется, даже если он короткий."""
    if len(drafts) != 1:
        return tuple(drafts)
    only = drafts[0]
    return (replace(only, quality=only_chunk_of_document(only.quality)),)
