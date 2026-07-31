"""Сборка блоков страницы в чанки.

Смена секции — жёсткое условие разрыва, недобором до целевого размера оно не
подавляется: правило устава «chunk не должен объединять несвязанные разделы
только ради достижения фиксированного размера» живёт здесь.

Заголовок секции в текст чанка не входит: иначе текст перестал бы быть точным
срезом страницы, а вместе с ним отвалилось бы всё цитирование. Заголовок
сохраняется последним элементом heading_path.

Перекрытие делается сдвигом начала влево, а не копированием текста в отдельное
поле — по той же причине.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from document_worker.domain.chunking.blocks import BlockKind
from document_worker.domain.chunking.quality import ChunkQualityEvaluator
from document_worker.domain.constants import MAX_CHUNK_OVERLAP_CHARS
from document_worker.domain.value_objects.text import TextSpan

if TYPE_CHECKING:
    from collections.abc import Sequence

    from document_worker.domain.chunking.blocks import Block
    from document_worker.domain.chunking.policy import ChunkingPolicy
    from document_worker.domain.chunking.quality import ChunkQuality
    from document_worker.domain.chunking.sentence_splitter import SentenceSplitter
    from document_worker.domain.entities.document_page import DocumentPage
    from document_worker.domain.interfaces.token_counter import TokenCounter
    from document_worker.domain.value_objects.identifiers import PageId
    from document_worker.domain.value_objects.paging import PageNumber


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """Готовый к сохранению фрагмент страницы; сущность собирает application."""

    page_id: PageId
    page_number: PageNumber
    ordinal: int
    span: TextSpan
    text: str
    token_count: int
    heading_path: tuple[str, ...]
    overlap_prefix_chars: int
    kind: BlockKind
    quality: ChunkQuality


def fitting_prefix_length(text: str, *, counter: TokenCounter, limit: int) -> int:
    """Наибольшая длина префикса, влезающая в предел токенов.

    Двоичный поиск, а не наращивание по символу: последнее дало бы двадцать
    тысяч вызовов токенизатора на один жёсткий разрез.
    """
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if counter.count(text[:middle]) <= limit:
            low = middle
        else:
            high = middle - 1
    return low


@dataclass(frozen=True, slots=True)
class ChunkAssembler:
    """Собирает чанки одной страницы из её блоков."""

    policy: ChunkingPolicy
    token_counter: TokenCounter
    sentences: SentenceSplitter
    quality: ChunkQualityEvaluator = field(default_factory=ChunkQualityEvaluator)

    def assemble(
        self,
        page: DocumentPage,
        blocks: Sequence[Block],
    ) -> tuple[ChunkDraft, ...]:
        """Режет длинные блоки, пакует их в чанки и добавляет перекрытие."""
        content = page.text.content
        pieces = [
            piece for block in blocks for piece in self._fit_block(content, block)
        ]
        chunks = self._merge_short(content, self._pack(content, pieces))
        return self._drafts(page, chunks)

    def _tokens(self, content: str, span: TextSpan) -> int:
        return self.token_counter.count(span.slice_of(content))

    def _fit_block(self, content: str, block: Block) -> list[Block]:
        """Режет блок каскадом, пока каждая часть не влезет в предел."""
        if (
            block.kind is BlockKind.HEADING
            or self._tokens(content, block.span) <= self.policy.max_tokens
        ):
            return [block]
        for boundaries in self._boundary_levels(content, block):
            parts = self._cut_at(content, block.span, boundaries)
            if parts is not None:
                return _rebuilt(block, parts, content)
        return _rebuilt(block, self._hard_cut(content, block.span), content)

    def _boundary_levels(
        self,
        content: str,
        block: Block,
    ) -> tuple[tuple[int, ...], ...]:
        """Границы реза по убыванию предпочтительности."""
        text = block.span.slice_of(content)
        offset = block.span.start
        primary = (
            _row_boundaries(text)
            if block.kind is BlockKind.TABLE
            else self.sentences.boundaries(text)
        )
        return (
            tuple(offset + at for at in primary),
            tuple(offset + at for at in _word_boundaries(text)),
        )

    def _cut_at(
        self,
        content: str,
        span: TextSpan,
        boundaries: Sequence[int],
    ) -> list[TextSpan] | None:
        """Режет диапазон по этим границам; None — уровень не помог."""
        if not boundaries:
            return None
        stops = [*boundaries, span.end]
        parts: list[TextSpan] = []
        start, index = span.start, 0
        while start < span.end:
            chosen: int | None = None
            while index < len(stops):
                stop = stops[index]
                if stop > start and not self._fits(content, TextSpan(start, stop)):
                    break
                chosen = stop if stop > start else chosen
                index += 1
            if chosen is None:
                return None
            parts.append(TextSpan(start, chosen))
            start = chosen
        return parts

    def _fits(self, content: str, span: TextSpan) -> bool:
        return self._tokens(content, span) <= self.policy.max_tokens

    def _hard_cut(self, content: str, span: TextSpan) -> list[TextSpan]:
        """Последнее средство: сплошная последовательность без единой границы."""
        parts: list[TextSpan] = []
        start = span.start
        while start < span.end:
            length = max(
                1,
                fitting_prefix_length(
                    content[start : span.end],
                    counter=self.token_counter,
                    limit=self.policy.max_tokens,
                ),
            )
            parts.append(TextSpan(start, start + length))
            start += length
        return parts

    def _pack(self, content: str, pieces: Sequence[Block]) -> list[Block]:
        """Складывает блоки в чанки, не переступая границу секции."""
        chunks: list[Block] = []
        current: Block | None = None
        for piece in pieces:
            current = self._absorb(content, chunks, current, piece)
        _push(chunks, current)
        return chunks

    def _absorb(
        self,
        content: str,
        chunks: list[Block],
        current: Block | None,
        piece: Block,
    ) -> Block | None:
        """Продолжает текущий чанк этим блоком либо закрывает его."""
        if piece.kind is BlockKind.HEADING:
            # Заголовок в текст чанка не входит, но разрывает предыдущий.
            _push(chunks, current)
            return None
        if current is None:
            return piece
        if self._must_break(content, current, piece) or not self._fits_target(
            content, current, piece
        ):
            chunks.append(current)
            return piece
        return _joined(current, piece)

    def _must_break(self, content: str, current: Block, piece: Block) -> bool:
        return (
            piece.section_key != current.section_key
            or piece.is_atomic
            or current.is_atomic
            or self._tokens(content, current.span) >= self.policy.target_tokens
        )

    def _fits_target(self, content: str, current: Block, piece: Block) -> bool:
        joined = TextSpan(current.span.start, piece.span.end)
        return self._tokens(content, joined) <= self.policy.target_tokens

    def _merge_short(self, content: str, chunks: Sequence[Block]) -> list[Block]:
        """Приклеивает короткий чанк к соседу той же страницы и той же секции."""
        merged: list[Block] = []
        for chunk in chunks:
            previous = merged[-1] if merged else None
            if previous is not None and self._should_merge(content, previous, chunk):
                merged[-1] = _joined(previous, chunk)
            else:
                merged.append(chunk)
        return merged

    def _should_merge(self, content: str, previous: Block, chunk: Block) -> bool:
        if previous.section_key != chunk.section_key:
            return False
        if previous.is_atomic or chunk.is_atomic:
            return False
        shortest = min(
            self._tokens(content, previous.span), self._tokens(content, chunk.span)
        )
        if shortest >= self.policy.min_tokens:
            return False
        return self._fits(content, TextSpan(previous.span.start, chunk.span.end))

    def _drafts(
        self,
        page: DocumentPage,
        chunks: Sequence[Block],
    ) -> tuple[ChunkDraft, ...]:
        content = page.text.content
        drafts: list[ChunkDraft] = []
        previous: tuple[Block, ChunkDraft] | None = None
        for chunk in chunks:
            overlap = self._overlap_for(page, chunk, previous)
            span = TextSpan(chunk.span.start - overlap, chunk.span.end)
            text = span.slice_of(content)
            if not text.strip():
                continue
            draft = self._draft(page, chunk, span=span, ordinal=len(drafts))
            drafts.append(draft)
            previous = (chunk, draft)
        return tuple(drafts)

    def _draft(
        self,
        page: DocumentPage,
        chunk: Block,
        *,
        span: TextSpan,
        ordinal: int,
    ) -> ChunkDraft:
        text = span.slice_of(page.text.content)
        overlap = chunk.span.start - span.start
        own_tokens = self.token_counter.count(text[overlap:])
        return ChunkDraft(
            page_id=page.id,
            page_number=page.number,
            ordinal=ordinal,
            span=span,
            text=text,
            token_count=self.token_counter.count(text),
            heading_path=chunk.heading_path,
            overlap_prefix_chars=overlap,
            kind=chunk.kind,
            quality=self.quality.evaluate(page=page, span=span, own_tokens=own_tokens),
        )

    def _overlap_for(
        self,
        page: DocumentPage,
        chunk: Block,
        previous: tuple[Block, ChunkDraft] | None,
    ) -> int:
        """Перекрытие с предыдущим чанком той же страницы и той же секции."""
        if previous is None or not self._overlap_allowed(chunk, previous):
            return 0
        block, draft = previous
        content = page.text.content
        # Начало обязано строго возрастать: совпадение начал двух чанков одной
        # страницы падает с 23505 по uq__document_chunks__page__start.
        lower_bound = draft.span.start + draft.overlap_prefix_chars + 1
        head = content[block.span.start : chunk.span.start]
        for at in self.sentences.boundaries(head):
            start = block.span.start + at
            if start >= lower_bound and self._overlap_fits(content, start, chunk):
                return chunk.span.start - start
        return 0

    def _overlap_allowed(
        self,
        chunk: Block,
        previous: tuple[Block, ChunkDraft],
    ) -> bool:
        block, draft = previous
        if chunk.section_key != block.section_key:
            return False
        if chunk.is_atomic or block.is_atomic:
            return False
        # Затягивать в чанк нераспознанный хвост соседа смысла нет.
        return not draft.quality.is_fully_illegible

    def _overlap_fits(self, content: str, start: int, chunk: Block) -> bool:
        if chunk.span.start - start > MAX_CHUNK_OVERLAP_CHARS:
            return False
        if (
            self._tokens(content, TextSpan(start, chunk.span.start))
            > self.policy.overlap_tokens
        ):
            return False
        return self._fits(content, TextSpan(start, chunk.span.end))


def _push(chunks: list[Block], current: Block | None) -> None:
    if current is not None:
        chunks.append(current)


def _joined(first: Block, second: Block) -> Block:
    return replace(first, span=TextSpan(first.span.start, second.span.end))


def _rebuilt(block: Block, parts: Sequence[TextSpan], content: str) -> list[Block]:
    return [
        replace(block, span=trimmed)
        for part in parts
        if not (trimmed := _trimmed(part, content)).is_empty
    ]


def _row_boundaries(text: str) -> tuple[int, ...]:
    return tuple(index + 1 for index, char in enumerate(text) if char == "\n")


def _word_boundaries(text: str) -> tuple[int, ...]:
    return tuple(
        index
        for index in range(1, len(text))
        if text[index - 1].isspace() and not text[index].isspace()
    )


def _trimmed(span: TextSpan, content: str) -> TextSpan:
    start, end = span.start, span.end
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    return TextSpan(start, end)
