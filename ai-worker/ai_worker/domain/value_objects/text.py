"""Диапазон цитаты внутри текста чанка.

Ссылки на чанк целиком недостаточно: фрагмент в восемьсот токенов подтверждает
почти любое утверждение. Диапазон превращает ссылку в проверяемое утверждение
«вот эти символы вот этого чанка», и проверка становится машинной.

Смещения — в кодовых точках Unicode, как и у чанков document-worker. Считать в
байтах значит промахиваться цитатой на каждом русском документе.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_worker.domain.errors import InvalidTextSpan


@dataclass(frozen=True, slots=True)
class QuoteSpan:
    """Полуинтервал `[start, end)` внутри текста чанка."""

    start: int
    end: int

    def __post_init__(self) -> None:
        """Отвергает вырожденный и перевёрнутый диапазон."""
        if self.start < 0:
            self._reject("начало диапазона отрицательно")
        if self.end <= self.start:
            self._reject("конец диапазона не больше начала")

    def _reject(self, reason: str) -> None:
        raise InvalidTextSpan(
            f"некорректный диапазон цитаты: {reason}",
            context={"start": self.start, "end": self.end, "reason": reason},
        )

    @property
    def length(self) -> int:
        """Длина цитаты в кодовых точках."""
        return self.end - self.start

    def slice_of(self, text: str) -> str:
        """Вырезает свой фрагмент текста.

        Raises:
            InvalidTextSpan: Диапазон выходит за пределы текста — то есть
                указывает на символы, которых в источнике нет.
        """
        if self.end > len(text):
            self._reject(f"диапазон выходит за пределы текста длиной {len(text)}")
        return text[self.start : self.end]

    def matches(self, text: str, *, quote: str) -> bool:
        """Совпадает ли цитата со срезом текста.

        Возвращает вердикт, а не исключение: несовпавшая цитата — штатный исход
        верификации, при котором утверждение просто не публикуется. Выход за
        пределы текста здесь тоже несовпадение, а не ошибка: и то и другое
        означает, что подтверждения нет.
        """
        if self.end > len(text):
            return False
        return text[self.start : self.end] == quote
