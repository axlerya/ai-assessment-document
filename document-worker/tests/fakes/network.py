"""Управляемый разрыв связи с внешним сервисом.

Недоступность хранилища на сквозном уровне не изобразить подменой порта в
настройках: адрес адаптер получает один раз при сборке сервиса, а сборка тут
настоящая. Поэтому сервис ходит через посредника, который стоит на своём
адресе всегда и по команде сценария рвёт связь.

Обрыв доводится до конца: живое keep-alive соединение переживает закрытие
порта, и не оборвав его, сценарий проверял бы работающее хранилище.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

RELAY_BUFFER = 64 * 1024


class BreakableLink:
    """Пересылает байты до целевого адреса, пока её не порвут."""

    # Присваивается в `start`: посредник без своего порта бесполезен, и все
    # его пользователи получают уже поднятый.
    _server: asyncio.Server

    def __init__(self, host: str, port: int) -> None:
        """Запоминает, куда пересылать; порт открывает `start`."""
        self._target = (host, port)
        self._live: set[asyncio.StreamWriter] = set()
        self._broken = False

    @property
    def url(self) -> str:
        """Адрес посредника — его и получает проверяемая сторона."""
        host, port = self._server.sockets[0].getsockname()[:2]
        return f"http://{host}:{port}"

    async def start(self) -> None:
        """Открывает порт посредника."""
        self._server = await asyncio.start_server(self._relay, "127.0.0.1", 0)

    async def stop(self) -> None:
        """Закрывает порт вместе со всеми живыми соединениями."""
        self._cut_live()
        self._server.close()
        await self._server.wait_closed()

    @contextlib.contextmanager
    def broken(self) -> Iterator[None]:
        """Рвёт связь на время блока."""
        self._broken = True
        self._cut_live()
        try:
            yield
        finally:
            self._broken = False

    async def _relay(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if self._broken:
            await _close(writer)
            return
        upstream_reader, upstream_writer = await asyncio.open_connection(*self._target)
        self._live |= {writer, upstream_writer}
        try:
            await asyncio.gather(
                _pump(reader, upstream_writer),
                _pump(upstream_reader, writer),
            )
        finally:
            self._live -= {writer, upstream_writer}

    def _cut_live(self) -> None:
        for writer in tuple(self._live):
            writer.close()


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Льёт прочитанное в другую сторону, закрывая её на выходе.

    Закрытие обязательно: без него встречный поток ждал бы на живом
    keep-alive соединении, которому уже некому отвечать.
    """
    try:
        while chunk := await reader.read(RELAY_BUFFER):
            writer.write(chunk)
            await writer.drain()
    except OSError:
        # Оборванное соединение — ожидаемый исход, а не поломка посредника.
        pass
    finally:
        await _close(writer)


async def _close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(OSError):
        await writer.wait_closed()
