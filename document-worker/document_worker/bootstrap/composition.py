"""Композиционный корень: единственное место, где создаются конкретные классы.

Контейнера внедрения зависимостей здесь нет намеренно. Он был бы нужен ради
области видимости «на сообщение», а её в этом сервисе не существует: сессия
базы живёт внутри транзакции, а не внутри обработки документа — обработка
идёт минутами, транзакция миллисекунды, и связывать их сроком жизни как раз
запрещено. Всё остальное — единственные экземпляры, которые собираются один
раз на старте, и для этого достаточно функции.

Ресурсы закрываются в обратном порядке через стек: пул процессов, соединение
с брокером и движок базы обязаны освободиться даже если старт оборвался
посередине.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_worker.application.services.message_claim import MessageClaimService
from document_worker.application.services.page_runner import PageSequenceRunner
from document_worker.application.services.source_loader import SourceDocumentLoader
from document_worker.application.use_cases.complete_document_processing import (
    CompleteDocumentProcessing,
)
from document_worker.application.use_cases.extract_document_text import (
    ExtractDocumentText,
)
from document_worker.application.use_cases.fail_document_processing import (
    FailDocumentProcessing,
)
from document_worker.application.use_cases.process_document import ProcessDocument
from document_worker.application.use_cases.process_document_page import (
    ProcessDocumentPage,
)
from document_worker.domain.normalization.normalizer import TextNormalizer
from document_worker.domain.policies.document_status import DocumentStatusPolicy
from document_worker.domain.policies.text_layer_quality import TextLayerQualityPolicy
from document_worker.infrastructure.cpu.executor import CpuPool
from document_worker.infrastructure.messaging.broker import build_broker
from document_worker.infrastructure.messaging.declare import declare_topology
from document_worker.infrastructure.messaging.retry_publisher import RetryPublisher
from document_worker.infrastructure.messaging.topology import (
    RETRY_LADDER,
    build_topology,
)
from document_worker.infrastructure.pdf.pdfplumber_text_reader import (
    PdfPlumberDocumentReader,
)
from document_worker.infrastructure.pdf.pikepdf_inspector import PikePdfInspector
from document_worker.infrastructure.pdf.pypdfium2_page_renderer import (
    PdfiumPageRenderer,
)
from document_worker.infrastructure.persistence.engine import (
    build_engine,
    build_session_factory,
    build_unit_of_work_factory,
)
from document_worker.infrastructure.storage.s3_object_storage import (
    S3Config,
    S3ObjectStorage,
)
from document_worker.infrastructure.storage.temp_workspace import (
    TempDirWorkspaceFactory,
)
from document_worker.infrastructure.system.clock import SystemClock, Uuid4IdGenerator
from document_worker.presentation.messaging.subscribers.process_document import (
    build_process_document_router,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from faststream.rabbit import RabbitBroker, RabbitRouter

    from document_worker.infrastructure.config.settings import AppSettings


@dataclass(frozen=True, slots=True)
class Services:
    """Собранный сервис: всё, что нужно точке входа."""

    settings: AppSettings
    broker: RabbitBroker
    router: RabbitRouter
    process_document: ProcessDocument


@contextlib.asynccontextmanager
async def build_services(settings: AppSettings) -> AsyncIterator[Services]:
    """Собирает сервис и освобождает его ресурсы при любом исходе."""
    config = settings.processing_config()
    clock = SystemClock()
    ids = Uuid4IdGenerator()

    async with contextlib.AsyncExitStack() as stack:
        engine = build_engine(
            settings.database.dsn.get_secret_value(),
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_timeout_s=settings.database.pool_timeout_s,
        )
        stack.push_async_callback(engine.dispose)
        uow_factory = build_unit_of_work_factory(build_session_factory(engine))

        storage = await stack.enter_async_context(S3ObjectStorage(_s3_config(settings)))
        pool = await stack.enter_async_context(
            CpuPool(max_workers=settings.processing.cpu_workers)
        )
        broker = build_broker(
            settings.rabbit.url.get_secret_value(),
            graceful_timeout_s=settings.rabbit.graceful_timeout_s,
        )
        stack.push_async_callback(broker.stop)

        topology = build_topology(
            consumer_timeout_ms=settings.rabbit.consumer_timeout_ms,
            delivery_limit=settings.rabbit.delivery_limit,
            declare_audit_queue=settings.rabbit.declare_audit_queue,
        )
        process_document = ProcessDocument(
            claim_service=MessageClaimService(
                uow_factory=uow_factory, clock=clock, ids=ids, config=config
            ),
            source_loader=SourceDocumentLoader(storage=storage, config=config),
            extract_text=ExtractDocumentText(
                inspector=PikePdfInspector(
                    pool=pool, max_pages=settings.processing.max_pages
                ),
                reader=PdfPlumberDocumentReader(pool=pool),
                renderer=PdfiumPageRenderer(
                    pool=pool, max_pixels=settings.processing.max_render_pixels
                ),
                policy=TextLayerQualityPolicy(),
            ),
            page_runner=PageSequenceRunner(
                process_page=ProcessDocumentPage(
                    uow_factory=uow_factory,
                    normalizer=TextNormalizer(),
                    ids=ids,
                    clock=clock,
                    config=config,
                )
            ),
            complete=CompleteDocumentProcessing(
                uow_factory=uow_factory,
                status_policy=DocumentStatusPolicy(),
                clock=clock,
                config=config,
            ),
            fail=FailDocumentProcessing(
                uow_factory=uow_factory, clock=clock, config=config
            ),
            workspaces=TempDirWorkspaceFactory(base_dir=_temp_root(settings)),
            config=config,
        )
        router = build_process_document_router(
            queue=topology.process_requested,
            exchange=topology.commands,
            processor=process_document,
            retrier=RetryPublisher(
                broker=broker,
                topology=topology,
                publish_timeout_s=settings.rabbit.publish_timeout_s,
            ),
            default_bucket=settings.s3.default_bucket,
            max_retries=len(RETRY_LADDER),
        )
        broker.include_router(router)
        await broker.connect()
        # Топология объявляется до старта потребителей: подписчик на очередь,
        # которой ещё нет, поднимет её с чужими аргументами.
        await declare_topology(broker, topology)
        yield Services(
            settings=settings,
            broker=broker,
            router=router,
            process_document=process_document,
        )


def _s3_config(settings: AppSettings) -> S3Config:
    return S3Config(
        endpoint_url=settings.s3.endpoint_url,
        region=settings.s3.region,
        access_key=settings.s3.access_key.get_secret_value(),
        secret_key=settings.s3.secret_key.get_secret_value(),
        read_timeout_s=settings.s3.download_timeout_s,
    )


def _temp_root(settings: AppSettings) -> Path:
    root = settings.processing.temp_dir
    root.mkdir(parents=True, exist_ok=True)
    return root
