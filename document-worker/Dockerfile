# Сборка идёт из каталога сервиса, а не из корня репозитория: рабочее
# пространство uv в контексте сборки тянуло бы за собой соседние проекты.
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Зависимости ставятся отдельным слоем: код меняется на каждом коммите,
# зависимости — раз в недели.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --no-editable

COPY document_worker ./document_worker
COPY migrations ./migrations
COPY scripts ./scripts
COPY alembic.ini ./
# `--no-editable` обязателен: ссылка на исходники вместо копии оставляет
# рабочий образ зависимым от каталога сборки.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Словарь BPE и модели распознавания укладываются в образ здесь: сеть в
# рантайме запрещена, а первый же документ иначе полез бы её скачивать в
# середине обработки.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache \
    OCR__MODEL_DIR=/opt/ocr-models
RUN uv run --no-dev --no-sync python scripts/prefetch_models.py \
    && test -n "$(ls -A /opt/tiktoken-cache)" \
    && test -n "$(ls -A /opt/ocr-models)"


FROM python:3.12-slim-bookworm AS runtime

# tini как первый процесс: без него сигнал остановки не доходит до питона,
# и оркестратор убивает контейнер по таймауту вместо аккуратной остановки.
RUN apt-get update \
    && apt-get install --no-install-recommends -y tini \
    && rm -rf /var/lib/apt/lists/*

# Непривилегированный пользователь с фиксированным идентификатором: под
# случайным номером тома монтируются с чужими правами.
RUN groupadd --gid 10001 worker \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin worker

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Библиотеки разбора PDF раскладывают работу по потокам сами, а мы уже
    # держим пул процессов: вложенный параллелизм только конкурирует за ядра.
    OMP_NUM_THREADS=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache \
    OCR__MODEL_DIR=/opt/ocr-models \
    PROCESSING__TEMP_DIR=/tmp/document-worker

WORKDIR /app

COPY --from=builder --chown=10001:10001 /app/.venv /app/.venv
COPY --from=builder --chown=10001:10001 /app/document_worker /app/document_worker
COPY --from=builder --chown=10001:10001 /app/migrations /app/migrations
COPY --from=builder --chown=10001:10001 /app/alembic.ini /app/alembic.ini
COPY --from=builder --chown=10001:10001 /opt/tiktoken-cache /opt/tiktoken-cache
COPY --from=builder --chown=10001:10001 /opt/ocr-models /opt/ocr-models

USER 10001:10001

EXPOSE 8080

# Проба ходит стандартной библиотекой: curl в образе — лишние мегабайты и
# лишняя поверхность.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health/ready', timeout=4).status == 200 else 1)"]

ENTRYPOINT ["/usr/bin/tini", "-g", "--"]
CMD ["document-worker"]
