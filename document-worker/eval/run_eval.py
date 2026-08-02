"""Прогон корпуса и отчёт.

Коды возврата разведены намеренно: «корпус подменён» — это не то же самое, что
«качество упало», и чинится по-разному. При несовпадении отпечатка метрики не
сравниваются вообще: сравнивать числа с числами по другому корпусу — значит
делать вид, что сравнение состоялось.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from document_worker.bootstrap.composition import build_processing
from document_worker.domain.chunking.policy import CHUNKING_VERSION
from document_worker.infrastructure.config.settings import AppSettings
from eval import corpus, fonts
from eval.local_storage import LocalCorpusStorage
from eval.runner import process_document
from eval.scoring import aggregate, by_category, score_document

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from eval.runner import ChunkOutcome
    from eval.scoring import PageScore

EXIT_OK: Final[int] = 0
EXIT_REGRESSION: Final[int] = 2
EXIT_THRESHOLD: Final[int] = 3
EXIT_CORPUS_CHANGED: Final[int] = 4

# Допуски: время распознавания и порядок строк не идеально повторяются между
# машинами, и без запаса гейт краснел бы от смены раннера, а не от качества.
TOLERANCE_WORSE: Final[float] = 0.02
TOLERANCE_BETTER: Final[float] = 0.03
# Инварианты допуска не имеют: ненулевое значение — дефект, а не деградация.
INVARIANTS: Final[tuple[str, ...]] = (
    "hallucination_rate",
    "chunk_page_linkage_errors",
)
LOWER_IS_BETTER: Final[frozenset[str]] = frozenset(
    {
        "cer",
        "wer",
        "false_ocr_rate",
        "missed_ocr_rate",
        "illegible_false_positive_rate",
        "hallucination_rate",
        "chunk_page_linkage_errors",
    }
)


@dataclass(frozen=True, slots=True)
class Report:
    """Отчёт одного прогона."""

    corpus_hash: str
    corpus_version: str
    chunking_version: str
    font_version: str
    environment: Mapping[str, str]
    aggregate: Mapping[str, float]
    by_category: Mapping[str, Mapping[str, float]]
    pages: tuple[PageScore, ...]


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа стенда оценки."""
    args = _parse(argv)
    return asyncio.run(_run(args))


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="eval.run_eval")
    parser.add_argument("--corpus", type=Path, default=Path("eval/corpus"))
    parser.add_argument("--out", type=Path, default=Path("eval/reports/latest"))
    parser.add_argument("--baseline", type=Path, default=Path("eval/baseline.json"))
    parser.add_argument("--seed", type=int, default=corpus.DEFAULT_SEED)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--write-baseline", action="store_true")
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    specs = _selected(args.categories)
    manifest = _corpus(args, specs)
    report = await _measure(args, manifest)
    _write(args.out, report)

    if args.write_baseline:
        _save_baseline(args.baseline, report)
        return EXIT_OK
    if not args.baseline.is_file():
        return EXIT_OK
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if args.categories is None and baseline["corpus_hash"] != report.corpus_hash:
        return EXIT_CORPUS_CHANGED
    return _compare(report, baseline)


def _selected(categories: Sequence[str] | None) -> tuple[corpus.DocumentSpec, ...]:
    if not categories:
        return corpus.CORPUS
    wanted = {corpus.Category(name) for name in categories}
    return tuple(spec for spec in corpus.CORPUS if spec.category in wanted)


def _corpus(
    args: argparse.Namespace,
    specs: Sequence[corpus.DocumentSpec],
) -> corpus.Manifest:
    font_dir = fonts.font_dir_from_env()
    fonts.download_missing(font_dir)
    manifest_path = args.corpus / corpus.MANIFEST_NAME
    if args.regenerate or not manifest_path.is_file():
        return corpus.generate(
            args.corpus, font_dir=font_dir, specs=specs, seed=args.seed
        )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return corpus.Manifest(
        corpus_version=raw["corpus_version"],
        seed=raw["seed"],
        font_version=raw["font_version"],
        font_checksum=raw["font_checksum"],
        corpus_hash=raw["corpus_hash"],
        documents=tuple(_truth(entry) for entry in raw["documents"]),
    )


def _truth(entry: Mapping[str, Any]) -> corpus.DocumentTruth:
    return corpus.DocumentTruth(
        doc_id=entry["doc_id"],
        category=entry["category"],
        page_count=entry["page_count"],
        sha256=entry["sha256"],
        pages=tuple(
            corpus.PageTruth(
                number=page["number"],
                expected_extraction_method=page["expected_extraction_method"],
                expected_page_status=page["expected_page_status"],
                section_boundaries=tuple(page["section_boundaries"]),
                unreadable_text=page["unreadable_text"],
            )
            for page in entry["pages"]
        ),
    )


async def _measure(args: argparse.Namespace, manifest: corpus.Manifest) -> Report:
    settings = AppSettings()
    scores: list[PageScore] = []
    chunks: list[ChunkOutcome] = []
    async with build_processing(
        settings, storage=LocalCorpusStorage(root=args.corpus)
    ) as processing:
        for truth in manifest.documents:
            outcome = await process_document(processing, doc_id=truth.doc_id)
            scores.extend(score_document(truth, outcome, corpus_root=args.corpus))
            chunks.extend(outcome.chunks)
    return Report(
        corpus_hash=manifest.corpus_hash,
        corpus_version=manifest.corpus_version,
        chunking_version=str(CHUNKING_VERSION),
        font_version=manifest.font_version,
        environment=_environment(),
        aggregate=aggregate(scores, chunks),
        by_category=by_category(scores),
        pages=tuple(scores),
    )


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _write(out: Path, report: Report) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "report.md").write_text(_markdown(report), encoding="utf-8")


def _markdown(report: Report) -> str:
    lines = [
        "# Оценка качества обработки",
        "",
        (
            f"Корпус `{report.corpus_hash[:12]}`, версия {report.corpus_version},"
            f" чанкование {report.chunking_version}, шрифт {report.font_version}."
        ),
        "",
        "## Сводно",
        "",
        "| Метрика | Значение |",
        "|---|---|",
    ]
    lines += [
        f"| `{name}` | {value:.4f} |"
        for name, value in sorted(report.aggregate.items())
    ]
    lines += [
        "",
        "## По категориям",
        "",
        "| Категория | CER | WER | boundary F1 | Стр. |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| `{category}` | {values['cer']:.4f} | {values['wer']:.4f}"
        f" | {values['boundary_f1']:.4f} | {values['pages']:.0f} |"
        for category, values in sorted(report.by_category.items())
    ]
    return "\n".join(lines) + "\n"


def _save_baseline(path: Path, report: Report) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "corpus_hash": report.corpus_hash,
                "corpus_version": report.corpus_version,
                "chunking_version": report.chunking_version,
                "font_version": report.font_version,
                "aggregate": report.aggregate,
                "by_category": report.by_category,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _compare(report: Report, baseline: Mapping[str, Any]) -> int:
    for name in INVARIANTS:
        if report.aggregate.get(name, 0.0) > 0.0:
            return EXIT_THRESHOLD
    for name, before in baseline["aggregate"].items():
        now = report.aggregate.get(name)
        if now is None or name in INVARIANTS:
            continue
        if _worse(name, before=float(before), now=now):
            return EXIT_REGRESSION
    return EXIT_OK


def _worse(name: str, *, before: float, now: float) -> bool:
    if name in LOWER_IS_BETTER:
        return now > before + TOLERANCE_WORSE
    return now < before - TOLERANCE_BETTER


if __name__ == "__main__":
    sys.exit(main())
