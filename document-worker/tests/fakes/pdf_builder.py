"""Генератор тестовых PDF.

Фикстуры собираются кодом, а не лежат в репозитории: бинарник в git невозможно
прочитать в ревью, а на вопрос «почему тест красный» он не отвечает. Все
генераторы детерминированы — тот же вход даёт те же байты.

Текст в сгенерированных документах латинский: кириллица требует встроенного
шрифта, а его файл на разных машинах разный. Проверяемая здесь механика —
координаты слов, склейка, отсутствие `/ToUnicode` — от письменности не зависит.
"""

from __future__ import annotations

import io
import zlib
from typing import TYPE_CHECKING, Final

import pikepdf
from PIL import Image, ImageDraw, TiffImagePlugin

from tests.fakes.page_images import make_page_image

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

PDF_SIGNATURE: Final[bytes] = b"%PDF-"
PAGE_WIDTH: Final[int] = 612
PAGE_HEIGHT: Final[int] = 792
FONT_SIZE: Final[int] = 12
LINE_HEIGHT: Final[int] = 18
MARGIN: Final[int] = 72

DEFAULT_LINES: Final[tuple[str, ...]] = (
    "SUPPLY CONTRACT No 42",
    "The Supplier undertakes to deliver the goods",
    "within thirty calendar days from the date hereof.",
)


def _standard_font(pdf: pikepdf.Pdf) -> pikepdf.Object:
    return pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Font,
            Subtype=pikepdf.Name.Type1,
            BaseFont=pikepdf.Name.Helvetica,
            Encoding=pikepdf.Name.WinAnsiEncoding,
        )
    )


def _unmappable_font(pdf: pikepdf.Pdf) -> pikepdf.Object:
    """Шрифт, по которому символ в текст не восстановить.

    Не из базовых четырнадцати, встроенной кодировки нет, `/ToUnicode` нет.
    """
    descriptor = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.FontDescriptor,
            FontName=pikepdf.Name("/AAAAAA+Custom"),
            Flags=4,
            ItalicAngle=0,
            Ascent=800,
            Descent=-200,
            CapHeight=700,
            StemV=80,
            FontBBox=[0, -200, 1000, 800],
        )
    )
    return pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Font,
            Subtype=pikepdf.Name.TrueType,
            BaseFont=pikepdf.Name("/AAAAAA+Custom"),
            FirstChar=1,
            LastChar=4,
            Widths=[500, 500, 500, 500],
            FontDescriptor=descriptor,
        )
    )


def _text_page(
    pdf: pikepdf.Pdf,
    blocks: Sequence[tuple[int, int, Sequence[str]]],
    *,
    font: pikepdf.Object,
) -> None:
    commands = ["BT", f"/F1 {FONT_SIZE} Tf"]
    for x, y, lines in blocks:
        commands.append(f"1 0 0 1 {x} {y} Tm")
        for index, line in enumerate(lines):
            if index:
                commands.append(f"0 -{LINE_HEIGHT} Td")
            commands.append(f"({_escape(line)}) Tj")
    commands.append("ET")
    _append_page(pdf, "\n".join(commands).encode("latin-1"), font=font)


def _append_page(
    pdf: pikepdf.Pdf,
    content: bytes,
    *,
    font: pikepdf.Object | None = None,
    xobject: pikepdf.Object | None = None,
) -> None:
    resources = pikepdf.Dictionary()
    if font is not None:
        resources.Font = pikepdf.Dictionary(F1=font)
    if xobject is not None:
        resources.XObject = pikepdf.Dictionary(Im1=xobject)
    pdf.pages.append(
        pikepdf.Page(
            pdf.make_indirect(
                pikepdf.Dictionary(
                    Type=pikepdf.Name.Page,
                    MediaBox=[0, 0, PAGE_WIDTH, PAGE_HEIGHT],
                    Resources=resources,
                    Contents=pdf.make_stream(content),
                )
            )
        )
    )


def _escape(line: str) -> str:
    return line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def make_text_pdf(
    path: Path,
    *,
    pages: int = 1,
    lines: Sequence[str] = DEFAULT_LINES,
) -> Path:
    """Цифровой документ с читаемым текстовым слоем."""
    with pikepdf.new() as pdf:
        font = _standard_font(pdf)
        for _ in range(pages):
            _text_page(
                pdf,
                [(MARGIN, PAGE_HEIGHT - MARGIN, lines)],
                font=font,
            )
        pdf.save(path, deterministic_id=True)
    return path


def make_two_column_pdf(path: Path) -> Path:
    """Две колонки: проверка порядка слов и их координат."""
    with pikepdf.new() as pdf:
        font = _standard_font(pdf)
        _text_page(
            pdf,
            [
                (MARGIN, PAGE_HEIGHT - MARGIN, ("left column first", "left second")),
                (
                    PAGE_WIDTH // 2,
                    PAGE_HEIGHT - MARGIN,
                    ("right column first", "right second"),
                ),
            ],
            font=font,
        )
        pdf.save(path, deterministic_id=True)
    return path


def make_glued_text_pdf(path: Path) -> Path:
    """Текст без пробелов: типичный признак сломанного слоя."""
    return make_text_pdf(path, lines=("SUPPLYCONTRACTNUMBERFORTYTWOSIGNEDTODAY",))


def make_broken_tounicode_pdf(path: Path) -> Path:
    """Шрифт без `/ToUnicode` и вне базовых четырнадцати."""
    with pikepdf.new() as pdf:
        font = _unmappable_font(pdf)
        _text_page(
            pdf, [(MARGIN, PAGE_HEIGHT - MARGIN, ("\x01\x02\x03\x04",))], font=font
        )
        pdf.save(path, deterministic_id=True)
    return path


def make_blank_page_pdf(path: Path) -> Path:
    """Страница без ресурсов: ни шрифтов, ни изображений."""
    with pikepdf.new() as pdf:
        _append_page(pdf, b"")
        pdf.save(path, deterministic_id=True)
    return path


def make_empty_pdf(path: Path) -> Path:
    """Документ без единой страницы."""
    with pikepdf.new() as pdf:
        pdf.save(path, deterministic_id=True)
    return path


def make_corrupted_pdf(path: Path) -> Path:
    """Документ с испорченной таблицей ссылок: чинится пересканированием."""
    make_text_pdf(path)
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"startxref", b"startxrEf", 1))
    return path


def make_truncated_pdf(path: Path) -> Path:
    """Сигнатура на месте, содержимого нет: чинить нечего.

    Именно такой файл проходит проверку по сигнатуре и падает на разборе.
    """
    path.write_bytes(PDF_SIGNATURE + b"1.7" + bytes(512))
    return path


def make_encrypted_pdf(path: Path, *, user_password: str = "secret") -> Path:  # noqa: S107 — пароль тестовой фикстуры
    """Документ, который без пароля не открыть.

    Байтовой воспроизводимости у зашифрованных файлов нет: ключ выводится из
    идентификатора, а его qpdf для шифрования делает случайным.
    """
    with pikepdf.new() as pdf:
        font = _standard_font(pdf)
        _text_page(pdf, [(MARGIN, PAGE_HEIGHT - MARGIN, DEFAULT_LINES)], font=font)
        pdf.save(
            path,
            encryption=pikepdf.Encryption(user=user_password, owner="owner"),
        )
    return path


def make_owner_encrypted_pdf(
    path: Path,
    *,
    owner_password: str = "owner",  # noqa: S107 — пароль тестовой фикстуры
) -> Path:
    """Документ с пустым пользовательским паролем: открывается, но защищён."""
    with pikepdf.new() as pdf:
        font = _standard_font(pdf)
        _text_page(pdf, [(MARGIN, PAGE_HEIGHT - MARGIN, DEFAULT_LINES)], font=font)
        pdf.save(
            path,
            encryption=pikepdf.Encryption(user="", owner=owner_password),
        )
    return path


def make_scan_image(
    *,
    width: int = 850,
    height: int = 1100,
    text: str = "SUPPLY CONTRACT",
) -> Image.Image:
    """Изображение страницы-скана без текстового слоя."""
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, width - 40, height - 40), outline=0, width=2)
    draw.text((80, 80), text, fill=0)
    for line in range(6):
        top = 160 + line * 40
        draw.line((80, top, width - 80, top), fill=0, width=3)
    return image


def make_scan_pdf(path: Path, *, jpeg_quality: int = 80) -> Path:
    """Скан: страница целиком состоит из растрового изображения."""
    image = make_scan_image()
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=jpeg_quality)
    with pikepdf.new() as pdf:
        xobject = pdf.make_stream(buffer.getvalue())
        xobject.Type = pikepdf.Name.XObject
        xobject.Subtype = pikepdf.Name.Image
        xobject.Width = image.width
        xobject.Height = image.height
        xobject.ColorSpace = pikepdf.Name.DeviceRGB
        xobject.BitsPerComponent = 8
        xobject.Filter = pikepdf.Name.DCTDecode
        _append_page(pdf, _draw_full_page_image(), xobject=pdf.make_indirect(xobject))
        pdf.save(path, deterministic_id=True)
    return path


def make_ccitt_g4_scan_pdf(path: Path) -> Path:
    """Скан в CCITT G4 — кодек, который распаковывает не всякая библиотека."""
    image = make_scan_image().convert("1")
    data, columns, rows = _ccitt_g4_bytes(image)
    with pikepdf.new() as pdf:
        xobject = pdf.make_stream(data)
        xobject.Type = pikepdf.Name.XObject
        xobject.Subtype = pikepdf.Name.Image
        xobject.Width = columns
        xobject.Height = rows
        xobject.ColorSpace = pikepdf.Name.DeviceGray
        xobject.BitsPerComponent = 1
        xobject.Filter = pikepdf.Name.CCITTFaxDecode
        xobject.DecodeParms = pikepdf.Dictionary(K=-1, Columns=columns, Rows=rows)
        _append_page(pdf, _draw_full_page_image(), xobject=pdf.make_indirect(xobject))
        pdf.save(path, deterministic_id=True)
    return path


def _draw_full_page_image() -> bytes:
    return f"q {PAGE_WIDTH} 0 0 {PAGE_HEIGHT} 0 0 cm /Im1 Do Q".encode("latin-1")


def _ccitt_g4_bytes(image: Image.Image) -> tuple[bytes, int, int]:
    """Возвращает поток G4 и размеры, вытащенные из TIFF, который пишет Pillow."""
    buffer = io.BytesIO()
    image.save(buffer, format="TIFF", compression="group4")
    buffer.seek(0)
    with TiffImagePlugin.TiffImageFile(buffer) as tiff:
        # tag_v2 появляется при разборе файла, поэтому в стабах его нет.
        tags: dict[int, tuple[int, ...]] = tiff.tag_v2  # type: ignore[attr-defined]
        offsets = tags[TiffImagePlugin.STRIPOFFSETS]
        counts = tags[TiffImagePlugin.STRIPBYTECOUNTS]
        raw = buffer.getvalue()
        data = b"".join(
            raw[offset : offset + count]
            for offset, count in zip(offsets, counts, strict=True)
        )
        return data, tiff.width, tiff.height


def make_non_pdf_file(path: Path) -> Path:
    """Файл с расширением PDF, но не PDF внутри."""
    path.write_bytes(zlib.compress("это не PDF".encode() * 100))
    return path


def make_ocr_scan_pdf(path: Path, *, pages: int = 1) -> Path:
    """Скан с текстом, который читается распознавателем.

    Отличается от `make_scan_pdf` кеглем: мелкий шрифт по умолчанию детектор
    не находит, и проверять на нём распознавание бессмысленно.
    """
    image = make_page_image()
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    with pikepdf.new() as pdf:
        for _ in range(pages):
            xobject = pdf.make_stream(buffer.getvalue())
            xobject.Type = pikepdf.Name.XObject
            xobject.Subtype = pikepdf.Name.Image
            xobject.Width = image.width
            xobject.Height = image.height
            xobject.ColorSpace = pikepdf.Name.DeviceRGB
            xobject.BitsPerComponent = 8
            xobject.Filter = pikepdf.Name.DCTDecode
            _append_page(
                pdf, _draw_full_page_image(), xobject=pdf.make_indirect(xobject)
            )
        pdf.save(path, deterministic_id=True)
    return path


def make_partially_readable_scan_pdf(path: Path) -> Path:
    """Скан, где одна страница читается, а вторая пуста.

    Пустая страница даёт `illegible`, и документ обязан стать
    `partially_processed`, а не притвориться полностью обработанным.
    """
    pages = (make_page_image(), make_page_image(lines=()))
    with pikepdf.new() as pdf:
        for image in pages:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=90)
            xobject = pdf.make_stream(buffer.getvalue())
            xobject.Type = pikepdf.Name.XObject
            xobject.Subtype = pikepdf.Name.Image
            xobject.Width = image.width
            xobject.Height = image.height
            xobject.ColorSpace = pikepdf.Name.DeviceRGB
            xobject.BitsPerComponent = 8
            xobject.Filter = pikepdf.Name.DCTDecode
            _append_page(
                pdf, _draw_full_page_image(), xobject=pdf.make_indirect(xobject)
            )
        pdf.save(path, deterministic_id=True)
    return path
