"""Тесты объектов хранилища: ссылка, контрольная сумма, MIME-тип, размер."""

from __future__ import annotations

import hashlib

import pytest

from document_worker.domain.constants import MAX_FILE_SIZE_BYTES
from document_worker.domain.errors import (
    ChecksumMismatch,
    DocumentTooLarge,
    EmptyDocument,
    InvalidChecksum,
    InvalidFileSize,
    InvalidMimeType,
    InvalidObjectRef,
    UnsupportedDocumentFormat,
)
from document_worker.domain.value_objects.storage import (
    Checksum,
    ChecksumAlgorithm,
    FileSize,
    MimeType,
    ObjectRef,
    SourceFile,
)

pytestmark = pytest.mark.unit

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _source_file(*, size: int = 1024, mime: str = "application/pdf") -> SourceFile:
    return SourceFile(
        ref=ObjectRef(bucket="documents", key="a/source.pdf"),
        mime_type=MimeType(mime),
        size=FileSize(size),
    )


@pytest.mark.parametrize(
    "key",
    [
        "../etc/passwd",
        "documents/../../secret",
        "documents/..",
        "/leading/slash",
        "back\\slash",
        "with\x00null",
        "with\nnewline",
        "",
        "x" * 1025,
    ],
)
def test_object_ref_rejects_unsafe_key(key: str) -> None:
    with pytest.raises(InvalidObjectRef):
        ObjectRef(bucket="documents", key=key)


def test_object_ref_accepts_dotted_key_without_traversal() -> None:
    ref = ObjectRef(bucket="documents", key="a/file..name.pdf")

    assert ref.key == "a/file..name.pdf"


@pytest.mark.parametrize(
    "bucket",
    [
        "ab",
        "x" * 64,
        "UPPER",
        "-leading-dash",
        "trailing-dash-",
        "double..dot",
        "192.168.0.1",
        "",
    ],
)
def test_object_ref_rejects_invalid_bucket(bucket: str) -> None:
    with pytest.raises(InvalidObjectRef):
        ObjectRef(bucket=bucket, key="source.pdf")


def test_object_ref_parse_uri_roundtrip() -> None:
    uri = "s3://documents/2026/07/source.pdf"

    assert ObjectRef.parse_uri(uri).to_uri() == uri


@pytest.mark.parametrize(
    "uri", ["documents/source.pdf", "http://host/key", "s3://only-bucket"]
)
def test_object_ref_parse_uri_rejects_foreign_scheme(uri: str) -> None:
    with pytest.raises(InvalidObjectRef):
        ObjectRef.parse_uri(uri)


def test_checksum_normalizes_hex_to_lowercase() -> None:
    checksum = Checksum(ChecksumAlgorithm.SHA256, "A" * 64)

    assert checksum.value == "a" * 64


def test_checksum_sha256_of_empty_bytes_matches_known_vector() -> None:
    assert Checksum.sha256_of(b"").value == EMPTY_SHA256


def test_checksum_sha256_of_matches_hashlib() -> None:
    payload = b"%PDF-1.7 fake"

    assert Checksum.sha256_of(payload).value == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("value", ["", "abc", "z" * 64, "a" * 63, "a" * 65])
def test_checksum_rejects_value_not_matching_algorithm(value: str) -> None:
    with pytest.raises(InvalidChecksum):
        Checksum(ChecksumAlgorithm.SHA256, value)


def test_checksum_matches_is_case_insensitive() -> None:
    assert Checksum(ChecksumAlgorithm.SHA256, "A" * 64).matches(
        Checksum(ChecksumAlgorithm.SHA256, "a" * 64)
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("application/pdf", "application/pdf"),
        ("APPLICATION/PDF", "application/pdf"),
        ("application/pdf; charset=binary", "application/pdf"),
        ("  application/pdf  ", "application/pdf"),
    ],
)
def test_mime_type_strips_parameters_and_lowercases(raw: str, expected: str) -> None:
    assert MimeType(raw).value == expected


@pytest.mark.parametrize(
    "raw", ["", "application", "/pdf", "application/", "app lication/pdf"]
)
def test_mime_type_rejects_malformed_value(raw: str) -> None:
    with pytest.raises(InvalidMimeType):
        MimeType(raw)


def test_mime_type_ensure_supported_raises_for_zip() -> None:
    with pytest.raises(UnsupportedDocumentFormat):
        MimeType("application/zip").ensure_supported()


def test_mime_type_pdf_is_supported() -> None:
    mime = MimeType("application/pdf")

    assert mime.is_pdf
    assert mime.is_supported()
    mime.ensure_supported()


@pytest.mark.parametrize("value", [-1, -1024])
def test_file_size_rejects_negative(value: int) -> None:
    with pytest.raises(InvalidFileSize):
        FileSize(value)


def test_file_size_ensure_within_raises_document_too_large() -> None:
    with pytest.raises(DocumentTooLarge):
        FileSize(MAX_FILE_SIZE_BYTES + 1).ensure_within(MAX_FILE_SIZE_BYTES)


def test_file_size_ensure_within_accepts_value_on_the_limit() -> None:
    FileSize(MAX_FILE_SIZE_BYTES).ensure_within(MAX_FILE_SIZE_BYTES)


def test_source_file_with_zero_size_raises_empty_document() -> None:
    with pytest.raises(EmptyDocument):
        _source_file(size=0).ensure_acceptable()


def test_source_file_with_unsupported_mime_raises() -> None:
    with pytest.raises(UnsupportedDocumentFormat):
        _source_file(mime="application/zip").ensure_acceptable()


def test_source_file_above_limit_raises_document_too_large() -> None:
    with pytest.raises(DocumentTooLarge):
        _source_file(size=MAX_FILE_SIZE_BYTES + 1).ensure_acceptable()


def test_source_file_accepts_valid_pdf() -> None:
    _source_file().ensure_acceptable()


def test_source_file_verify_raises_on_mismatch() -> None:
    source = SourceFile(
        ref=ObjectRef(bucket="documents", key="source.pdf"),
        mime_type=MimeType("application/pdf"),
        size=FileSize(10),
        checksum=Checksum(ChecksumAlgorithm.SHA256, "a" * 64),
    )

    with pytest.raises(ChecksumMismatch):
        source.verify(Checksum(ChecksumAlgorithm.SHA256, "b" * 64))


def test_source_file_without_declared_checksum_skips_verification() -> None:
    _source_file().verify(Checksum.sha256_of(b"anything"))
