# tests/unit/storage/test_s3_writer.py — v1
"""Tests for storage/s3_writer.py — mocked S3 client."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest

from ayextractor.storage.s3_writer import S3Writer


@pytest.fixture
def mock_s3_writer():
    """Create S3Writer with mocked boto3 client."""
    storage: dict[str, bytes] = {}

    mock_client = MagicMock()

    def put_object(Bucket, Key, Body, **kwargs):
        storage[Key] = Body if isinstance(Body, bytes) else Body.encode("utf-8")

    def get_object(Bucket, Key):
        if Key not in storage:
            raise Exception(f"NoSuchKey: {Key}")
        return {"Body": io.BytesIO(storage[Key])}

    def head_object(Bucket, Key):
        if Key not in storage:
            error = MagicMock()
            raise mock_client.exceptions.ClientError(error, "HeadObject")

    def copy_object(Bucket, CopySource, Key):
        src_key = CopySource["Key"]
        storage[Key] = storage[src_key]

    def list_objects_v2(Bucket, Prefix, Delimiter="/"):
        contents = []
        prefixes = set()
        for key in storage:
            if key.startswith(Prefix):
                rest = key[len(Prefix):]
                if Delimiter in rest:
                    dir_name = Prefix + rest.split(Delimiter)[0] + Delimiter
                    prefixes.add(dir_name)
                else:
                    contents.append({"Key": key})
        return {
            "Contents": contents,
            "CommonPrefixes": [{"Prefix": p} for p in sorted(prefixes)],
        }

    mock_client.put_object = put_object
    mock_client.get_object = get_object
    mock_client.head_object = head_object
    mock_client.copy_object = copy_object
    mock_client.list_objects_v2 = list_objects_v2
    mock_client.exceptions = MagicMock()
    mock_client.exceptions.ClientError = type("ClientError", (Exception,), {})

    with patch("ayextractor.storage.s3_writer.S3Writer.__init__", return_value=None):
        writer = S3Writer.__new__(S3Writer)
        writer._s3 = mock_client
        writer._bucket = "test-bucket"
        writer._prefix = "ayextractor/"

    return writer


class TestS3Writer:
    @pytest.mark.asyncio
    async def test_write_and_read(self, mock_s3_writer):
        await mock_s3_writer.write("test/file.txt", b"hello s3")
        data = await mock_s3_writer.read("test/file.txt")
        assert data == b"hello s3"

    @pytest.mark.asyncio
    async def test_write_string(self, mock_s3_writer):
        await mock_s3_writer.write("text.txt", "content string")
        data = await mock_s3_writer.read("text.txt")
        assert data == b"content string"

    @pytest.mark.asyncio
    async def test_exists_true(self, mock_s3_writer):
        await mock_s3_writer.write("exists.txt", b"data")
        assert await mock_s3_writer.exists("exists.txt") is True

    @pytest.mark.asyncio
    async def test_exists_false(self, mock_s3_writer):
        assert await mock_s3_writer.exists("nonexistent.txt") is False

    @pytest.mark.asyncio
    async def test_copy(self, mock_s3_writer):
        await mock_s3_writer.write("src.txt", b"original")
        await mock_s3_writer.copy("src.txt", "dst.txt")
        data = await mock_s3_writer.read("dst.txt")
        assert data == b"original"

    @pytest.mark.asyncio
    async def test_list_dir(self, mock_s3_writer):
        await mock_s3_writer.write("dir/a.txt", b"a")
        await mock_s3_writer.write("dir/b.txt", b"b")
        items = await mock_s3_writer.list_dir("dir")
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_full_key(self, mock_s3_writer):
        assert mock_s3_writer._full_key("test.txt") == "ayextractor/test.txt"

    def test_import_error_without_boto3(self):
        """Clear ImportError when boto3 is not available."""
        import sys
        boto3_mod = sys.modules.get("boto3")
        sys.modules["boto3"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="boto3"):
                S3Writer(bucket="test")
        finally:
            if boto3_mod is not None:
                sys.modules["boto3"] = boto3_mod
            else:
                sys.modules.pop("boto3", None)
