# tests/unit/storage/test_writer_factory.py — v2
"""Tests for storage/writer_factory.py."""

from __future__ import annotations

import pytest

from ayextractor.config.settings import Settings
from ayextractor.storage.local_writer import LocalWriter
from ayextractor.storage.writer_factory import create_writer


class TestCreateWriter:
    def test_default_local(self):
        s = Settings(_env_file=None)
        writer = create_writer(s)
        assert isinstance(writer, LocalWriter)

    def test_s3_missing_bucket(self):
        s = Settings(_env_file=None, output_writer="s3", output_s3_bucket="")
        with pytest.raises(ValueError, match="OUTPUT_S3_BUCKET"):
            create_writer(s)
