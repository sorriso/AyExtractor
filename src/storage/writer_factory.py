# src/storage/writer_factory.py — v2
"""Factory: instantiate output writer from configuration.

See spec project layout for writer backends (local, s3).
"""

from __future__ import annotations

from ayextractor.config.settings import Settings
from ayextractor.storage.base_output_writer import BaseOutputWriter
from ayextractor.storage.local_writer import LocalWriter


def create_writer(settings: Settings) -> BaseOutputWriter:
    """Create the appropriate output writer based on settings.

    Args:
        settings: Application settings (OUTPUT_WRITER env var).

    Returns:
        BaseOutputWriter instance.

    Raises:
        ValueError: If writer type is not supported.
    """
    if settings.output_writer == "local":
        return LocalWriter()

    if settings.output_writer == "s3":
        from ayextractor.storage.s3_writer import S3Writer
        if not settings.output_s3_bucket:
            raise ValueError(
                "OUTPUT_S3_BUCKET must be set when OUTPUT_WRITER=s3"
            )
        return S3Writer(
            bucket=settings.output_s3_bucket,
            prefix=settings.output_s3_prefix,
            region=settings.output_s3_region or None,
        )

    raise ValueError(f"Unsupported output writer: {settings.output_writer!r}")
