# src/storage/s3_writer.py — v1
"""S3-compatible output writer (OUTPUT_WRITER=s3).

Supports AWS S3, MinIO, and other S3-compatible storage.
Requires 'boto3' package: pip install boto3.
See spec §5 for output structure.
"""

from __future__ import annotations

import io
import logging

from ayextractor.storage.base_output_writer import BaseOutputWriter

logger = logging.getLogger(__name__)


class S3Writer(BaseOutputWriter):
    """Write outputs to S3-compatible object storage."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "ayextractor/",
        region: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        """Initialize S3 writer.

        Args:
            bucket: S3 bucket name.
            prefix: Key prefix for all objects (e.g. "ayextractor/").
            region: AWS region (optional, uses boto3 default if not set).
            endpoint_url: Custom endpoint for MinIO/compatible storage.
        """
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "boto3 package required for S3 writer: pip install boto3"
            ) from e

        kwargs: dict = {}
        if region:
            kwargs["region_name"] = region
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url

        self._s3 = boto3.client("s3", **kwargs)
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""

    def _full_key(self, path: str) -> str:
        """Build the full S3 key from a relative path."""
        return f"{self._prefix}{path}"

    async def write(self, path: str, content: bytes | str) -> None:
        """Write content to S3."""
        key = self._full_key(path)
        if isinstance(content, str):
            body = content.encode("utf-8")
        else:
            body = content
        self._s3.put_object(Bucket=self._bucket, Key=key, Body=body)
        logger.debug("S3 write: s3://%s/%s (%d bytes)", self._bucket, key, len(body))

    async def read(self, path: str) -> bytes:
        """Read content from S3."""
        key = self._full_key(path)
        response = self._s3.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    async def exists(self, path: str) -> bool:
        """Check if an S3 object exists."""
        key = self._full_key(path)
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except self._s3.exceptions.ClientError:
            return False
        except Exception:
            return False

    async def copy(self, src: str, dst: str) -> None:
        """Copy an S3 object."""
        src_key = self._full_key(src)
        dst_key = self._full_key(dst)
        self._s3.copy_object(
            Bucket=self._bucket,
            CopySource={"Bucket": self._bucket, "Key": src_key},
            Key=dst_key,
        )

    async def create_symlink(self, target: str, link: str) -> None:
        """S3 doesn't support symlinks — create a redirect marker object instead."""
        link_key = self._full_key(link)
        target_key = self._full_key(target)
        # Store a small JSON marker indicating the redirect
        marker = f'{{"redirect_to": "{target_key}"}}'.encode("utf-8")
        self._s3.put_object(
            Bucket=self._bucket,
            Key=link_key,
            Body=marker,
            Metadata={"x-ayextractor-symlink": target_key},
        )
        logger.debug("S3 symlink marker: %s → %s", link_key, target_key)

    async def list_dir(self, path: str) -> list[str]:
        """List objects under a prefix (simulates directory listing)."""
        prefix = self._full_key(path)
        if not prefix.endswith("/"):
            prefix += "/"

        response = self._s3.list_objects_v2(
            Bucket=self._bucket, Prefix=prefix, Delimiter="/"
        )

        items: list[str] = []
        # Files (direct children)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            name = key[len(prefix):]
            if name:  # Skip the prefix itself
                items.append(name)

        # Subdirectories
        for cp in response.get("CommonPrefixes", []):
            dir_name = cp["Prefix"][len(prefix):].rstrip("/")
            if dir_name:
                items.append(dir_name + "/")

        return sorted(items)
