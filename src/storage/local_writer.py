# src/storage/local_writer.py — v2
"""Local filesystem output writer (default backend).

See spec §5 for output structure.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ayextractor.storage.base_output_writer import BaseOutputWriter


class LocalWriter(BaseOutputWriter):
    """Write outputs to the local filesystem."""

    def __init__(self, base_path: str | None = None) -> None:
        """Initialize with optional base path.

        Args:
            base_path: Root directory for all writes. If None, paths are absolute.
        """
        self._base = Path(base_path) if base_path else None

    def _resolve(self, path: str) -> Path:
        """Resolve a path relative to base_path."""
        if self._base is not None:
            return self._base / path
        return Path(path)

    async def write(self, path: str, content: bytes | str) -> None:
        """Write content to a local file path."""
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")

    async def read(self, path: str) -> bytes:
        """Read content from a local file path."""
        return self._resolve(path).read_bytes()

    async def exists(self, path: str) -> bool:
        """Check if a local path exists."""
        return self._resolve(path).exists()

    async def copy(self, src: str, dst: str) -> None:
        """Copy file or directory."""
        src_path = self._resolve(src)
        dst_path = self._resolve(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.is_dir():
            shutil.copytree(str(src_path), str(dst_path), dirs_exist_ok=True)
        else:
            shutil.copy2(str(src_path), str(dst_path))

    async def create_symlink(self, target: str, link: str) -> None:
        """Create a symbolic link."""
        link_path = self._resolve(link)
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(target, target_is_directory=Path(target).is_dir())

    async def list_dir(self, path: str) -> list[str]:
        """List directory contents."""
        p = self._resolve(path)
        if not p.is_dir():
            return []
        return [entry.name for entry in sorted(p.iterdir())]
