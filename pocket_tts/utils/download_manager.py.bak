import asyncio
import hashlib
import logging
import platform
from pathlib import Path
from typing import Dict, Optional, Callable, Any

import aiohttp
from qtpy.QtCore import QThread, Signal, QObject

from pocket_tts.utils.path_manager import PathManager

logger = logging.getLogger(__name__)

class DownloadError(Exception):
    """Raised when a download fails."""
    pass

class DownloadWorker(QObject):
    """Qt worker for download operations with progress signals."""

    progress = Signal(str, int)  # component_name, percentage
    finished = Signal(str)       # component_name
    error = Signal(str, str)     # component_name, error_message

    def __init__(self, component_name: str, download_info: Dict[str, Any]):
        super().__init__()
        self.component_name = component_name
        self.download_info = download_info
        self.cancelled = False

    def cancel(self):
        """Cancel the download."""
        self.cancelled = True

    def run(self):
        """Execute the download."""
        try:
            asyncio.run(self._download_async())
        except Exception as e:
            self.error.emit(self.component_name, str(e))

    async def _download_async(self):
        """Async download implementation."""
        url = self.download_info["url"]
        dest_path = PathManager.get_cache_dir() / self.component_name
        expected_hash = self.download_info.get("hash")

        # Create parent directories
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded and verified
        if dest_path.exists() and expected_hash:
            if self._verify_file(dest_path, expected_hash):
                self.progress.emit(self.component_name, 100)
                self.finished.emit(self.component_name)
                return

        # Download the file
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))

                    with open(dest_path, 'wb') as f:
                        downloaded = 0
                        async for chunk in response.content.iter_chunked(8192):
                            if self.cancelled:
                                return
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percentage = int((downloaded / total_size) * 100)
                                self.progress.emit(self.component_name, percentage)

                    # Verify download
                    if expected_hash and not self._verify_file(dest_path, expected_hash):
                        raise DownloadError(f"Hash verification failed for {self.component_name}")

                    self.finished.emit(self.component_name)

        except Exception as e:
            if dest_path.exists():
                dest_path.unlink()  # Clean up failed download
            raise DownloadError(f"Download failed: {e}")

    def _verify_file(self, file_path: Path, expected_hash: str) -> bool:
        """Verify file integrity using SHA256 hash."""
        if not expected_hash.startswith("sha256:"):
            return True  # Skip verification if no hash provided

        expected = expected_hash[7:]  # Remove "sha256:" prefix
        sha256 = hashlib.sha256()

        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            return sha256.hexdigest() == expected
        except Exception:
            return False

class DownloadManager:
    """Manages runtime downloads of dependencies and models."""

    def __init__(self):
        self.cache_dir = PathManager.get_cache_dir()
        self.required_components = self._get_required_components()

    def _get_required_components(self) -> Dict[str, Dict[str, Any]]:
        """Define all components that need to be downloaded."""
        # Get platform-specific PyTorch URL
        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "linux":
            if "x86_64" in machine:
                torch_url = "https://download.pytorch.org/whl/cpu/torch-2.5.0%2Bcpu-cp310-cp310-linux_x86_64.whl"
            else:
                torch_url = "https://download.pytorch.org/whl/cpu/torch-2.5.0%2Bcpu-cp310-cp310-manylinux2014_aarch64.whl"
        elif system == "darwin":  # macOS
            if "arm64" in machine:
                torch_url = "https://download.pytorch.org/whl/cpu/torch-2.5.0-cp310-none-macosx_11_0_arm64.whl"
            else:
                torch_url = "https://download.pytorch.org/whl/cpu/torch-2.5.0-cp310-none-macosx_10_9_x86_64.whl"
        elif system == "windows":
            torch_url = "https://download.pytorch.org/whl/cpu/torch-2.5.0%2Bcpu-cp310-cp310-win_amd64.whl"
        else:
            raise DownloadError(f"Unsupported platform: {system} {machine}")

        return {
            "torch": {
                "url": torch_url,
                "size": "~200MB",
                "description": "PyTorch ML framework"
            },
            "tts_model": {
                "url": "https://huggingface.co/kyutai/pocket-tts/resolve/main/tts_b6369a24.safetensors",
                "size": "~50MB",
                "description": "TTS model weights"
            },
            "tokenizer": {
                "url": "https://huggingface.co/kyutai/pocket-tts/resolve/main/tokenizer.model",
                "size": "~5MB",
                "description": "Text tokenizer model"
            }
        }

    def is_component_downloaded(self, component_name: str) -> bool:
        """Check if a component is already downloaded and valid."""
        component_info = self.required_components.get(component_name)
        if not component_info:
            return False

        dest_path = self.cache_dir / component_name
        expected_hash = component_info.get("hash")

        if not dest_path.exists():
            return False

        if expected_hash:
            return self._verify_file(dest_path, expected_hash)

        return True  # No hash to verify, assume valid

    def get_download_size_info(self) -> Dict[str, str]:
        """Get size information for all components."""
        return {name: info["size"] for name, info in self.required_components.items()}

    def start_download(self, component_name: str, progress_callback: Callable = None,
                      error_callback: Callable = None) -> Optional[QThread]:
        """Start downloading a component using Qt thread."""
        if component_name not in self.required_components:
            if error_callback:
                error_callback(component_name, f"Unknown component: {component_name}")
            return None

        if self.is_component_downloaded(component_name):
            if progress_callback:
                progress_callback(component_name, 100)
            return None

        # Create download worker
        worker = DownloadWorker(component_name, self.required_components[component_name])

        # Connect signals
        if progress_callback:
            worker.progress.connect(progress_callback)
        if error_callback:
            worker.error.connect(error_callback)

        # Start in thread
        thread = QThread()
        worker.moveToThread(thread)

        # Connect thread lifecycle
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        thread.start()
        return thread

    def _verify_file(self, file_path: Path, expected_hash: str) -> bool:
        """Verify file integrity."""
        if not expected_hash or not expected_hash.startswith("sha256:"):
            return True

        expected = expected_hash[7:]
        sha256 = hashlib.sha256()

        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            return sha256.hexdigest() == expected
        except Exception:
            return False