from pathlib import Path
import platform
import os

class PathManager:
    """Cross-platform path management for cache and user data directories."""

    @staticmethod
    def get_cache_dir() -> Path:
        """Directory for downloaded models and dependencies (runtime downloads)."""
        if platform.system() == "Windows":
            return Path(os.environ["USERPROFILE"]) / ".pocket_tts" / "cache"
        elif platform.system() == "Darwin":  # macOS
            return Path.home() / "Library" / "Caches" / "pocket-tts"
        else:  # Linux
            return Path.home() / ".cache" / "pocket-tts"

    @staticmethod
    def get_user_data_dir() -> Path:
        """Directory for user files: outputs, settings, custom voices."""
        if platform.system() == "Windows":
            return Path(os.environ["APPDATA"]) / "pocket-tts"
        elif platform.system() == "Darwin":  # macOS
            return Path.home() / "Library" / "Application Support" / "pocket-tts"
        else:  # Linux
            data_home = os.environ.get("XDG_DATA_HOME", "~/.local/share")
            return Path(data_home).expanduser() / "pocket-tts"

    @staticmethod
    def ensure_directories():
        """Create all required directories on startup."""
        dirs = [
            PathManager.get_cache_dir(),
            PathManager.get_user_data_dir(),
            PathManager.get_user_data_dir() / "output",
            PathManager.get_user_data_dir() / "voices",
            PathManager.get_user_data_dir() / "logs"
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_output_dir() -> Path:
        """Directory for generated audio outputs."""
        return PathManager.get_user_data_dir() / "output"

    @staticmethod
    def get_voices_dir() -> Path:
        """Directory for user-provided voice samples."""
        return PathManager.get_user_data_dir() / "voices"

    @staticmethod
    def get_logs_dir() -> Path:
        """Directory for application logs."""
        return PathManager.get_user_data_dir() / "logs"

    @staticmethod
    def get_settings_file() -> Path:
        """Path to user settings file."""
        return PathManager.get_user_data_dir() / "settings.json"