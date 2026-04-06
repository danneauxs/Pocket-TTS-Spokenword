from pathlib import Path
from qtpy.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                           QLabel, QPushButton, QWidget, QMessageBox)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont, QPalette, QColor

from pocket_tts.utils.download_manager import DownloadManager
from pocket_tts.utils.path_manager import PathManager

class SetupWindow(QMainWindow):
    """Setup window for first-run downloads and initialization."""

    def __init__(self):
        super().__init__()
        self.download_manager = DownloadManager()
        self.completed_downloads = set()
        self.failed_components = set()
        self.active_threads = []
        self.cancelled = False

        # Ensure directories exist
        PathManager.ensure_directories()

        self.init_ui()
        self.start_setup()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Pocket TTS - Setup")
        self.setMinimumSize(650, 550)
        self.resize(750, 650)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title_label = QLabel("Setting up Pocket TTS")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Authentication message
        auth_title = QLabel("🔐 Download Required")
        auth_title_font = QFont()
        auth_title_font.setPointSize(16)
        auth_title_font.setBold(True)
        auth_title.setFont(auth_title_font)
        auth_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(auth_title)

        auth_message = QLabel(
            "To use Pocket TTS, you need to download the required model files.\n\n"
            "1. Visit: https://huggingface.co/kyutai/pocket-tts\n"
            "2. Accept the model terms of service\n"
            "3. Click the download buttons below\n"
            "4. The app will automatically detect and install the files"
        )
        auth_message.setWordWrap(True)
        auth_message.setAlignment(Qt.AlignCenter)
        auth_font = QFont()
        auth_font.setPointSize(11)
        auth_message.setFont(auth_font)
        layout.addWidget(auth_message)

        # Download buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)

        self.tts_button = QPushButton("📥 Download TTS Model")
        self.tts_button.clicked.connect(self.download_tts_model)
        self.tts_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: 2px solid #2196F3;
                padding: 12px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 11px;
                min-width: 160px;
            }
            QPushButton:hover {
                background-color: #1976D2;
                border-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
                border-color: #0D47A1;
            }
        """)
        button_layout.addWidget(self.tts_button)

        self.tokenizer_button = QPushButton("📥 Download Tokenizer")
        self.tokenizer_button.clicked.connect(self.download_tokenizer)
        self.tokenizer_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: 2px solid #2196F3;
                padding: 12px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 11px;
                min-width: 160px;
            }
            QPushButton:hover {
                background-color: #1976D2;
                border-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
                border-color: #0D47A1;
            }
        """)
        button_layout.addWidget(self.tokenizer_button)

        layout.addLayout(button_layout)

        # Status section
        status_layout = QVBoxLayout()
        status_layout.setSpacing(5)

        self.status_label = QLabel("Ready to download. Click the buttons above to start.")
        status_font = QFont()
        status_font.setPointSize(10)
        self.status_label.setFont(status_font)
        self.status_label.setStyleSheet("color: #666666;")
        status_layout.addWidget(self.status_label)

        # Individual file status
        self.tts_status = QLabel("TTS Model: Not downloaded")
        self.tts_status.setFont(status_font)
        status_layout.addWidget(self.tts_status)

        self.tokenizer_status = QLabel("Tokenizer: Not downloaded")
        self.tokenizer_status.setFont(status_font)
        status_layout.addWidget(self.tokenizer_status)

        layout.addLayout(status_layout)

        # Control buttons
        button_layout = QHBoxLayout()




        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_setup)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: 2px solid #f44336;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
                border-color: #d32f2f;
            }
        """)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        # Authentication error area (hidden initially)
        self.auth_error_widget = QWidget()
        self.auth_error_widget.hide()

        auth_layout = QVBoxLayout(self.auth_error_widget)
        auth_layout.setSpacing(10)

        auth_title = QLabel("🔐 Authentication Required")
        auth_title_font = QFont()
        auth_title_font.setPointSize(14)
        auth_title_font.setBold(True)
        auth_title.setFont(auth_title_font)
        auth_layout.addWidget(auth_title)

        auth_message = QLabel(
            "To download the TTS models, you need to accept the terms on HuggingFace.\n\n"
            "1. Go to: https://huggingface.co/kyutai/pocket-tts\n"
            "2. Create a HuggingFace account (if you don't have one)\n"
            "3. Accept the model terms of service\n"
            "4. Return here and click 'Continue Download'"
        )
        auth_message.setWordWrap(True)
        auth_message_font = QFont()
        auth_message_font.setPointSize(11)
        auth_message.setFont(auth_message_font)
        auth_message.setMinimumHeight(120)  # Ensure minimum height for visibility
        auth_layout.addWidget(auth_message)

        # Add stretch to push content up
        auth_layout.addStretch()

        layout.addWidget(self.auth_error_widget)

        # Add layout stretch for proper space allocation
        layout.addStretch()

        # Set dark theme for better visibility
        self.apply_dark_theme()

    def download_tts_model(self):
        """Handle TTS model download button click."""
        import webbrowser
        url = "https://huggingface.co/kyutai/pocket-tts/resolve/main/tts_b6369a24.safetensors"
        webbrowser.open(url)
        self.log(f"Opened TTS model download: {url}")
        self.tts_status.setText("TTS Model: Downloading...")
        self.tts_status.setStyleSheet("color: #FF9800; font-weight: bold;")
        self.status_label.setText("Downloading TTS Model... Monitor your browser.")

    def download_tokenizer(self):
        """Handle tokenizer download button click."""
        import webbrowser
        url = "https://huggingface.co/kyutai/pocket-tts/resolve/main/tokenizer.model"
        webbrowser.open(url)
        self.log(f"Opened tokenizer download: {url}")
        self.tokenizer_status.setText("Tokenizer: Downloading...")
        self.tokenizer_status.setStyleSheet("color: #FF9800; font-weight: bold;")
        self.status_label.setText("Downloading Tokenizer... Monitor your browser.")

    def start_download_monitoring(self):
        """Start monitoring the Downloads folder for completed files."""
        from qtpy.QtCore import QTimer

        self.downloads_dir = Path.home() / "Downloads"
        self.monitoring_files = {
            "tts_b6369a24.safetensors": "tts_model",
            "tokenizer.model": "tokenizer"
        }

        # Check for existing files first
        self.check_existing_downloads()

        # Set up periodic monitoring
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.check_downloads)
        self.monitor_timer.start(2000)  # Check every 2 seconds

        self.status_label.setText("Monitoring Downloads folder for completed files...")

    def check_existing_downloads(self):
        """Check if any required files already exist in Downloads."""
        for filename, cache_name in self.monitoring_files.items():
            file_path = self.downloads_dir / filename
            if file_path.exists() and self.is_download_complete(str(file_path)):
                self.handle_completed_download(filename, cache_name)

    def check_downloads(self):
        """Check for newly completed downloads."""
        for filename, cache_name in self.monitoring_files.items():
            file_path = self.downloads_dir / filename
            if file_path.exists() and self.is_download_complete(str(file_path)):
                self.handle_completed_download(filename, cache_name)

    def is_download_complete(self, file_path: str) -> bool:
        """Check if a download is complete using file locking detection."""
        try:
            with open(file_path, 'rb+') as f:
                f.seek(0, 2)  # Seek to end
            return True
        except (IOError, OSError, PermissionError):
            return False

    def handle_completed_download(self, filename: str, cache_name: str):
        """Handle a completed download."""
        if cache_name in self.completed_downloads:
            return  # Already processed

        file_path = self.downloads_dir / filename
        cache_path = PathManager.get_cache_dir() / cache_name

        try:
            # Move file to cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.rename(cache_path)
            self.completed_downloads.add(cache_name)

            # Update UI
            if cache_name == "tts_model":
                self.tts_status.setText("TTS Model: ✅ Installed")
                self.tts_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            elif cache_name == "tokenizer":
                self.tokenizer_status.setText("Tokenizer: ✅ Installed")
                self.tokenizer_status.setStyleSheet("color: #4CAF50; font-weight: bold;")

            self.log(f"Successfully installed {filename} to cache")
            self.check_setup_complete()

        except Exception as e:
            self.log(f"Failed to install {filename}: {e}")
            if cache_name == "tts_model":
                self.tts_status.setText("TTS Model: ❌ Install failed")
                self.tts_status.setStyleSheet("color: #f44336; font-weight: bold;")
            elif cache_name == "tokenizer":
                self.tokenizer_status.setText("Tokenizer: ❌ Install failed")
                self.tokenizer_status.setStyleSheet("color: #f44336; font-weight: bold;")

    def check_setup_complete(self):
        """Check if all downloads are complete."""
        required_components = {"tts_model", "tokenizer"}
        if required_components.issubset(self.completed_downloads):
            self.status_label.setText("All downloads complete! Launching Pocket TTS...")
            self.log("Setup complete - launching main application")

            # Stop monitoring
            if hasattr(self, 'monitor_timer'):
                self.monitor_timer.stop()

            # Launch main app after a short delay
            from qtpy.QtCore import QTimer
            QTimer.singleShot(2000, self.launch_main_app)

    def log(self, message: str):
        """Add a message to the log."""
        print(message)  # Print to console

    def apply_dark_theme(self):
        """Apply a dark theme to the setup window."""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)

    def start_setup(self):
        """Begin the setup process."""
        self.log("Starting Pocket TTS setup...")
        self.log(f"Cache directory: {PathManager.get_cache_dir()}")
        self.log(f"User data directory: {PathManager.get_user_data_dir()}")

        # Check if all components are already downloaded
        all_downloaded = all(
            self.download_manager.is_component_downloaded(comp)
            for comp in self.download_manager.required_components.keys()
        )

        if all_downloaded:
            self.log("All components already downloaded. Launching application...")
            self.launch_main_app()
            return

        # Start monitoring for downloads
        self.start_download_monitoring()





    def cancel_setup(self):
        """Cancel the setup process."""
        self.cancelled = True
        self.log("Setup cancelled by user.")

        # Cancel active downloads
        for thread in self.active_threads:
            if hasattr(thread, 'worker') and hasattr(thread.worker, 'cancel'):
                thread.worker.cancel()

        # Close application
        QApplication.quit()

    def launch_main_app(self):
        """Launch the main application."""
        try:
            self.log("Launching main application...")
            # Import and launch main GUI
            from pocket_tts.gui.main_window import main
            self.close()
            main()
        except Exception as e:
            QMessageBox.critical(
                self, "Launch Error",
                f"Failed to launch main application: {e}"
            )
            QApplication.quit()

    def log(self, message: str):
        """Add a message to the log."""
        print(message)  # Print to console

    def closeEvent(self, event):
        """Handle window close event."""
        if not self.cancelled:
            # If user closes window during setup, cancel
            self.cancel_setup()
        event.accept()