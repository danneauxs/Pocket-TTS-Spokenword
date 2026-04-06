"""
Regenerate Tab for Audiobook Generator GUI.
Allows users to regenerate individual audio chunks.
"""

import os
import json
import logging
from pathlib import Path
import subprocess
import platform
from typing import Dict, Any, List

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QListWidget, QListWidgetItem, QTextEdit, QComboBox,
    QGroupBox, QFileDialog, QMessageBox, QProgressDialog, QDialog,
    QFormLayout, QSplitter, QCheckBox, QApplication
)
from qtpy.QtCore import Qt

from pocket_tts.preprocessing.emotion_analyzer import EmotionAnalyzer
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.data.audio import audio_read

logger = logging.getLogger(__name__)


class RegenerateTab(QWidget):
    """Tab for regenerating individual audio chunks."""

    def __init__(self):
        super().__init__()
        self.tts_folder = None
        self.chunks_data = {}  # Dict[int, chunk_metadata]
        self.current_chunk = None
        self.current_chunk_idx = None
        self.temp_audio_path = None
        self.tts_model = None
        self.emotion_analyzer = None
        self.fail_info = {}  # Dict[str, fail_data] for chunks from fail.log

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout = QVBoxLayout(self)

        # TTS Folder section
        self.create_folder_section(layout)

        # Search section
        self.create_search_section(layout)

        # Results and editor splitter
        self.create_main_splitter(layout)

        # Concatenation section
        self.create_concat_section(layout)

        # Initialize disabled state
        self.enable_search_controls(False)

    def create_folder_section(self, parent_layout):
        """Create TTS folder selection section."""
        group = QGroupBox("TTS Folder Selection")
        layout = QHBoxLayout(group)

        # Label
        layout.addWidget(QLabel("TTS Folder:"))

        # Path display
        self.folder_path_label = QLabel("No folder selected")
        self.folder_path_label.setStyleSheet("border: 1px solid #ccc; padding: 5px;")
        self.folder_path_label.setMinimumWidth(400)
        layout.addWidget(self.folder_path_label)

        # Browse button
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_tts_folder)
        layout.addWidget(browse_btn)

        parent_layout.addWidget(group)

    def create_search_section(self, parent_layout):
        """Create search section."""
        group = QGroupBox("Search Chunks")
        layout = QHBoxLayout(group)

        # Keyword search
        layout.addWidget(QLabel("Keyword:"))
        self.keyword_input = QLineEdit()
        self.keyword_input.returnPressed.connect(self.search_by_keyword)
        layout.addWidget(self.keyword_input)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.search_by_keyword)
        layout.addWidget(search_btn)

        # Separator
        layout.addSpacing(20)

        # Chunk ID search
        layout.addWidget(QLabel("Chunk ID:"))
        self.chunk_id_input = QLineEdit()
        self.chunk_id_input.setMaximumWidth(80)
        self.chunk_id_input.returnPressed.connect(self.search_by_id)
        layout.addWidget(self.chunk_id_input)

        go_btn = QPushButton("Go")
        go_btn.clicked.connect(self.search_by_id)
        layout.addWidget(go_btn)

        # Separator
        layout.addSpacing(20)

        # Fail report button
        load_fail_btn = QPushButton("Load Fail Report")
        load_fail_btn.clicked.connect(self.load_fail_report)
        layout.addWidget(load_fail_btn)

        # Stretch to push everything left
        layout.addStretch()

        parent_layout.addWidget(group)

    def create_main_splitter(self, parent_layout):
        """Create the main splitter with results and editor."""
        splitter = QSplitter(Qt.Horizontal)

        # Left side: Results list
        self.create_results_section(splitter)

        # Right side: Chunk editor
        self.create_editor_section(splitter)

        parent_layout.addWidget(splitter)

    def create_results_section(self, splitter):
        """Create results list section."""
        group = QGroupBox("Search Results")
        layout = QVBoxLayout(group)

        self.results_list_label = QLabel("Results: None")
        layout.addWidget(self.results_list_label)

        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.on_chunk_selected)
        layout.addWidget(self.results_list)

        splitter.addWidget(group)

    def create_editor_section(self, splitter):
        """Create chunk editor section."""
        group = QGroupBox("Chunk Editor")
        layout = QVBoxLayout(group)

        # Chunk ID label
        self.chunk_id_label = QLabel("Chunk: None selected")
        layout.addWidget(self.chunk_id_label)

        # Text editor
        text_label = QLabel("Text:")
        layout.addWidget(text_label)

        self.text_editor = QTextEdit()
        self.text_editor.setMaximumHeight(100)
        layout.addWidget(self.text_editor)

        # Emotion row
        emotion_layout = QHBoxLayout()
        emotion_layout.addWidget(QLabel("Emotion:"))
        self.emotion_combo = QComboBox()
        self.emotion_combo.addItems(["Neutral", "Joy", "Anger", "Sadness",
                                    "Fear", "Surprise", "Disgust"])
        emotion_layout.addWidget(self.emotion_combo)

        emotion_layout.addWidget(QLabel("Confidence:"))
        self.confidence_label = QLabel("0.00")
        emotion_layout.addWidget(self.confidence_label)

        view_details_btn = QPushButton("View Details")
        view_details_btn.clicked.connect(self.show_emotion_dialog)
        emotion_layout.addWidget(view_details_btn)

        emotion_layout.addStretch()
        layout.addLayout(emotion_layout)

        # Voice selection
        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("Voice:"))
        self.voice_combo = QComboBox()
        self.voice_combo.addItem("-- Select Voice --", None)
        voice_layout.addWidget(self.voice_combo)
        voice_layout.addStretch()
        layout.addLayout(voice_layout)

        # Button row
        button_layout = QHBoxLayout()

        self.play_orig_btn = QPushButton("Play Original")
        self.play_orig_btn.clicked.connect(self.play_original_audio)
        self.play_orig_btn.setEnabled(False)
        button_layout.addWidget(self.play_orig_btn)

        self.regenerate_btn = QPushButton("Regenerate")
        self.regenerate_btn.clicked.connect(self.regenerate_chunk)
        self.regenerate_btn.setEnabled(False)
        button_layout.addWidget(self.regenerate_btn)

        self.play_new_btn = QPushButton("Play New")
        self.play_new_btn.clicked.connect(self.play_regenerated_audio)
        self.play_new_btn.setEnabled(False)
        button_layout.addWidget(self.play_new_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_regenerated_chunk)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)

        layout.addLayout(button_layout)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
        layout.addWidget(self.status_label)

        splitter.addWidget(group)

    def create_concat_section(self, parent_layout):
        """Create concatenation section."""
        group = QGroupBox("Concatenation")
        layout = QHBoxLayout(group)

        concat_btn = QPushButton("Save Concatenated Audiobook")
        concat_btn.clicked.connect(self.concatenate_all_chunks)
        layout.addWidget(concat_btn)

        layout.addWidget(QLabel("M4B:"))
        self.m4b_checkbox = QCheckBox()
        layout.addWidget(self.m4b_checkbox)

        layout.addWidget(QLabel("Normalization:"))
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["peak", "loudness", "simple", "none"])
        self.norm_combo.setCurrentText("peak")
        layout.addWidget(self.norm_combo)

        layout.addStretch()

        self.output_label = QLabel("Output: None")
        layout.addWidget(self.output_label)

        parent_layout.addWidget(group)

    # Implementation methods

    def enable_search_controls(self, enabled: bool):
        """Enable/disable search controls based on folder selection."""
        self.keyword_input.setEnabled(enabled)
        self.chunk_id_input.setEnabled(enabled)

    def browse_tts_folder(self):
        """Open folder dialog and validate TTS folder structure."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select TTS Folder", str(Path.home())
        )

        if not folder_path:
            return

        folder = Path(folder_path)

        # Validate folder structure
        audio_chunks_dir = folder / "audio_chunks"
        chunks_json = folder / "text_chunks" / "audiobook.chunks.json"

        if not audio_chunks_dir.exists():
            QMessageBox.warning(
                self, "Invalid Folder",
                "Selected folder does not contain 'audio_chunks' subdirectory.\n"
                "Please select a valid TTS output folder."
            )
            return

        if not chunks_json.exists():
            QMessageBox.warning(
                self, "Invalid Folder",
                "Selected folder does not contain 'text_chunks/audiobook.chunks.json'.\n"
                "Please select a valid TTS output folder."
            )
            return

        # Load data
        try:
            self.load_chunks_json(chunks_json)
            self.populate_voice_dropdown(folder)
            self.tts_folder = folder

            # Update UI
            self.folder_path_label.setText(str(folder))
            self.enable_search_controls(True)
            self.status_label.setText("✓ TTS folder loaded successfully")

            # Clear previous results
            self.results_list.clear()
            self.results_list_label.setText("Results: None")
            self.clear_chunk_editor()

        except Exception as e:
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to load TTS folder data: {e}"
            )
            logger.error(f"Failed to load TTS folder {folder}: {e}")

    def load_chunks_json(self, json_path: Path):
        """Load and parse audiobook.chunks.json."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Store chunks indexed by ID
        self.chunks_data = {}
        chunks = data.get('chunks', [])
        for chunk in chunks:
            idx = chunk.get('index', 0)
            self.chunks_data[idx] = chunk

        logger.info(f"Loaded {len(self.chunks_data)} chunks from {json_path}")

    def populate_voice_dropdown(self, folder: Path):
        """Find WAV files in TTS folder and populate voice dropdown."""
        self.voice_combo.clear()
        self.voice_combo.addItem("-- Select Voice --", None)

        # Scan for .wav files (exclude audio_chunks/ subdirectory)
        voice_files = []
        for wav_file in folder.glob("*.wav"):
            if not str(wav_file).startswith(str(folder / "audio_chunks")):
                voice_files.append(wav_file)

        # Add built-in voices
        builtin_voices = ["alba", "marius", "javert", "jean",
                          "fantine", "cosette", "eponine", "azelma"]

        if len(voice_files) == 1:
            # Auto-select single voice
            voice_file = voice_files[0]
            self.voice_combo.addItem(voice_file.stem, str(voice_file))
            self.voice_combo.setCurrentIndex(1)  # Index 1 (after -- Select Voice --)
        else:
            # Multiple voices - leave blank, add all options
            for vf in voice_files:
                self.voice_combo.addItem(vf.stem, str(vf))
            for bv in builtin_voices:
                self.voice_combo.addItem(f"{bv} (built-in)", bv)

    def clear_chunk_editor(self):
        """Clear the chunk editor when no chunk is selected."""
        self.chunk_id_label.setText("Chunk: None selected")
        self.text_editor.clear()
        self.emotion_combo.setCurrentText("Neutral")
        self.confidence_label.setText("0.00")
        self.voice_combo.setCurrentIndex(0)  # -- Select Voice --

        self.play_orig_btn.setEnabled(False)
        self.regenerate_btn.setEnabled(False)
        self.play_new_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.current_chunk = None
        self.current_chunk_idx = None
        self.temp_audio_path = None

    # Search methods

    def search_by_keyword(self):
        """Search chunks by text keyword (case-insensitive)."""
        keyword = self.keyword_input.text().strip()
        if not keyword:
            QMessageBox.information(self, "Empty Search",
                "Please enter a keyword to search for.")
            return

        if not self.chunks_data:
            QMessageBox.information(self, "No Data",
                "Please select a TTS folder first.")
            return

        # Filter chunks containing keyword
        results = []
        for idx, chunk in self.chunks_data.items():
            text = chunk.get('text', '').lower()
            if keyword.lower() in text:
                results.append((idx, chunk))

        self.display_search_results(results, f"Keyword: '{keyword}'")

    def search_by_id(self):
        """Search chunk by exact ID."""
        chunk_id_str = self.chunk_id_input.text().strip()
        if not chunk_id_str:
            return

        if not self.chunks_data:
            QMessageBox.information(self, "No Data",
                "Please select a TTS folder first.")
            return

        # Normalize ID (handle "5" or "00005")
        try:
            chunk_id = int(chunk_id_str)
        except ValueError:
            QMessageBox.warning(self, "Invalid ID",
                "Chunk ID must be a number.")
            return

        if chunk_id in self.chunks_data:
            chunk = self.chunks_data[chunk_id]
            self.display_search_results([(chunk_id, chunk)], f"ID: {chunk_id}")
        else:
            QMessageBox.information(self, "Not Found",
                f"Chunk {chunk_id} not found in dataset.")

    def load_fail_report(self):
        """Load failed chunks from fail.log."""
        if not self.tts_folder:
            QMessageBox.information(self, "No Folder",
                "Please select a TTS folder first.")
            return

        fail_log = self.tts_folder / "fail.log"
        if not fail_log.exists():
            QMessageBox.information(self, "No Fail Report",
                "fail.log not found in the selected TTS folder.")
            return

        # Parse fail.log
        failed_chunks = self.parse_fail_log(fail_log)

        if not failed_chunks:
            QMessageBox.information(self, "No Failures",
                "No failed chunks found in fail.log.")
            return

        # Match with chunks_data
        results = []
        for chunk_id_str in failed_chunks.keys():
            # Extract ID from "chunk_00203" → 203
            try:
                idx = int(chunk_id_str.split('_')[1])
                if idx in self.chunks_data:
                    results.append((idx, self.chunks_data[idx]))
            except (IndexError, ValueError):
                continue

        self.display_search_results(results, f"Failed Chunks ({len(results)})")

        # Store fail info for showing details later
        self.fail_info = failed_chunks

    def parse_fail_log(self, fail_log_path: Path) -> Dict[str, Dict[str, Any]]:
        """Parse fail.log format and extract chunk IDs + error info."""
        failed_chunks = {}

        try:
            with open(fail_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            current_chunk = None
            current_info = {}

            for line in lines:
                line = line.strip()

                if line.startswith("Chunk: "):
                    # Save previous chunk if exists
                    if current_chunk and current_info:
                        failed_chunks[current_chunk] = current_info

                    # Start new chunk
                    current_chunk = line.split(": ")[1]
                    current_info = {'chunk_id': current_chunk}

                elif line.startswith("Status: "):
                    current_info['status'] = line.split(": ", 1)[1]

                elif line.startswith("Original Text: "):
                    current_info['original_text'] = line.split(": ", 1)[1]

                elif line.startswith("Transcribed Text: "):
                    current_info['transcribed_text'] = line.split(": ", 1)[1]

                elif line.startswith("Hallucination: "):
                    current_info['hallucination'] = line.split(": ", 1)[1]

                elif line.startswith("Truncation: "):
                    current_info['truncation'] = line.split(": ", 1)[1]

                elif line.startswith("Explanation: "):
                    current_info['explanation'] = line.split(": ", 1)[1]

            # Save last chunk
            if current_chunk and current_info:
                failed_chunks[current_chunk] = current_info

        except Exception as e:
            logger.error(f"Failed to parse fail.log: {e}")
            QMessageBox.warning(self, "Parse Error",
                f"Failed to parse fail.log: {e}")

        return failed_chunks

    def display_search_results(self, results: List[tuple], title: str):
        """Populate results list widget."""
        self.results_list.clear()
        self.results_list_label.setText(f"Results: {title}")

        if not results:
            self.results_list.addItem("No results found")
            return

        for idx, chunk in sorted(results, key=lambda x: x[0]):
            text_preview = chunk.get('text', '')[:50].replace('\n', ' ')
            if len(chunk.get('text', '')) > 50:
                text_preview += "..."

            item_text = f"chunk_{idx:05d}: {text_preview}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, idx)  # Store chunk index
            self.results_list.addItem(item)

    # Chunk selection and editing

    def on_chunk_selected(self, item: QListWidgetItem):
        """Handle chunk selection from results list."""
        chunk_idx = item.data(Qt.UserRole)
        if chunk_idx is None:
            return

        chunk = self.chunks_data.get(chunk_idx)
        if not chunk:
            return

        # Store current chunk
        self.current_chunk = chunk
        self.current_chunk_idx = chunk_idx

        # Update UI
        self.chunk_id_label.setText(f"Chunk: chunk_{chunk_idx:05d}")
        self.text_editor.setPlainText(chunk.get('text', ''))

        # Set emotion
        emotion_str = chunk.get('emotion', 'neutral')
        if isinstance(emotion_str, str):
            emotion_display = emotion_str.capitalize()
        else:
            emotion_display = str(emotion_str).split('.')[-1].capitalize()

        emotion_index = self.emotion_combo.findText(emotion_display)
        if emotion_index >= 0:
            self.emotion_combo.setCurrentIndex(emotion_index)

        # Set confidence
        confidence = chunk.get('emotion_confidence', 0.0)
        self.confidence_label.setText(f"{confidence:.2f}")

        # Check if audio file exists
        audio_path = self.tts_folder / "audio_chunks" / f"chunk_{chunk_idx:05d}.wav"
        self.play_orig_btn.setEnabled(audio_path.exists())

        # Enable regenerate
        self.regenerate_btn.setEnabled(True)

        # Disable new/save (no regen yet)
        self.play_new_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.temp_audio_path = None

        # If from fail report, show error details
        chunk_id_str = f"chunk_{chunk_idx:05d}"
        if hasattr(self, 'fail_info') and chunk_id_str in self.fail_info:
            self.show_fail_info_dialog(chunk_idx)

    def show_fail_info_dialog(self, chunk_idx: int):
        """Show ASR failure details in popup."""
        chunk_id_str = f"chunk_{chunk_idx:05d}"
        fail_data = self.fail_info.get(chunk_id_str, {})

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Failure Info: {chunk_id_str}")
        dialog.setMinimumWidth(600)

        layout = QVBoxLayout(dialog)

        # Status
        status_label = QLabel(f"<b>Status:</b> {fail_data.get('status', 'N/A')}")
        layout.addWidget(status_label)

        # Original Text
        layout.addWidget(QLabel("<b>Original Text:</b>"))
        orig_text = QTextEdit()
        orig_text.setPlainText(fail_data.get('original_text', ''))
        orig_text.setReadOnly(True)
        orig_text.setMaximumHeight(80)
        layout.addWidget(orig_text)

        # Transcribed Text
        layout.addWidget(QLabel("<b>ASR Transcribed Text:</b>"))
        trans_text = QTextEdit()
        trans_text.setPlainText(fail_data.get('transcribed_text', ''))
        trans_text.setReadOnly(True)
        trans_text.setMaximumHeight(80)
        layout.addWidget(trans_text)

        # Hallucination/Truncation
        if 'hallucination' in fail_data:
            hall_label = QLabel(f"<b>Hallucination:</b> {fail_data['hallucination']}")
            hall_label.setWordWrap(True)
            layout.addWidget(hall_label)

        if 'truncation' in fail_data:
            trunc_label = QLabel(f"<b>Truncation:</b> {fail_data['truncation']}")
            trunc_label.setWordWrap(True)
            layout.addWidget(trunc_label)

        # Explanation
        expl_label = QLabel(f"<b>Explanation:</b> {fail_data.get('explanation', 'N/A')}")
        expl_label.setWordWrap(True)
        layout.addWidget(expl_label)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec_()

    def show_emotion_dialog(self):
        """Show detailed emotion scores and allow override."""
        if not self.current_chunk:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Emotion Details")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        # Current emotion
        emotion_str = self.current_chunk.get('emotion', 'neutral')
        layout.addWidget(QLabel(f"<b>Current Emotion:</b> {emotion_str}"))

        # Emotion scores
        layout.addWidget(QLabel("<b>Emotion Scores:</b>"))
        scores = self.current_chunk.get('emotion_scores', {})

        scores_widget = QWidget()
        scores_layout = QFormLayout(scores_widget)
        for emotion, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            score_label = QLabel(f"{score:.4f}")
            scores_layout.addRow(f"{emotion.capitalize()}:", score_label)
        layout.addWidget(scores_widget)

        # Override option
        layout.addWidget(QLabel("<b>Override Emotion:</b>"))
        override_combo = QComboBox()
        emotions = ['neutral', 'joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust']
        for e in emotions:
            override_combo.addItem(e.capitalize())

        # Set current
        current_index = emotions.index(emotion_str) if emotion_str in emotions else 0
        override_combo.setCurrentIndex(current_index)
        layout.addWidget(override_combo)

        # TTS Parameters (read-only info)
        layout.addWidget(QLabel("<b>TTS Parameters:</b>"))
        tts_params = self.current_chunk.get('tts_params', {})
        params_text = QTextEdit()
        params_text.setPlainText(json.dumps(tts_params, indent=2))
        params_text.setReadOnly(True)
        params_text.setMaximumHeight(100)
        layout.addWidget(params_text)

        # Buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")

        def apply_emotion():
            selected_emotion = override_combo.currentText().lower()
            # Update in-memory chunk data (not saved to disk)
            self.current_chunk['emotion'] = selected_emotion
            self.emotion_combo.setCurrentText(override_combo.currentText())
            dialog.accept()

        ok_btn.clicked.connect(apply_emotion)
        cancel_btn.clicked.connect(dialog.reject)

        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        dialog.exec_()

    # Audio playback

    def play_original_audio(self):
        """Play the original chunk WAV file."""
        if self.current_chunk_idx is None:
            return

        audio_path = self.tts_folder / "audio_chunks" / f"chunk_{self.current_chunk_idx:05d}.wav"
        self.play_audio(str(audio_path))

    def play_regenerated_audio(self):
        """Play the regenerated temporary WAV file."""
        if self.temp_audio_path and self.temp_audio_path.exists():
            self.play_audio(str(self.temp_audio_path))

    def play_audio(self, file_path: str):
        """Play audio using system default player."""
        if not Path(file_path).exists():
            QMessageBox.warning(self, "File Not Found",
                f"Audio file not found: {Path(file_path).name}")
            return

        try:
            system = platform.system()
            if system == "Linux":
                subprocess.Popen(["aplay", file_path],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            elif system == "Darwin":  # macOS
                subprocess.Popen(["afplay", file_path])
            elif system == "Windows":
                os.startfile(file_path)

            self.status_label.setText(f"Playing: {Path(file_path).name}")

        except Exception as e:
            QMessageBox.warning(self, "Playback Error",
                f"Failed to play audio: {e}")

    # Regeneration functionality

    def regenerate_chunk(self):
        """Regenerate audio for selected chunk with edited text/emotion."""
        if not self.current_chunk:
            QMessageBox.warning(self, "No Chunk Selected",
                "Please select a chunk first.")
            return

        # Get edited text
        edited_text = self.text_editor.toPlainText().strip()
        if not edited_text:
            QMessageBox.warning(self, "Empty Text",
                "Cannot regenerate with empty text.")
            return

        # Get selected voice
        voice_data = self.voice_combo.currentData()
        if voice_data is None:
            QMessageBox.warning(self, "No Voice Selected",
                "Please select a voice before regenerating.")
            return

        # Get emotion (might be overridden)
        selected_emotion = self.emotion_combo.currentText().lower()

        # Show progress
        self.status_label.setText("Regenerating audio...")
        self.regenerate_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            # Re-analyze emotion if text was edited
            if edited_text != self.current_chunk.get('text', ''):
                self.status_label.setText("Analyzing emotion...")
                QApplication.processEvents()

                emotion_result = self._analyze_emotion(edited_text)

                # Update chunk data (in-memory only)
                self.current_chunk['text'] = edited_text
                self.current_chunk['emotion'] = emotion_result['emotion']
                self.current_chunk['emotion_scores'] = emotion_result['scores']
                self.current_chunk['emotion_confidence'] = emotion_result['confidence']

                # Update UI
                confidence = emotion_result['confidence']
                self.confidence_label.setText(f"{confidence:.2f}")

            # Load TTS model
            self.status_label.setText("Loading TTS model...")
            QApplication.processEvents()
            model = self._ensure_tts_model_loaded()

            # Get voice state
            self.status_label.setText("Loading voice...")
            QApplication.processEvents()
            voice_state = model.get_state_for_audio_prompt(voice_data, truncate=True)

            # Generate audio
            self.status_label.setText("Generating audio...")
            QApplication.processEvents()

            # Extract frames_after_eos from chunk TTS params (controls pause duration)
            tts_params = self.current_chunk.get('tts_params', {})
            frames_after_eos = tts_params.get('frames_after_eos', 2)
            
            logger.debug(f"Regenerating with frames_after_eos={frames_after_eos}")

            audio = model.generate_audio(
                voice_state,          # First positional: model_state
                edited_text,          # Second positional: text_to_generate
                frames_after_eos=frames_after_eos  # Only valid keyword argument
            )

            # Save to temp file
            chunk_idx = self.current_chunk_idx
            temp_path = self.tts_folder / "audio_chunks" / f"chunk_{chunk_idx:05d}.temp.wav"
            self._save_audio(audio, temp_path)

            self.temp_audio_path = temp_path

            # Update UI
            self.status_label.setText("✓ Regeneration complete!")
            self.play_new_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.regenerate_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Regeneration Failed",
                f"Failed to regenerate audio: {e}")
            self.status_label.setText("✗ Regeneration failed")
            self.regenerate_btn.setEnabled(True)

    def _ensure_tts_model_loaded(self):
        """Lazy-load TTS model."""
        if self.tts_model is None:
            try:
                self.tts_model = TTSModel.load_model()
                logger.info("TTS model loaded successfully for regeneration")
            except Exception as e:
                error_msg = f"Failed to load TTS model: {e}"
                logger.error(error_msg)
                raise RuntimeError(f"{error_msg}\n\nThis may be due to missing model files or insufficient memory.\nPlease check the logs for more details.")
        return self.tts_model

    def _analyze_emotion(self, text: str):
        """Analyze emotion for edited text."""
        if self.emotion_analyzer is None:
            self.emotion_analyzer = EmotionAnalyzer()

        return self.emotion_analyzer.analyze(text)

    def _save_audio(self, audio_tensor, output_path: Path):
        """Save audio tensor to WAV file."""
        import scipy.io.wavfile
        import torch

        sample_rate = 24000

        # Convert to int16 PCM
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.cpu().numpy()
        else:
            audio_np = audio_tensor

        audio_int16 = (audio_np.clip(-1, 1) * 32767).astype('int16')

        scipy.io.wavfile.write(str(output_path), sample_rate, audio_int16)

    # Save functionality

    def save_regenerated_chunk(self):
        """Replace original with regenerated audio."""
        if not self.temp_audio_path or not self.temp_audio_path.exists():
            QMessageBox.warning(self, "No Regenerated Audio",
                "Please regenerate audio before saving.")
            return

        chunk_idx = self.current_chunk_idx

        # Paths
        original_path = self.tts_folder / "audio_chunks" / f"chunk_{chunk_idx:05d}.wav"
        backup_dir = self.tts_folder / "Regenerated"
        backup_path = backup_dir / f"chunk_{chunk_idx:05d}.wav"

        try:
            # Create backup directory
            backup_dir.mkdir(exist_ok=True)

            # Move original to backup (overwrite if exists)
            if original_path.exists():
                import shutil
                shutil.move(str(original_path), str(backup_path))
                logger.info(f"Backed up original to: {backup_path}")

            # Rename temp to original
            self.temp_audio_path.rename(original_path)
            logger.info(f"Saved regenerated audio as: {original_path}")

            # Update UI
            self.status_label.setText("✓ Chunk saved! Original backed up.")
            self.save_btn.setEnabled(False)  # Disable until next regen
            self.temp_audio_path = None

            # Update play original button (now plays new version)
            self.play_orig_btn.setEnabled(True)

            QMessageBox.information(self, "Saved",
                f"Chunk saved successfully!\nOriginal backed up to:\n{backup_path.name}")

        except Exception as e:
            QMessageBox.critical(self, "Save Failed",
                f"Failed to save regenerated chunk: {e}")
            logger.error(f"Save failed: {e}")

    # Concatenation functionality

    def concatenate_all_chunks(self):
        """Concatenate all audio chunks into final audiobook."""
        if not self.tts_folder:
            QMessageBox.warning(self, "No Folder Selected",
                "Please select a TTS folder first.")
            return

        audio_chunks_dir = self.tts_folder / "audio_chunks"

        # Find all chunk files
        chunk_files = sorted(audio_chunks_dir.glob("chunk_*.wav"),
                            key=lambda x: int(x.stem.split('_')[1]))

        # Filter out .temp.wav files
        chunk_files = [f for f in chunk_files if not f.name.endswith('.temp.wav')]

        if not chunk_files:
            QMessageBox.warning(self, "No Chunks Found",
                "No audio chunks found to concatenate.")
            return

        # Show progress dialog
        progress = QProgressDialog("Concatenating audio chunks...", "Cancel", 
                                  0, len(chunk_files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        try:
            # Load and concatenate chunks
            audio_chunks = []
            for i, chunk_file in enumerate(chunk_files):
                if progress.wasCanceled():
                    return

                progress.setValue(i)
                progress.setLabelText(f"Loading {chunk_file.name}...")
                QApplication.processEvents()

                chunk_audio, sr = audio_read(str(chunk_file))
                audio_chunks.append(chunk_audio)

            progress.setLabelText("Combining audio...")
            QApplication.processEvents()

            # Concatenate
            import torch
            final_audio = torch.cat(audio_chunks, dim=1)
            final_audio = final_audio.squeeze(0)  # Remove channel dimension

            # Save WAV
            output_wav = self.tts_folder.parent / "audiobook.wav"
            self._save_audio(final_audio.numpy(), output_wav)

            logger.info(f"Saved concatenated WAV: {output_wav}")

            # Calculate duration
            samples = final_audio.shape[0] if len(final_audio.shape) == 1 else final_audio.shape[1]
            duration_sec = samples / 24000
            duration_str = f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}"

            result_msg = f"Audiobook saved:\n{output_wav.name}\nDuration: {duration_str}"

            # M4B conversion if requested
            if self.m4b_checkbox.isChecked():
                progress.setLabelText("Converting to M4B...")
                progress.setValue(len(chunk_files))
                QApplication.processEvents()

                output_m4b = output_wav.with_suffix('.m4b')
                self._convert_to_m4b(output_wav, output_m4b)

                result_msg += f"\n\nM4B saved:\n{output_m4b.name}"

            progress.setValue(len(chunk_files))
            progress.close()

            # Show result
            self.output_label.setText(f"Output: {output_wav.name}")
            self.status_label.setText("✓ Concatenation complete!")

            QMessageBox.information(self, "Success", result_msg)

        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Concatenation Failed",
                f"Failed to concatenate chunks: {e}")
            logger.error(f"Concatenation failed: {e}")

    def _convert_to_m4b(self, wav_path: Path, m4b_path: Path):
        """Convert WAV to M4B using existing converter."""
        from pocket_tts.audio.m4b_converter import WavToM4bConverter

        # Get normalization type
        norm_type = self.norm_combo.currentText()

        # Use config from main tab settings (or defaults)
        m4b_config = {
            'enabled': True,
            'normalization_type': norm_type,
            'speed': 1.0,
            'sample_rate': 24000,
            'target_db': -1.5
        }

        converter = WavToM4bConverter(m4b_config)
        success = converter.convert_to_m4b(wav_path=str(wav_path), output_path=str(m4b_path))
        if not success:
            logger.error(f"M4B conversion failed: {m4b_path}")
            raise RuntimeError(f"M4B conversion failed: {m4b_path}")

        logger.info(f"Converted to M4B: {m4b_path}")