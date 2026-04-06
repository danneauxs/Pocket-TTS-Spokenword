"""
M4B Audio Converter Module
Converts WAV files to M4B audiobook format using FFmpeg.
"""

import subprocess
import re
import time
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

class WavToM4bConverter:
    """Convert WAV files to M4B audiobook format with configurable options."""

    DEFAULT_CONFIG = {
        "speed": 1.0,
        "sample_rate": 24000,
        "enable_normalization": True,
        "normalization_type": "peak",  # "none", "peak", "loudness", "simple"
        "target_db": -1.5,
        "ffmpeg_path": "ffmpeg"  # Allow overriding ffmpeg path
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize converter with optional configuration overrides.

        Args:
            config: Optional dict with configuration overrides
        """
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # Validate configuration
        if self.config["normalization_type"] not in ["none", "peak", "loudness", "simple"]:
            raise ValueError(f"Invalid normalization_type: {self.config['normalization_type']}")

        if not self.check_ffmpeg():
            logger.warning("FFmpeg not found in PATH. M4B conversion will fail.")

    def convert_to_m4b(self, wav_path: Union[str, Path], output_path: Union[str, Path], 
                      speed: Optional[float] = None,
                      sample_rate: Optional[int] = None, 
                      normalization_type: Optional[str] = None) -> bool:
        """
        Convert WAV to M4B with configurable options.

        Args:
            wav_path: Path to input WAV file
            output_path: Path for output M4B file
            speed: Playback speed multiplier (override config)
            sample_rate: Output sample rate (override config)
            normalization_type: Normalization strategy (override config)

        Returns:
            bool: True if successful, False otherwise
        """
        wav_path = Path(wav_path)
        output_path = Path(output_path)

        if not wav_path.exists():
            raise FileNotFoundError(f"Input WAV file not found: {wav_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine settings (method args override config)
        speed_to_use = speed if speed is not None else self.config["speed"]
        sample_rate_to_use = sample_rate if sample_rate is not None else self.config["sample_rate"]
        norm_type = normalization_type if normalization_type is not None else self.config["normalization_type"]
        enable_norm = self.config["enable_normalization"]

        try:
            if not enable_norm or norm_type == "none":
                return self._convert_basic(wav_path, output_path, speed_to_use, sample_rate_to_use)

            elif norm_type == "loudness":
                return self._convert_with_loudness_normalization(
                    wav_path, output_path, speed_to_use, sample_rate_to_use
                )

            elif norm_type == "peak":
                return self._convert_with_peak_normalization(
                    wav_path, output_path, self.config["target_db"], speed_to_use, sample_rate_to_use
                )

            elif norm_type == "simple":
                return self._convert_with_simple_normalization(
                    wav_path, output_path, self.config["target_db"], speed_to_use, sample_rate_to_use
                )
            
            return False
            
        except Exception as e:
            logger.error(f"M4B conversion failed: {e}")
            return False

    def _convert_basic(self, wav_path: Path, output_path: Path, speed: float, sample_rate: int) -> bool:
        """Basic conversion without normalization."""
        logger.info(f"Converting to M4B with speed {speed}x (Basic)...")

        audio_filter = []
        if speed != 1.0:
            audio_filter = ["-filter:a", f"atempo={speed}"]

        cmd = (
            [self._get_ffmpeg_cmd(), "-y", "-i", str(wav_path)]
            + audio_filter
            + ["-ar", str(sample_rate), "-c:a", "aac", "-b:a", "128k", str(output_path)]
        )

        return self._run_ffmpeg_with_progress(cmd)

    def _convert_with_peak_normalization(self, wav_path: Path, output_path: Path, target_db: float, speed: float, sample_rate: int) -> bool:
        """Convert with peak normalization."""
        logger.info(f"Converting to M4B with peak normalization ({target_db}dB) and speed {speed}x...")

        audio_filters = [f"loudnorm=I=-16:TP={target_db}:LRA=11"]
        if speed != 1.0:
            audio_filters.append(f"atempo={speed}")

        cmd = [
            self._get_ffmpeg_cmd(),
            "-y",
            "-i",
            str(wav_path),
            "-af",
            ",".join(audio_filters),
            "-ar",
            str(sample_rate),
            "-c:a",
            "aac",
            "-b:a", 
            "128k",
            str(output_path),
        ]

        return self._run_ffmpeg_with_progress(cmd)

    def _convert_with_loudness_normalization(self, wav_path: Path, output_path: Path, speed: float, sample_rate: int) -> bool:
        """Convert with two-pass loudness normalization (EBU R128)."""
        logger.info("Converting to M4B with loudness normalization (EBU R128)...")

        # Step 1: Analyze audio loudness
        logger.info("Analyzing audio loudness...")
        analyze_cmd = [
            self._get_ffmpeg_cmd(),
            "-y",
            "-i",
            str(wav_path),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ]

        result = subprocess.run(analyze_cmd, capture_output=True, text=True)

        # Extract loudness measurements from stderr
        loudness_data = None
        for line in result.stderr.split("\n"):
            if line.strip().startswith("{"):
                try:
                    loudness_data = json.loads(line.strip())
                    break
                except json.JSONDecodeError:
                    continue

        if not loudness_data:
            logger.warning("Could not analyze loudness, falling back to peak normalization...")
            return self._convert_with_peak_normalization(
                wav_path, output_path, self.config["target_db"], speed, sample_rate
            )

        # Step 2: Apply normalization with measured values
        logger.info("Applying normalization...")

        audio_filters = [
            f"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I={loudness_data['input_i']}:measured_LRA={loudness_data['input_lra']}:measured_TP={loudness_data['input_tp']}:measured_thresh={loudness_data['input_thresh']}:offset={loudness_data['target_offset']}:linear=true:print_format=summary"
        ]
        if speed != 1.0:
            audio_filters.append(f"atempo={speed}")

        cmd = [
            self._get_ffmpeg_cmd(),
            "-y",
            "-i",
            str(wav_path),
            "-af",
            ",".join(audio_filters),
            "-ar",
            str(sample_rate),
            "-c:a",
            "aac",
            "-b:a", 
            "128k",
            str(output_path),
        ]

        return self._run_ffmpeg_with_progress(cmd)

    def _convert_with_simple_normalization(self, wav_path: Path, output_path: Path, target_db: float, speed: float, sample_rate: int) -> bool:
        """Convert with simple volume adjustment."""
        logger.info("Converting to M4B with simple normalization...")

        audio_filters = [f"volume={target_db}dB"]
        if speed != 1.0:
            audio_filters.append(f"atempo={speed}")

        cmd = [
            self._get_ffmpeg_cmd(),
            "-y",
            "-i",
            str(wav_path),
            "-af",
            ",".join(audio_filters),
            "-ar",
            str(sample_rate),
            "-c:a",
            "aac",
            "-b:a", 
            "128k",
            str(output_path),
        ]

        return self._run_ffmpeg_with_progress(cmd)

    def add_metadata(self, m4b_path: Union[str, Path], cover_path: Optional[Union[str, Path]] = None, 
                     metadata_dict: Optional[Dict[str, str]] = None) -> bool:
        """
        Add metadata and cover art to M4B file.
        Note: This creates a temp file and replaces the original.
        """
        m4b_path = Path(m4b_path)
        if not m4b_path.exists():
            return False
            
        temp_path = m4b_path.with_suffix('.temp.m4b')
        
        cmd = [self._get_ffmpeg_cmd(), "-y", "-i", str(m4b_path)]

        if cover_path and Path(cover_path).exists():
            cmd.extend([
                "-i", str(cover_path),
                "-map", "0", "-map", "1",
                "-c", "copy",
                "-disposition:v:0", "attached_pic"
            ])
        else:
            cmd.extend(["-map", "0", "-c", "copy"])

        # Add metadata from dictionary
        if metadata_dict:
            for key, val in metadata_dict.items():
                if val:  # Only add if value exists
                    cmd.extend(["-metadata", f"{key}={val}"])

        cmd.append(str(temp_path))
        
        try:
            self._run_ffmpeg(cmd)
            # Replace original with temp
            shutil.move(str(temp_path), str(m4b_path))
            logger.info("Metadata added successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to add metadata: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False

    def _run_ffmpeg_with_progress(self, cmd: list) -> bool:
        """Execute FFmpeg command with progress monitoring."""
        start_time = time.time()
        
        try:
            process = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)

            audio_secs = 0.0
            for line in process.stderr:
                # Log only critical errors or periodic status to avoid log spam
                match = re.search(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})", line)
                if match:
                    h, m, s, ms = map(int, match.groups())
                    audio_secs = h * 3600 + m * 60 + s + ms / 100
                    elapsed = time.time() - start_time
                    factor = audio_secs / elapsed if elapsed > 0 else 0.0
                    # Could update a progress callback here if needed
                    # print(f"Tape: {match.group(0)} | {factor:.2f}x realtime", end="\r")

            process.wait()
            
            if process.returncode != 0:
                logger.error(f"FFmpeg process failed with return code {process.returncode}")
                return False
                
            logger.info("Conversion complete.")
            return True
            
        except Exception as e:
            logger.error(f"Error running FFmpeg: {e}")
            return False

    def _run_ffmpeg(self, cmd: list):
        """Run FFmpeg command with error handling."""
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg command failed: {' '.join(cmd)}")
            logger.error(f"Exit code: {e.returncode}")
            logger.error(f"stderr: {e.stderr}")
            raise RuntimeError(f"FFmpeg failed with exit code {e.returncode}")

    def _get_ffmpeg_cmd(self) -> str:
        return self.config.get("ffmpeg_path", "ffmpeg")

    @staticmethod
    def check_ffmpeg() -> bool:
        """Check if FFmpeg is available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
