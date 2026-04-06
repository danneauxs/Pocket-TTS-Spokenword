"""
Parameter mapping for emotion-driven TTS parameter calculation.
Maps emotions, punctuation, and boundaries to TTS parameters.
"""

import logging
from typing import Dict, Any, Optional

from .schema import EmotionType, BoundaryType, TTSParams

logger = logging.getLogger(__name__)

class ParameterMapper:
    """
    Maps emotion/punctuation/boundaries to TTS parameters.

    Calculates optimal temperature, pause durations, and other TTS settings
    based on emotional context and text structure.
    """

    # Default emotion mappings (can be overridden by config)
    DEFAULT_EMOTION_MAPPINGS = {
        EmotionType.JOY: {'temperature': 0.90, 'speed_factor': 1.15, 'eos_threshold': -4.0},
        EmotionType.SURPRISE: {'temperature': 0.85, 'speed_factor': 1.10, 'eos_threshold': -3.5},
        EmotionType.ANGER: {'temperature': 0.80, 'speed_factor': 1.10, 'eos_threshold': -3.5},
        EmotionType.NEUTRAL: {'temperature': 0.70, 'speed_factor': 1.0, 'eos_threshold': -4.0},
        EmotionType.SADNESS: {'temperature': 0.65, 'speed_factor': 0.85, 'eos_threshold': -5.0},
        EmotionType.FEAR: {'temperature': 0.60, 'speed_factor': 0.90, 'eos_threshold': -4.5},
        EmotionType.DISGUST: {'temperature': 0.65, 'speed_factor': 0.95, 'eos_threshold': -4.0}
    }

    # Punctuation pause mappings (frames)
    DEFAULT_PUNCTUATION_PAUSES = {
        '.': 2,     # 160ms - normal sentence
        '!': 1,     # 80ms - urgent, quick
        '?': 3,     # 240ms - thoughtful question
        '...': 4,   # 320ms - trailing off
        ',': 1,     # 80ms - brief pause
        '--': 2,    # 160ms - em dash
        ';': 2,     # 160ms - semicolon
        ':': 2,     # 160ms - colon
    }

    # Boundary additional pauses (milliseconds)
    DEFAULT_BOUNDARY_PAUSES = {
        BoundaryType.SENTENCE_END: 200,      # Default short pause
        BoundaryType.PARAGRAPH_BREAK: 600,   # Default medium pause
        BoundaryType.CHAPTER_START: 2000,    # Default long pause
    }

    # Emotion pause multipliers
    DEFAULT_EMOTION_MULTIPLIERS = {
        EmotionType.ANGER: 1.3,     # Angry pauses are 30% longer
        EmotionType.FEAR: 1.2,      # Fearful pauses 20% longer
        EmotionType.SADNESS: 1.4,   # Sad pauses 40% longer
        EmotionType.JOY: 0.9,       # Joyful pauses 10% shorter
        EmotionType.NEUTRAL: 1.0,   # Baseline
        EmotionType.SURPRISE: 1.1,  # Slightly longer for surprise
        EmotionType.DISGUST: 1.1,   # Slightly longer for disgust
    }

    def __init__(self,
                 config=None,
                 emotion_mappings: Optional[Dict[str, Dict[str, float]]] = None,
                 punctuation_pauses: Optional[Dict[str, int]] = None,
                 boundary_pauses: Optional[Dict[BoundaryType, int]] = None,
                 emotion_multipliers: Optional[Dict[EmotionType, float]] = None,
                 speed_range: float = 0.10,
                 base_temperature: float = 0.7,
                 base_eos_threshold: float = -4.0,
                 base_frames_after_eos: int = 2):
        """
        Initialize parameter mapper with config or custom mappings.

        Args:
            config: Config object to load pause values from
            emotion_mappings: Custom emotion → parameter mappings
            punctuation_pauses: Custom punctuation → pause mappings
            boundary_pauses: Custom boundary → pause mappings
            emotion_multipliers: Custom emotion pause multipliers
            speed_range: Maximum speed variation range (± from neutral)
            base_temperature: Base temperature value from GUI/config
            base_eos_threshold: Base EOS threshold value from GUI/config
            base_frames_after_eos: Base frames after EOS value from GUI/config
        """
        # Store config for later use
        self.config = config

        # Store base TTS parameters
        self.base_temperature = base_temperature
        self.base_eos_threshold = base_eos_threshold
        self.base_frames_after_eos = base_frames_after_eos
        
        # Load emotion mappings (these stay as defaults)
        self.emotion_mappings = emotion_mappings or self._convert_emotion_mappings(self.DEFAULT_EMOTION_MAPPINGS)

        # Load punctuation pauses (these stay as defaults)
        self.punctuation_pauses = punctuation_pauses or self.DEFAULT_PUNCTUATION_PAUSES

        # Load boundary pauses: prioritize parameter over config over defaults
        if boundary_pauses:
            self.boundary_pauses = boundary_pauses
            logger.info("Using custom boundary pause values")
        elif config and hasattr(config, 'pauses'):
            # Load raw milliseconds directly (no frame conversion)
            self.boundary_pauses = {
                BoundaryType.SENTENCE_END: config.pauses['base_durations']['sentence_end'],
                BoundaryType.PARAGRAPH_BREAK: config.pauses['base_durations']['paragraph_break'],
                BoundaryType.CHAPTER_START: config.pauses['base_durations']['chapter_start']
            }
            logger.info("Loaded pause values from config")
        else:
            # Use hardcoded defaults as fallback
            self.boundary_pauses = self.DEFAULT_BOUNDARY_PAUSES
            logger.info("Using default pause values")

        # Load emotion multipliers (these stay as defaults)
        self.emotion_multipliers = emotion_multipliers or self.DEFAULT_EMOTION_MULTIPLIERS
        self.speed_range = speed_range
        


        # Load short sentence mitigation settings
        if config and hasattr(config, 'mitigation'):
            self.short_sentence_threshold = config.mitigation.get('short_sentence_threshold', 5)
            self.temperature_short_sentence = config.mitigation.get('temperature_short_sentence', 0.5)
            logger.info(f"Short sentence mitigation enabled: threshold={self.short_sentence_threshold}, temp={self.temperature_short_sentence}")
        else:
            self.short_sentence_threshold = 5
            self.temperature_short_sentence = 0.5
            logger.info("Using default short sentence mitigation settings")

        logger.info("ParameterMapper initialized")

    def _convert_emotion_mappings(self, enum_mappings: Dict[EmotionType, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Convert EmotionType keys to string keys for easier config."""
        return {emotion.value: params for emotion, params in enum_mappings.items()}

    def calculate_params(self,
                         emotion: EmotionType,
                         punctuation: str,
                         boundary_type: BoundaryType,
                         has_emphasis: bool = False,
                         word_count: Optional[int] = None) -> TTSParams:
        """
        Calculate TTS parameters for a text chunk.

        Args:
            emotion: Dominant emotion of the chunk
            punctuation: Ending punctuation character
            boundary_type: Type of text boundary
            has_emphasis: Whether chunk contains emphasis (affects quality)
            word_count: Number of words in the chunk (for short sentence mitigation)

        Returns:
            TTSParams: Complete parameter set for TTS generation
        """
        emotion_str = emotion.value

        # Get base emotion parameters
        emotion_params = self.emotion_mappings.get(emotion_str, self.emotion_mappings['neutral'])

        # Calculate temperature: adjust base temperature by emotion ratio (relative to neutral 0.7)
        emotion_temp_ratio = emotion_params['temperature'] / 0.7  # Scale relative to neutral
        calculated_temp = self.base_temperature * emotion_temp_ratio
        speed_factor = emotion_params['speed_factor']

        # Apply short sentence mitigation if applicable
        is_short_sentence = word_count is not None and word_count <= self.short_sentence_threshold
        if is_short_sentence:
            calculated_temp = self.temperature_short_sentence
            logger.debug(f"Short sentence detected ({word_count} words), overriding temperature to {calculated_temp}")

        # Clamp speed factor to specified range
        if speed_factor > 1.0:
            speed_factor = min(speed_factor, 1.0 + self.speed_range)
        elif speed_factor < 1.0:
            speed_factor = max(speed_factor, 1.0 - self.speed_range)

        # Calculate pause duration
        pause_frames = self._calculate_pause_frames(
            punctuation, boundary_type, emotion
        )

        # Use base EOS threshold (could be adjusted by emotion if desired in future)
        eos_threshold = self.base_eos_threshold

        # Quality boost for emphasis
        lsd_steps = 2 if has_emphasis else 1

        params = TTSParams(
            temperature=calculated_temp,
            frames_after_eos=self.base_frames_after_eos,
            eos_threshold=eos_threshold,
            lsd_decode_steps=lsd_steps
        )

        logger.debug(f"Calculated params for {emotion_str}: temp={calculated_temp:.2f}, "
                    f"pause={pause_frames}, eos={eos_threshold}, quality={lsd_steps}")

        return params

    def _calculate_pause_frames(self,
                               punctuation: str,
                               boundary_type: BoundaryType,
                               emotion: EmotionType) -> int:
        """
        Calculate total pause frames based on punctuation, boundary, and emotion.

        Args:
            punctuation: Ending punctuation character
            boundary_type: Type of boundary
            emotion: Emotion affecting pause length

        Returns:
            Total pause frames
        """
        # DISABLED: Old logic calculated large frame counts based on duration.
        # We now use a minimal fixed value (2 frames ~= 160ms) to allow natural
        # voice decay, while structural silence is appended digitally in post-processing.
        
        # Original logic preserved for reference:
        # punct_pause = self.punctuation_pauses.get(punctuation, 2)
        # boundary_pause = self.boundary_pauses.get(boundary_type, 0)
        # emotion_mult = self.emotion_multipliers.get(emotion, 1.0)
        # total_pause = int((punct_pause + boundary_pause) * emotion_mult + 0.5)
        # return max(1, total_pause)

        return 2  # Hardcoded minimal decay for natural ending

    def calculate_silence_duration_ms(self,
                                    boundary_type: BoundaryType) -> int:
        """
        Get exact silence duration in milliseconds for the given boundary type.
        Uses a strict hierarchy: Chapter > Paragraph > Sentence.
        
        Args:
            boundary_type: Type of boundary
            
        Returns:
            Duration in milliseconds
        """
        # Strict hierarchy lookup
        if boundary_type == BoundaryType.CHAPTER_START:
            return self.boundary_pauses.get(BoundaryType.CHAPTER_START, 2000)
        elif boundary_type == BoundaryType.PARAGRAPH_BREAK:
            return self.boundary_pauses.get(BoundaryType.PARAGRAPH_BREAK, 600)
        else:
            return self.boundary_pauses.get(BoundaryType.SENTENCE_END, 200)

    def get_emotion_stats(self, emotion_results: list) -> Dict[str, Any]:
        """
        Calculate statistics about emotion distribution.

        Args:
            emotion_results: List of emotion analysis results

        Returns:
            Statistics dictionary
        """
        if not emotion_results:
            return {}

        emotion_counts = {}
        confidence_sum = 0.0

        for result in emotion_results:
            emotion = result['dominant_emotion']
            confidence = result['confidence']

            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            confidence_sum += confidence

        total_chunks = len(emotion_results)
        avg_confidence = confidence_sum / total_chunks

        # Sort by frequency
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            'total_chunks': total_chunks,
            'emotion_distribution': emotion_counts,
            'dominant_emotion': sorted_emotions[0][0] if sorted_emotions else None,
            'average_confidence': avg_confidence,
            'emotion_rank': sorted_emotions
        }

    def validate_config(self) -> bool:
        """
        Validate that current configuration is complete and valid.

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check emotion mappings
            required_emotions = ['joy', 'surprise', 'anger', 'neutral', 'sadness', 'fear', 'disgust']
            for emotion in required_emotions:
                if emotion not in self.emotion_mappings:
                    logger.error(f"Missing emotion mapping for: {emotion}")
                    return False

                mapping = self.emotion_mappings[emotion]
                required_keys = ['temperature', 'speed_factor', 'eos_threshold']
                for key in required_keys:
                    if key not in mapping:
                        logger.error(f"Missing {key} in emotion mapping for: {emotion}")
                        return False

            # Check pause configurations
            if not self.punctuation_pauses:
                logger.error("No punctuation pause mappings defined")
                return False

            if not self.boundary_pauses:
                logger.error("No boundary pause mappings defined")
                return False

            if not self.emotion_multipliers:
                logger.error("No emotion pause multipliers defined")
                return False

            logger.info("ParameterMapper configuration validated successfully")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current configuration for debugging.

        Returns:
            Configuration summary
        """
        return {
            'emotion_mappings_count': len(self.emotion_mappings),
            'punctuation_pauses_count': len(self.punctuation_pauses),
            'boundary_pauses_count': len(self.boundary_pauses),
            'emotion_multipliers_count': len(self.emotion_multipliers),
            'speed_range': self.speed_range,
            'is_valid': self.validate_config()
        }