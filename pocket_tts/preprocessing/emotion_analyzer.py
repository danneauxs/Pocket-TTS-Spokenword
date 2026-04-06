"""
Emotion analysis for audiobook text preprocessing.
Uses DistilRoBERTa to detect emotions in text chunks.
"""

import logging
from typing import Dict, List, Optional, Any

from .schema import EmotionType

try:
    from transformers import pipeline, Pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None
    Pipeline = None
    torch = None

logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    """
    Analyzes emotions in text using DistilRoBERTa.

    Supports 7 emotions: joy, surprise, anger, neutral, sadness, fear, disgust
    """

    EMOTIONS = ['joy', 'surprise', 'anger', 'neutral', 'sadness', 'fear', 'disgust']
    MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

    def __init__(self, device: str = "auto", model_name: Optional[str] = None):
        """
        Initialize emotion analyzer.

        Args:
            device: Device to run model on ("auto", "cpu", "cuda")
            model_name: Custom model name (defaults to emotion model)
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not available. Using mock emotion analysis.")
            self.device = "cpu"
            self.model_name = model_name or self.MODEL_NAME
            self.pipeline = None
            return

        self.device = self._resolve_device(device)
        self.model_name = model_name or self.MODEL_NAME

        logger.info(f"Initializing EmotionAnalyzer with model: {self.model_name} on {self.device}")

        # Initialize pipeline
        self.pipeline: Optional[Pipeline] = None
        self._load_model()

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            if TRANSFORMERS_AVAILABLE and torch:
                return "cuda" if torch.cuda.is_available() else "cpu"
            else:
                return "cpu"
        return device

    def _load_model(self) -> None:
        """Load the emotion classification model."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Cannot load emotion model: transformers not available")
            return

        try:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,  # Return all emotion scores
                device=self.device,
                torch_dtype=torch_dtype
            )
            logger.info("Emotion model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            raise RuntimeError(f"Could not load emotion model: {e}")

    def analyze(self, text: str, keyword_boosts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze emotion in text.

        Args:
            text: Text to analyze
            keyword_boosts: Optional keyword-based emotion adjustments

        Returns:
            Dict with emotion analysis results
        """
        if not TRANSFORMERS_AVAILABLE:
            return self._mock_analyze(text, keyword_boosts)

        if not self.pipeline:
            raise RuntimeError("Emotion model not loaded")

        if not text.strip():
            return self._default_emotion_result()

        try:
            # Get raw emotion scores
            results = self.pipeline(text)[0]  # top_k=None returns list of dicts

            # Convert to emotion -> score mapping
            emotion_scores = {}
            for result in results:
                label = result['label'].lower()
                score = result['score']
                emotion_scores[label] = score

            # Ensure all emotions are present (fill missing with 0)
            for emotion in self.EMOTIONS:
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = 0.0

            # Find dominant emotion
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            dominant_label = dominant_emotion[0]
            confidence = dominant_emotion[1]

            # Apply keyword boosts if provided
            if keyword_boosts:
                dominant_label, confidence = self._apply_keyword_boosts(
                    text, emotion_scores, keyword_boosts
                )

            # Map string to enum
            emotion_enum = getattr(EmotionType, dominant_label.upper())

            return {
                'emotion': emotion_enum,
                'dominant_emotion': dominant_label,
                'confidence': confidence,
                'scores': emotion_scores
            }

        except Exception as e:
            logger.warning(f"Emotion analysis failed for text: {e}")
            return self._default_emotion_result()

    def _mock_analyze(self, text: str, keyword_boosts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Mock emotion analysis for when transformers isn't available.
        Returns neutral emotion with mock scores.
        """
        logger.info("Using mock emotion analysis (transformers not available)")

        # Create mock scores - neutral dominant
        emotion_scores = {emotion: 0.1 for emotion in self.EMOTIONS}
        emotion_scores['neutral'] = 0.7

        # Apply keyword boosts if provided
        dominant_label = 'neutral'
        confidence = 0.7

        if keyword_boosts:
            dominant_label, confidence = self._apply_keyword_boosts(
                text, emotion_scores, keyword_boosts
            )

        emotion_enum = getattr(EmotionType, dominant_label.upper())

        return {
            'emotion': emotion_enum,
            'dominant_emotion': dominant_label,
            'confidence': confidence,
            'scores': emotion_scores
        }

    def analyze_batch(self, texts: List[str],
                     keyword_boosts: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Analyze emotions for multiple texts efficiently.

        Args:
            texts: List of texts to analyze
            keyword_boosts: Optional keyword boosts

        Returns:
            List of emotion analysis results
        """
        if not TRANSFORMERS_AVAILABLE:
            return [self._mock_analyze(text, keyword_boosts) for text in texts]

        if not self.pipeline:
            raise RuntimeError("Emotion model not loaded")

        if not texts:
            return []

        try:
            # Batch process with pipeline
            results = self.pipeline(texts)

            batch_results = []
            for i, text_result in enumerate(results):
                text = texts[i]

                # Convert result format
                emotion_scores = {}
                for result in text_result:
                    label = result['label'].lower()
                    score = result['score']
                    emotion_scores[label] = score

                # Ensure all emotions present
                for emotion in self.EMOTIONS:
                    if emotion not in emotion_scores:
                        emotion_scores[emotion] = 0.0

                # Find dominant
                dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                dominant_label = dominant_emotion[0]
                confidence = dominant_emotion[1]

                # Apply keyword boosts
                if keyword_boosts:
                    dominant_label, confidence = self._apply_keyword_boosts(
                        text, emotion_scores, keyword_boosts
                    )

                emotion_enum = getattr(EmotionType, dominant_label.upper())

                batch_results.append({
                    'emotion': emotion_enum,
                    'dominant_emotion': dominant_label,
                    'confidence': confidence,
                    'scores': emotion_scores
                })

            return batch_results

        except Exception as e:
            logger.warning(f"Batch emotion analysis failed: {e}")
            return [self._default_emotion_result() for _ in texts]

    def _apply_keyword_boosts(self, text: str, emotion_scores: Dict[str, float],
                            boosts: Dict[str, Any]) -> tuple[str, float]:
        """
        Apply keyword-based emotion boosts.

        Args:
            text: Original text
            emotion_scores: Current emotion scores
            boosts: Boost configuration

        Returns:
            Tuple of (adjusted_emotion, adjusted_confidence)
        """
        text_lower = text.lower()
        boosts_applied = {}

        # Check each emotion's keywords
        for emotion, config in boosts.items():
            if emotion not in emotion_scores:
                continue

            keywords = config.get('keywords', [])
            boost_amount = config.get('temperature_boost', 0.0)

            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)

            if matches > 0:
                # Apply boost
                current_score = emotion_scores[emotion]
                new_score = min(current_score + boost_amount, 1.0)
                emotion_scores[emotion] = new_score
                boosts_applied[emotion] = new_score - current_score

        # If boosts were applied, recalculate dominant emotion
        if boosts_applied:
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            return dominant_emotion[0], dominant_emotion[1]

        # No boosts applied, return original
        original_dominant = max(emotion_scores.items(), key=lambda x: x[1])
        return original_dominant[0], original_dominant[1]

    def _default_emotion_result(self) -> Dict[str, Any]:
        """Return default emotion result for error cases."""
        return {
            'emotion': EmotionType.NEUTRAL,
            'dominant_emotion': 'neutral',
            'confidence': 1.0,
            'scores': {emotion: 1.0 if emotion == 'neutral' else 0.0
                      for emotion in self.EMOTIONS}
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'emotions_supported': self.EMOTIONS,
            'pipeline_loaded': self.pipeline is not None
        }

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            if TRANSFORMERS_AVAILABLE and torch:
                torch.cuda.empty_cache()  # Clear GPU cache if applicable
            logger.info("Emotion model unloaded")

    def reload_model(self) -> None:
        """Reload the model."""
        self.unload_model()
        self._load_model()