"""
Smart chunking algorithm for audiobook text preprocessing.
Handles sentence vs paragraph modes with merge/split logic.
"""

import logging
from typing import List
from .schema import ChunkMetadata, BoundaryType, EmotionType, TextStructure

logger = logging.getLogger(__name__)

class SmartChunker:
    """Creates optimal chunks from structured text with configurable modes."""

    def __init__(self,
                 mode: str = "sentence",
                 min_words: int = 5,
                 max_words: int = 50,
                 respect_boundaries: bool = True):
        """
        Initialize chunker with configuration.

        Args:
            mode: "sentence" or "paragraph" - chunking strategy
            min_words: Minimum words per chunk (prevents tiny chunks)
            max_words: Maximum words per chunk (model token limit)
            respect_boundaries: Whether to split at \n and chapter markers
        """
        if mode not in ["sentence", "paragraph"]:
            raise ValueError("mode must be 'sentence' or 'paragraph'")

        self.mode = mode
        self.min_words = min_words
        self.max_words = max_words
        self.respect_boundaries = respect_boundaries

        logger.info(f"SmartChunker initialized: mode={mode}, "
                   f"min_words={min_words}, max_words={max_words}")

    def chunk(self, structure: TextStructure) -> List[ChunkMetadata]:
        """
        Create chunks from analyzed text structure.

        Args:
            structure: Analyzed text structure from StructureDetector

        Returns:
            List of ChunkMetadata objects
        """
        logger.info(f"Starting chunking in {self.mode} mode")

        if self.mode == "sentence":
            chunks = self._chunk_by_sentences(structure)
        else:  # paragraph mode
            chunks = self._chunk_by_paragraphs(structure)

        logger.info(f"Chunking complete: {len(chunks)} chunks created")

        # Convert to ChunkMetadata with default values
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            metadata = ChunkMetadata(
                index=i,
                text=chunk['text'],
                word_count=chunk['word_count'],
                character_count=len(chunk['text']),
                boundary_type=chunk['boundary_type'],
                punctuation=chunk.get('punctuation', ''),
                start_position=chunk['start_position'],
                end_position=chunk['end_position'],
                emotion=EmotionType.NEUTRAL,  # Placeholder - will be analyzed later
                emotion_scores={'neutral': 1.0},
                emotion_confidence=1.0,
                tts_params={},
                post_process={}
            )
            chunk_metadata.append(metadata)

        return chunk_metadata

    def _chunk_by_sentences(self, structure: TextStructure) -> List[dict]:
        """
        STRICT sentence chunking: Create exactly one chunk per sentence.
        No merging, no word count limits, no minimums.
        Each sentence becomes its own chunk.
        """
        chunks = []

        for sentence in structure.sentences:
            # Create one chunk per sentence - no conditions or merging
            chunk = {
                'text': sentence.text,
                'word_count': sentence.word_count,
                'start_position': sentence.start_position,
                'end_position': sentence.end_position,
                'boundary_type': BoundaryType.SENTENCE_END,
                'punctuation': sentence.punctuation
            }
            chunks.append(chunk)

        return chunks

    def _chunk_by_paragraphs(self, structure: TextStructure) -> List[dict]:
        """
        Paragraph-based chunking: Accumulate sentences up to max_words,
        but break at paragraph boundaries.
        """
        chunks = []

        for paragraph in structure.paragraphs:
            # Find sentences in this paragraph
            para_sentences = self._get_sentences_in_paragraph(paragraph, structure.sentences)

            if not para_sentences:
                continue

            # Try to fit all sentences in one chunk
            total_words = sum(s.word_count for s in para_sentences)

            if total_words <= self.max_words:
                # Whole paragraph fits
                chunks.append(self._create_paragraph_chunk(para_sentences, paragraph))
            else:
                # Split paragraph into multiple chunks
                chunks.extend(self._split_paragraph(para_sentences, paragraph))

        return chunks

    def _get_sentences_in_paragraph(self, paragraph, all_sentences):
        """Get all sentences that belong to a paragraph."""
        return [s for s in all_sentences
                if paragraph.start_position <= s.start_position < paragraph.end_position]

    def _create_paragraph_chunk(self, sentences, paragraph):
        """Create a chunk from a complete paragraph."""
        text = ' '.join(s.text for s in sentences)
        return {
            'text': text,
            'word_count': sum(s.word_count for s in sentences),
            'start_position': paragraph.start_position,
            'end_position': paragraph.end_position,
            'boundary_type': BoundaryType.PARAGRAPH_BREAK,
            'punctuation': sentences[-1].punctuation if sentences else ''
        }

    def _split_paragraph(self, sentences, paragraph):
        """Split a long paragraph into multiple chunks."""
        chunks = []
        current_chunk_sentences = []
        current_word_count = 0
        chunk_groups = []  # Collect all chunk groups first

        for sentence in sentences:
            # Check if adding this sentence would exceed max_words
            if current_word_count + sentence.word_count > self.max_words and current_chunk_sentences:
                # Save current chunk group
                chunk_groups.append(current_chunk_sentences)
                current_chunk_sentences = []
                current_word_count = 0

            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_word_count += sentence.word_count

        # Save any remaining chunk group
        if current_chunk_sentences:
            chunk_groups.append(current_chunk_sentences)

        # Now create chunks with correct boundary types
        for i, chunk_sentences in enumerate(chunk_groups):
            is_last_chunk = (i == len(chunk_groups) - 1)
            boundary_type = BoundaryType.PARAGRAPH_BREAK if is_last_chunk else BoundaryType.SENTENCE_END
            chunks.append(self._create_partial_paragraph_chunk(
                chunk_sentences, paragraph, boundary_type))

        return chunks

    def _create_partial_paragraph_chunk(self, sentences, paragraph, boundary_type=None):
        """Create a chunk from part of a paragraph."""
        if boundary_type is None:
            boundary_type = BoundaryType.SENTENCE_END  # Backward compatibility

        text = ' '.join(s.text for s in sentences)
        return {
            'text': text,
            'word_count': sum(s.word_count for s in sentences),
            'start_position': sentences[0].start_position,
            'end_position': sentences[-1].end_position,
            'boundary_type': boundary_type,
            'punctuation': sentences[-1].punctuation
        }



    def get_statistics(self, chunks: List[ChunkMetadata]) -> dict:
        """Generate statistics about chunking results."""
        if not chunks:
            return {}

        word_counts = [c.word_count for c in chunks]
        boundary_types = [c.boundary_type.value for c in chunks]

        return {
            'total_chunks': len(chunks),
            'total_words': sum(word_counts),
            'avg_words_per_chunk': sum(word_counts) / len(chunks),
            'min_words_per_chunk': min(word_counts),
            'max_words_per_chunk': max(word_counts),
            'boundary_type_distribution': {
                bt: boundary_types.count(bt) for bt in set(boundary_types)
            }
        }