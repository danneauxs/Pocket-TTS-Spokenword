"""
Structure detection for audiobook text preprocessing.
Detects chapters, paragraphs, and sentence boundaries.
"""

import re
import logging
from typing import List
from .schema import ChapterInfo, ParagraphInfo, SentenceInfo, TextStructure

logger = logging.getLogger(__name__)

class StructureDetector:
    """Detects chapters, paragraphs, and sentences in text."""

    # Chapter detection patterns
    CHAPTER_PATTERNS = [
        r'^Chapter\s+\d+',          # Chapter 1
        r'^Chapter\s+[IVXLCDM]+',   # Chapter IV (Roman numerals)
        r'^CHAPTER\s+\d+',          # CHAPTER 1
        r'^Ch\.\s*\d+',             # Ch. 1
        r'^\d+\.\s*$',              # "1." alone on line
        r'^Part\s+\d+',             # Part 1
        r'^Book\s+\d+',             # Book 1
        r'^Prologue\s*$',           # Prologue
        r'^Epilogue\s*$',           # Epilogue
        r'^Introduction\s*$',       # Introduction
    ]

    # Sentence ending punctuation
    SENTENCE_ENDERS = r'[.!?]+'

    def __init__(self):
        self.chapter_regexes = [re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                               for pattern in self.CHAPTER_PATTERNS]

    def analyze(self, text: str) -> TextStructure:
        """
        Perform complete structural analysis of input text.

        Args:
            text: Raw input text

        Returns:
            TextStructure: Complete analysis results
        """
        logger.info("Starting structural analysis of text (%d characters)", len(text))
        logger.info("Raw text sample: %r", text[:200])

        # FIRST: Normalize text formatting
        text = self.normalize_formatting(text)

        # Detect chapters
        chapters = self.detect_chapters(text)
        logger.info("Detected %d chapters", len(chapters))

        # Detect paragraphs
        paragraphs = self.detect_paragraphs(text)
        logger.info("Detected %d paragraphs", len(paragraphs))

        # Segment sentences within each paragraph (prevents spanning)
        sentences = []
        for paragraph in paragraphs:
            para_sentences = self.segment_sentences_in_paragraph(paragraph, text)
            sentences.extend(para_sentences)
        logger.info("Detected %d sentences", len(sentences))

        # Calculate statistics
        total_words = sum(len(s.text.split()) for s in sentences)
        total_characters = len(text)

        # Mark chapter and paragraph relationships
        sentences = self._mark_relationships(sentences, chapters, paragraphs)

        structure = TextStructure(
            chapters=chapters,
            paragraphs=paragraphs,
            sentences=sentences,
            total_words=total_words,
            total_characters=total_characters
        )

        logger.info("Structural analysis complete: %d words, %d sentences",
                   total_words, len(sentences))
        return structure

    def detect_chapters(self, text: str) -> List[ChapterInfo]:
        """
        Detect all chapter markers in the text.

        Returns list of ChapterInfo objects sorted by position.
        """
        chapters = []

        for i, regex in enumerate(self.chapter_regexes):
            matches = regex.finditer(text)
            for match in matches:
                # Extract chapter number if present
                chapter_num = None
                title = match.group(0).strip()

                # Try to extract number
                num_match = re.search(r'\d+', title)
                if num_match:
                    chapter_num = int(num_match.group(0))

                # Check if Roman numeral
                roman_match = re.search(r'[IVXLCDM]+', title, re.IGNORECASE)
                is_roman = roman_match is not None and chapter_num is None

                chapter_info = ChapterInfo(
                    title=title,
                    start_position=match.start(),
                    chapter_number=chapter_num,
                    is_roman_numeral=is_roman
                )
                chapters.append(chapter_info)

        # Sort by position and remove duplicates (if multiple patterns match same location)
        chapters.sort(key=lambda c: c.start_position)
        unique_chapters = []
        seen_positions = set()

        for chapter in chapters:
            if chapter.start_position not in seen_positions:
                unique_chapters.append(chapter)
                seen_positions.add(chapter.start_position)

        return unique_chapters

    def normalize_formatting(self, text: str) -> str:
        """Normalize text formatting for consistent processing."""
        # FIRST: Normalize Unicode punctuation to ASCII (critical for contractions)
        try:
            from .text_normalizer import normalize_unicode_punctuation
            text = normalize_unicode_punctuation(text)
        except ImportError:
            # Fallback if text_normalizer not available
            pass

        # Normalize excessive newlines to exactly 2
        text = self._normalize_newlines(text)
        # Placeholder for future formatting rules
        text = self._normalize_other_formatting(text)
        return text

    def _normalize_newlines(self, text: str) -> str:
        """Normalize 2+ consecutive newlines to exactly 2."""
        import re
        # Replace any sequence of 2+ newlines (with optional whitespace) with exactly \n\n
        return re.sub(r'\n\s*\n+', '\n\n', text)

    def _normalize_other_formatting(self, text: str) -> str:
        """Placeholder for additional formatting normalizations."""
        # Future extensions can be added here
        return text

    def detect_paragraphs(self, text: str) -> List[ParagraphInfo]:
        """
        Detect paragraph boundaries using double newlines.
        
        Returns list of ParagraphInfo objects.
        """
        paragraphs = []
        
        # Split by double newlines (paragraph breaks)
        para_texts = re.split(r'\n\s*\n', text.strip())
        
        current_pos = 0
        for para_text in para_texts:
            if not para_text.strip():  # Skip empty paragraphs
                continue
            
            start_pos = text.find(para_text.strip(), current_pos)
            if start_pos == -1:
                continue
            
            end_pos = start_pos + len(para_text)
            
            para_info = ParagraphInfo(
                start_position=start_pos,
                end_position=end_pos,
                text=para_text.strip()
            )
            paragraphs.append(para_info)
            
            current_pos = end_pos
        
        return paragraphs

    def save_paragraph_structure(self, paragraphs: List[ParagraphInfo], sentences: List[SentenceInfo], output_path: str):
        """
        Save paragraph structure to JSON file for debugging and verification.
        
        Args:
            paragraphs: List of detected paragraphs
            sentences: List of all sentences in text
            output_path: Path to save JSON file
        """
        try:
            import json
            from datetime import datetime
            
            # Build paragraph data with associated sentences
            paragraph_data = []
            for i, paragraph in enumerate(paragraphs):
                # Find sentences that belong to this paragraph
                para_sentences = [s for s in sentences 
                                if paragraph.start_position <= s.start_position < paragraph.end_position]
                
                paragraph_dict = {
                    'index': i,
                    'text': paragraph.text,
                    'start_position': paragraph.start_position,
                    'end_position': paragraph.end_position,
                    'sentence_count': len(para_sentences),
                    'sentences': [
                        {
                            'text': s.text,
                            'start_position': s.start_position,
                            'end_position': s.end_position,
                            'punctuation': s.punctuation,
                            'ends_paragraph': s.ends_paragraph
                        }
                        for s in para_sentences
                    ]
                }
                paragraph_data.append(paragraph_dict)
            
            # Build metadata
            metadata = {
                'source_file': output_path,
                'total_paragraphs': len(paragraphs),
                'total_sentences': len(sentences),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Build complete structure
            structure_data = {
                '_metadata': metadata,
                'paragraphs': paragraph_data
            }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structure_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Paragraph structure saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save paragraph structure: {e}")
            raise

    def segment_sentences(self, text: str) -> List[SentenceInfo]:
        """
        Segment text into sentences based on punctuation.

        Uses regex to split on sentence-ending punctuation while preserving
        the punctuation in the results.
        """
        sentences = []

        # Split text into sentences, keeping the delimiters
        # This regex splits on sentence enders but keeps them
        parts = re.split(r'([.!?]+)', text)

        current_pos = 0
        i = 0

        while i < len(parts):
            # Get sentence text (may span multiple parts)
            sentence_text = ""

            # Accumulate text until we hit punctuation
            while i < len(parts) and not re.match(r'[.!?]+', parts[i]):
                sentence_text += parts[i]
                i += 1

            if not sentence_text.strip():
                i += 1
                continue

            # Get punctuation (if available)
            punctuation = ""
            if i < len(parts):
                punctuation = parts[i]
                i += 1

            # Find position in original text
            start_pos = text.find(sentence_text.strip(), current_pos)
            if start_pos == -1:
                # Fallback: approximate position
                start_pos = current_pos

            end_pos = start_pos + len(sentence_text.strip()) + len(punctuation)

            # Create sentence info (paragraph ending will be set later)
            sentence_info = SentenceInfo(
                text=sentence_text.strip() + punctuation,
                start_position=start_pos,
                end_position=end_pos,
                word_count=len(sentence_text.split()),
                punctuation=punctuation,
                ends_paragraph=False  # Will be updated in _mark_relationships
            )

            sentences.append(sentence_info)
            current_pos = end_pos

        # Handle case where text doesn't end with punctuation
        if current_pos < len(text):
            remaining_text = text[current_pos:].strip()
            if remaining_text:
                ends_paragraph = self._check_ends_paragraph(text, len(text))
                sentence_info = SentenceInfo(
                    text=remaining_text,
                    start_position=current_pos,
                    end_position=len(text),
                    word_count=len(remaining_text.split()),
                    punctuation="",
                    ends_paragraph=ends_paragraph
                )
                sentences.append(sentence_info)

        return sentences

    def segment_sentences_in_paragraph(self, paragraph: ParagraphInfo, full_text: str) -> List[SentenceInfo]:
        """
        Segment sentences within a single paragraph's boundaries.

        This ensures sentences never span across paragraph breaks.
        """
        # Extract the paragraph's text from the full text
        para_start = paragraph.start_position
        para_end = paragraph.end_position
        para_text = full_text[para_start:para_end]

        sentences = []

        # Split paragraph text into sentences using the same logic as segment_sentences
        parts = re.split(r'([.!?]+)', para_text)
        
        current_pos = 0  # Relative to paragraph
        i = 0

        while i < len(parts):
            # Get sentence text (may span multiple parts)
            sentence_text = ""

            # Accumulate text until we hit punctuation
            while i < len(parts) and not re.match(r'[.!?]+', parts[i]):
                sentence_text += parts[i]
                i += 1

            if not sentence_text.strip():
                i += 1
                continue

            # Get punctuation (if available)
            punctuation = ""
            if i < len(parts):
                punctuation = parts[i]
                i += 1

            # Find position within paragraph text
            rel_start_pos = para_text.find(sentence_text.strip(), current_pos)
            if rel_start_pos == -1:
                # Fallback: approximate position
                rel_start_pos = current_pos

            rel_end_pos = rel_start_pos + len(sentence_text.strip()) + len(punctuation)

            # Convert to absolute positions
            abs_start_pos = para_start + rel_start_pos
            abs_end_pos = para_start + rel_end_pos

            # Create sentence info
            full_text = sentence_text.strip() + punctuation
            word_count = len(sentence_text.split())
            
            # Check if this is just closing punctuation (orphaned by regex split)
            # Smart quotes: ” (u201d), “ (u201c), etc.
            is_closing_punct = re.match(r'^[”"\'\)\]\}\s]+$', full_text)

            if is_closing_punct and sentences:
                # Append to previous sentence instead of creating a new one
                last_sentence = sentences[-1]
                logger.debug(f"Appending orphaned punctuation '{full_text}' to previous sentence: '{last_sentence.text}'")
                
                # Update previous sentence
                # We need to preserve the space between if it existed in original, but here we work with stripped chunks mostly.
                # However, full_text is stripped. 
                # Let's just append it. If there was a space, it might be lost here, 
                # but usually closing quotes attach directly.
                last_sentence.text += full_text
                last_sentence.end_position = abs_end_pos
                # Word count shouldn't change for punctuation
            else:
                sentence_info = SentenceInfo(
                    text=full_text,
                    start_position=abs_start_pos,
                    end_position=abs_end_pos,
                    word_count=word_count,
                    punctuation=punctuation,
                    ends_paragraph=False  # Only the last sentence ends the paragraph
                )

                sentences.append(sentence_info)

            current_pos = rel_end_pos

        # Update the last sentence to mark paragraph end
        if sentences:
            sentences[-1].ends_paragraph = True

        # Handle case where paragraph doesn't end with punctuation
        if current_pos < len(para_text):
            remaining_text = para_text[current_pos:]  # Don't strip yet to preserve relative position
            stripped_text = remaining_text.strip()
            
            if stripped_text:
                # Check if this is just closing punctuation (orphaned by regex split)
                # Smart quotes: ” (u201d), “ (u201c), etc.
                is_closing_punct = re.match(r'^[”"\'\)\]\}\s]+$', stripped_text)
                
                if is_closing_punct and sentences:
                    # Append to previous sentence instead of creating a new one
                    last_sentence = sentences[-1]
                    logger.debug(f"Appending orphaned punctuation '{stripped_text}' to previous sentence: '{last_sentence.text}'")
                    
                    # Update previous sentence
                    last_sentence.text += remaining_text # Append with original whitespace if any
                    last_sentence.text = last_sentence.text.strip() # Then strip the result
                    last_sentence.end_position = para_end
                    # Word count shouldn't change for punctuation
                else:
                    # Create new sentence as normal
                    word_count = len(stripped_text.split())
                    sentence_info = SentenceInfo(
                        text=stripped_text,
                        start_position=para_start + current_pos,
                        end_position=para_end,  # End of paragraph
                        word_count=word_count,
                        punctuation="",
                        ends_paragraph=True  # Last sentence in paragraph always ends it
                    )
                    sentences.append(sentence_info)

        # Update the last sentence to mark paragraph end (redundant but safe)
        if sentences:
            sentences[-1].ends_paragraph = True

        return sentences

    def _check_ends_paragraph(self, text: str, position: int) -> bool:
        """
        Check if text at given position is followed by paragraph-ending newlines.

        A paragraph ends if followed by:
        - \n\n (double newline)
        - \n\n\n+ (multiple newlines)
        - End of text
        """
        if position >= len(text):
            return True  # End of text always ends paragraph

        # Look ahead for newlines, allowing for whitespace
        remaining = text[position:]
        # Match \n\n or more, with optional whitespace
        newline_match = re.match(r'\s*\n\s*\n+', remaining)

        return newline_match is not None

    def _mark_relationships(self, sentences: List[SentenceInfo],
                          chapters: List[ChapterInfo],
                          paragraphs: List[ParagraphInfo]) -> List[SentenceInfo]:
        """
        Mark which sentences start chapters and end paragraphs.
        """
        # Create lookup sets for quick checking
        chapter_starts = {c.start_position for c in chapters}
        para_ends = {p.end_position for p in paragraphs}

        for sentence in sentences:
            # Check if this sentence starts a chapter
            if sentence.start_position in chapter_starts:
                sentence.is_chapter = True

            # Check if this sentence ends a paragraph
            # Allow some tolerance for text processing differences
            for para_end in para_ends:
                if abs(sentence.end_position - para_end) <= 5:  # 5 char tolerance
                    sentence.ends_paragraph = True
                    break

        return sentences

    def get_statistics(self, structure: TextStructure) -> dict:
        """
        Generate statistics about the structural analysis.
        """
        return {
            'total_characters': structure.total_characters,
            'total_words': structure.total_words,
            'total_sentences': len(structure.sentences),
            'total_paragraphs': len(structure.paragraphs),
            'total_chapters': len(structure.chapters),
            'avg_words_per_sentence': structure.total_words / len(structure.sentences) if structure.sentences else 0,
            'avg_sentences_per_paragraph': len(structure.sentences) / len(structure.paragraphs) if structure.paragraphs else 0,
        }