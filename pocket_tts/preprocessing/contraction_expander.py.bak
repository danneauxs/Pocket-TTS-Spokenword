"""Contraction expansion pre-processor for TTS.

This module provides text normalization that expands contractions before
they are tokenized by the SentencePiece tokenizer. This prevents mispronunciation
of contractions like "I'd" (pronounced as "ID").

Usage:
    from pocket_tts.preprocessing.contraction_expander import expand_contractions

    text = "I'd like to go, but I can't."
    expanded = expand_contractions(text)
    # Result: "I would like to go, but I cannot."
"""

import re
from typing import Dict


CONTRACTIONS_EXPAND: Dict[str, str] = {
    # "I'd" contractions
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "it'd": "it would",
    "there'd": "there would",
    "here'd": "here would",
    "that'd": "that would",
    "what'd": "what would",
    "who'd": "who would",
    "when'd": "when would",
    "where'd": "where would",
    "how'd": "how would",

    # "I'd've" contractions
    "i'd've": "i would have",
    "you'd've": "you would have",
    "he'd've": "he would have",
    "she'd've": "she would have",
    "we'd've": "we would have",
    "they'd've": "they would have",
    "it'd've": "it would have",

    # "shouldn't've" → "should not have" (nested contraction)
    "shouldn't've": "should not have",
    "couldn't've": "could not have",
    "wouldn't've": "would not have",
    "mightn't've": "might not have",
    "mustn't've": "must not have",
    "needn't've": "need not have",
    "daren't've": "dare not have",
    "oughtn't've": "ought not have",
    "you shouldn't": "you should not",
    "i shouldn't": "i should not",
    "we shouldn't": "we should not",
    "they shouldn't": "they should not",
    "he shouldn't": "he should not",
    "she shouldn't": "she should not",
    "it shouldn't": "it should not",

    # "I had" contractions (for past tense - "I'd" can mean "I had")
    "i'd already": "i had already",
    "i'd never": "i had never",
    "i'd always": "i had always",
    "i'd just": "i had just",
    "i'd already": "i had already",
    "i'd already": "i had already",

    # Edge case: "I'd" can mean "I had" in past tense
    # This is context-dependent, so we handle common phrases
    # For now, default to "I would" as more common

    # "don't" contractions
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "won't": "will not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "mustn't": "must not",
    "aren't": "are not",
    "isn't": "is not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "needn't": "need not",
    "daren't": "dare not",
    "oughtn't": "ought not",
    "mightn't": "might not",
    "oughtn't": "ought not",

    # "I'm" contractions
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "who's": "who is",
    "when's": "when is",
    "where's": "where is",
    "how's": "how is",
    "here's": "here is",
    "there's": "there is",
    "let's": "let us",

    # "I've" contractions
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "he've": "he have",
    "she've": "she have",
    "it's": "it have",  # Note: it's = it is usually, but sometimes it have (archaic)

    # "I'll" contractions
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "there'll": "there will",
    "here'll": "here will",

    # "I'm" + "ve" (double contractions)
    "i'm've": "i am have",
    "you're've": "you are have",
    "we're've": "we are have",
    "they're've": "they are have",
    "it's've": "it is have",

    # "I'm" + "ll" (double contractions)
    "i'm'll": "i am will",
    "you're'll": "you are will",
    "we're'll": "we are will",
    "they're'll": "they are will",

    # "I've" + "ll" (triple contractions - rare but possible)
    "i've'll": "i have will",
    "you've'll": "you have will",
    "we've'll": "we have will",
    "they've'll": "they have will",

    # "d" after vowel - could be had/would/could
    "he'd": "he would",  # Already covered above
    "she'd": "she would",  # Already covered above
    "i'd": "i would",  # Already covered above

    # "s" after pronoun - could be is/has/us
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "when's": "when is",
    "how's": "how is",
    "here's": "here is",

    # Contractions with "re"
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "they're": "they are",

    # Contractions with "ll"
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",

    # Negative contractions
    "ain't": "am not",
    "ain't": "is not",
    "ain't": "are not",
    "y'all": "you all",
    "youda": "you would have",
    "woulda": "would have",
    "shoulda": "should have",
    "coulda": "could have",
    "mighta": "might have",
    "musta": "must have",
}


def expand_contractions(text: str) -> str:
    """Expand contractions in text to their full forms.

    This function replaces contractions like "I'd", "don't", "can't" with their
    expanded forms ("i would", "do not", "cannot") to prevent mispronunciation
    by the TTS model.

    Args:
        text: Input text that may contain contractions.

    Returns:
        Text with all contractions expanded to their full forms.

    Examples:
        >>> expand_contractions("I'd like to go, but I can't.")
        'i would like to go, but i cannot.'

        >>> expand_contractions("Don't worry, it's fine.")
        'do not worry, it is fine.'

        >>> expand_contractions("We'd better go now.")
        'we would better go now.'
    """
    result = text

    for contraction, expansion in CONTRACTIONS_EXPAND.items():
        # Use word boundary matching to avoid partial matches
        # Case-insensitive to handle capitalized contractions at sentence start
        pattern = r"\b" + re.escape(contraction) + r"\b"
        result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)

    return result


def expand_contractions_case_sensitive(text: str) -> str:
    """Expand contractions preserving original capitalization.

    This version preserves the original capitalization of the first letter
    of contractions at sentence start.

    Args:
        text: Input text that may contain contractions.

    Returns:
        Text with contractions expanded, first letter capitalized if original was.

    Examples:
        >>> expand_contractions_case_sensitive("I'd like to go.")
        'I would like to go.'

        >>> expand_contractions_case_sensitive("i'd like to go.")
        'i would like to go.'
    """
    result = text

    for contraction, expansion in CONTRACTIONS_EXPAND.items():
        pattern = r"\b" + re.escape(contraction) + r"\b"

        def replace_match(match: re.Match) -> str:
            original = match.group(0)
            expanded = expansion
            # Preserve capitalization
            if original[0].isupper():
                expanded = expansion[0].upper() + expansion[1:]
            return expanded

        result = re.sub(pattern, replace_match, result, flags=re.IGNORECASE)

    return result


def is_contraction(word: str) -> bool:
    """Check if a word is a known contraction.

    Args:
        word: A word to check.

    Returns:
        True if the word is a known contraction, False otherwise.

    Examples:
        >>> is_contraction("I'd")
        True

        >>> is_contraction("hello")
        False
    """
    word_lower = word.lower()
    return word_lower in CONTRACTIONS_EXPAND


def get_expansion(contraction: str) -> str | None:
    """Get the expansion for a contraction.

    Args:
        contraction: A contraction word.

    Returns:
        The expanded form of the contraction, or None if not a contraction.

    Examples:
        >>> get_expansion("I'd")
        'i would'

        >>> get_expansion("hello")
        None
    """
    return CONTRACTIONS_EXPAND.get(contraction.lower())


if __name__ == "__main__":
    test_texts = [
        "I'd like to go, but I can't.",
        "Don't worry, it's fine.",
        "I'm happy that you're here.",
        "We'd better go now.",
        "They wouldn't do that.",
        "He said he'd come.",
        "She told me she's busy.",
        "It's a beautiful day.",
        "Can't you see?",
        "I don't know what's happening.",
        "I'd've gone if I'd known.",
        "You shouldn't've done that.",
        "We'll see when we're there.",
        "They've been here before.",
    ]

    print("=" * 60)
    print("CONTRACTION EXPANSION TEST")
    print("=" * 60)

    for text in test_texts:
        expanded = expand_contractions_case_sensitive(text)
        print(f"\nOriginal: {text}")
        print(f"Expanded: {expanded}")
