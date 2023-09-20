"""Markov Text Generator.

Patrick Wang, 2023

Resources:
Jelinek 1985 "Markov Source Modeling of Text Generation"
"""

import nltk

from mtg import finish_sentence


def test_generator():
    """Test Markov text generator."""
    corpus = nltk.word_tokenize(nltk.corpus.gutenberg.raw("austen-sense.txt").lower())

    words = finish_sentence(
        ["she", "was", "not"],
        3,
        corpus,
        randomize=False,
    )
    print(words)
    assert words == [
        "she",
        "was",
        "not",
        "in",
        "the",
        "world",
        ".",
    ] or words == [
        "she",
        "was",
        "not",
        "in",
        "the",
        "world",
        ",",
        "and",
        "the",
        "two",
    ]


if __name__ == "__main__":
    test_generator()
