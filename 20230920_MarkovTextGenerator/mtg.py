# bare-bones text generator

import nltk
import random
from collections import Counter


def create_n_gram_dictionary(n, corpus):
    tokens = [tuple(corpus[i : i + n]) for i in range(len(corpus) - n + 1)]
    n_gram_dictionary = dict(Counter(tokens))
    return n_gram_dictionary


def next_word(corpus, n, sentence, randomize):
    if n == 1:
        last_n_words = ()
    elif n >= 1:
        last_n_words = sentence[-n + 1 :]

    n_gram_dictionary = create_n_gram_dictionary(n, corpus)
    n_gram_dictionary_subset = {}

    for key, value in n_gram_dictionary.items():
        if key[:-1] == tuple(last_n_words):
            n_gram_dictionary_subset[key[-1]] = value
            pass

    if len(n_gram_dictionary_subset) > 0:
        if randomize == True:
            weighted_subset = []
            for key, value in n_gram_dictionary_subset.items():
                for i in range(value):
                    weighted_subset.append(key)
                    pass
            return random.choice(
                weighted_subset
            )  # Select a random probability, giving higher probability to instances with greater occurrence.

        else:
            for key, value in n_gram_dictionary_subset.items():
                if value == max(n_gram_dictionary_subset.values()):
                    return key
    else:
        return next_word(corpus, n - 1, sentence, randomize)


def finish_sentence(sentence, n, corpus, randomize=False):
    while True:
        new_word = next_word(corpus, n, sentence, randomize)
        sentence.append(new_word)

        if new_word in (".", "?", "!") or len(sentence) >= 10:
            break
        pass
    return sentence


if __name__ == "__main__":
    random.seed(123)
    austen_sense = nltk.word_tokenize(
        nltk.corpus.gutenberg.raw("austen-sense.txt").lower()
    )
    zen_of_python = [
        "Beautiful",
        "is",
        "better",
        "than",
        "ugly.",
        "Explicit",
        "is",
        "better",
        "than",
        "implicit.",
        "Simple",
        "is",
        "better",
        "than",
        "complex.",
        "Complex",
        "is",
        "better",
        "than",
        "complicated.",
        "Flat",
        "is",
        "better",
        "than",
        "nested.",
        "Sparse",
        "is",
        "better",
        "than",
        "dense.",
        "Readability",
        "counts.",
        "Special",
        "cases",
        "aren't",
        "special",
        "enough",
        "to",
        "break",
        "the",
        "rules.",
        "Although",
        "practicality",
        "beats",
        "purity.",
        "Errors",
        "should",
        "never",
        "pass",
        "silently.",
        "Unless",
        "explicitly",
        "silenced.",
        "In",
        "the",
        "face",
        "of",
        "ambiguity,",
        "refuse",
        "the",
        "temptation",
        "to",
        "guess.",
        "There",
        "should",
        "be",
        "one--",
        "and",
        "preferably",
        "only",
        "one",
        "--obvious",
        "way",
        "to",
        "do",
        "it.",
        "Although",
        "that",
        "way",
        "may",
        "not",
        "be",
        "obvious",
        "at",
        "first",
        "unless",
        "you're",
        "Dutch.",
        "Now",
        "is",
        "better",
        "than",
        "never.",
        "Although",
        "never",
        "is",
        "often",
        "better",
        "than",
        "*right*",
        "now.",
        "If",
        "the",
        "implementation",
        "is",
        "hard",
        "to",
        "explain,",
        "it's",
        "a",
        "bad",
        "idea.",
        "If",
        "the",
        "implementation",
        "is",
        "easy",
        "to",
        "explain,",
        "it",
        "may",
        "be",
        "a",
        "good",
        "idea.",
        "Namespaces",
        "are",
        "one",
        "honking",
        "great",
        "idea",
        "--",
        "let's",
        "do",
        "more",
        "of",
        "those!",
    ]

    test_cases = [
        # some test cases for the assigment
        (["she", "was", "not"], 3, austen_sense, False, "austen-sense"),
        (["she", "was", "not"], 1, austen_sense, False, "austen-sense"),
        (["robot"], 3, austen_sense, False, "austen-sense"),
        (["robot"], 2, austen_sense, False, "austen-sense"),
        # Other cases (the same as first, but using randomize = True)
        (["she", "was", "not"], 3, austen_sense, True, "austen-sense"),
        (["robot"], 2, austen_sense, True, "austen-sense"),
        # Other cases with zen of python
        (["C++", "Python"], 2, zen_of_python, False, "zen_of_python"),
        (["easy"], 2, zen_of_python, True, "zen_of_python"),
        (["never", "is"], 1, zen_of_python, True, "zen_of_python"),
    ]
    print("\nprinting some test cases ...")
    for sentence, n, corpus, randomize, courpus_name in test_cases:
        print(
            f"\n\nsentence: {sentence}\nn: {n}\ncorpus: {courpus_name}\nrandomize: {randomize} \noutput: {finish_sentence(sentence, n, corpus, randomize)} "
        )
