import nltk
import random
from collections import Counter


def create_n_gram_dictionary(n, corpus):
    tokens = [tuple(corpus[i : i + n]) for i in range(len(corpus) - n + 1)]
    n_gram_dictionary = dict(Counter(tokens))
    return n_gram_dictionary


def next_word(corpus, n, sentence, randomize=False):
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
        return next_word(corpus, n - 1, sentence, randomize=False)


def finish_sentence(sentence, n, corpus, randomize=False):
    while True:
        new_word = next_word(corpus, n, sentence, randomize)
        sentence.append(new_word)

        if new_word in (".", "?", "!") or len(sentence) >= 10:
            break
        pass
    return sentence


# OJO!! ESTE VALOR CORPUS Y N VIENE DADO EN EL TEST, DEBERIA SER UN PARAMETRO

if __name__ == "__main__":
    print("probando casos de test/n")

    corpus = nltk.word_tokenize(nltk.corpus.gutenberg.raw("austen-sense.txt").lower())
    sentence = ["she", "was", "not"]
    n = 3
    randomize = False
    print(
        f"\n\nsentence: {sentence}\nn: {n}\nrandomize: {randomize} \nresultado: {finish_sentence(sentence, n, corpus, False)}"
    )

    sentence = ["she", "was", "not"]
    n = 1
    randomize = False
    print(
        f"\n\nsentence: {sentence}\nn: {n}\nrandomize: {randomize} \nresultado: {finish_sentence(sentence, n, corpus, False)}"
    )

    sentence = ["robot"]
    n = 3
    randomize = False
    print(
        f"\n\nsentence: {sentence}\nn: {n}\nrandomize: {randomize} \nresultado: {finish_sentence(sentence, n, corpus, False)}"
    )

    sentence = ["robot"]
    n = 2
    randomize = False
    print(
        f"\n\nsentence: {sentence}\nn: {n}\nrandomize: {randomize} \nresultado: {finish_sentence(sentence, n, corpus, False)}"
    )
