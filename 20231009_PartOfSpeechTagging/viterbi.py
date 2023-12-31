"""Viterbi Algorithm for inferring the most likely sequence of states from an HMM.

Patrick Wang, 2021
"""
from typing import Sequence, Tuple, TypeVar
import numpy as np
import nltk

Q = TypeVar("Q")
V = TypeVar("V")


def post_list_func(corpus):
    """
    This function will return a list with the possible part-of-speech (POS) tags
    from the corpus. We will use the order of this
    list to construct the pi, A, and B matrices later on.
    """
    pos_set = set()
    for sentence in corpus:
        for word, pos in sentence:
            pos_set.add(pos)
            pass
    return sorted(list(pos_set))


def vocabulary_func(corpus):
    """
    This function will return the vocabulary
    from the corpus. We will use the order of this
    list to construct B matrix later on.
    """
    vocabulary_set = set()
    for sentence in corpus:
        for word, pos in sentence:
            vocabulary_set.add(word)
    return sorted(list(vocabulary_set)) + [None]


def pi_func(corpus, post_list):
    """
    This function takes the corpus and our ordered list of POS tags as parameters
    and returns the initial state distribution matrix, pi.
    """
    output = np.ones(len(post_list))  # smoothing
    for sentence in corpus:
        for j in range(len(post_list)):
            if sentence[0][1] == post_list[j]:
                output[j] += 1
                pass
    return output / output.sum()


def transition_matrix_func(corpus, post_list):
    """
    This function takes the corpus and our ordered list of POS tags as parameters
    and returns the transition matrix, A.
    """
    n_pos = len(post_list)
    output = np.zeros((n_pos, n_pos))
    # (1 for smoothing)
    dict_transition_count = {(x, y): 1 for x in post_list for y in post_list}
    dict_pos_first_count = {i: 0 for i in post_list}
    dict_transition_percentage = {}

    # dict_transition_count counts the number of transitions between pairs of Part of Speech tags in the corpus.
    for sentence in corpus:
        for i in range(len(sentence) - 1):
            dict_transition_count[(sentence[i][1], sentence[i + 1][1])] += 1
            pass

    # The next dictionary counts how many times a specific part of speech appears at the beginning of a transition.
    for key, value in dict_transition_count.items():
        dict_pos_first_count[key[0]] += value
        pass

    # This dictionary stores the transition matrix.
    for key, value in dict_transition_count.items():
        dict_transition_percentage[key] = value / dict_pos_first_count[key[0]]
        pass

    # Now, we transform our final dictionary into an np.array.
    for row in range(n_pos):
        for column in range(n_pos):
            output[row, column] = dict_transition_percentage[
                (post_list[row], post_list[column])
            ]
            pass
    return output


def observation_matrix_func(corpus, post_list, vocabulary_list):
    """
    This function takes the corpus, our ordered list of POS tags, and the
    ordered list of vocabulary as parameters and returns the observation/emission matrix, B.
    """
    output = np.zeros((len(post_list), len(vocabulary_list)))

    dict_observation_word_pos_count = {
        (word, pos): 1 for word in vocabulary_list for pos in post_list
    }
    dict_observation_pos_count = {pos: 0 for pos in post_list}
    dict_observation_word_percentage = {}

    # The next dictionary counts how many times a word is tagged with each POS
    for sentence in corpus:
        for word_pos in sentence:
            dict_observation_word_pos_count[word_pos] += 1
            pass

    # The following dictionary counts the total occurrences of each pos.
    for key, value in dict_observation_word_pos_count.items():
        dict_observation_pos_count[key[1]] += value
        pass

    # This dictionary stores the observation matrix.
    for key, value in dict_observation_word_pos_count.items():
        dict_observation_word_percentage[key] = value / (
            dict_observation_pos_count[key[1]]
        )
        pass

    # Now, we transform our final dictionary into an np.array.
    for row in range(len(post_list)):
        for column in range(len(vocabulary_list)):
            output[row, column] = dict_observation_word_percentage[
                (vocabulary_list[column], post_list[row])
            ]
            pass

    return output


###


def viterbi(
    obs: Sequence[int],
    pi: np.ndarray[Tuple[V], np.dtype[np.float_]],
    A: np.ndarray[Tuple[Q, Q], np.dtype[np.float_]],
    B: np.ndarray[Tuple[Q, V], np.dtype[np.float_]],
) -> tuple[list[int], float]:
    """Infer most likely state sequence using the Viterbi algorithm.

    Args:
        obs: An iterable of ints representing observations.
        pi: A 1D numpy array of floats representing initial state probabilities.
        A: A 2D numpy array of floats representing state transition probabilities.
        B: A 2D numpy array of floats representing emission probabilities.

    Returns:
        A tuple of:
        * A 1D numpy array of ints representing the most likely state sequence.
        * A float representing the probability of the most likely state sequence.
    """
    N = len(obs)
    Q, V = B.shape  # num_states, num_observations

    # d_{ti} = max prob of being in state i at step t
    #   AKA viterbi
    # \psi_{ti} = most likely state preceeding state i at step t
    #   AKA backpointer

    # initialization
    log_d = [np.log(pi) + np.log(B[:, obs[0]])]
    log_psi = [np.zeros((Q,))]

    # recursion
    for z in obs[1:]:
        log_da = np.expand_dims(log_d[-1], axis=1) + np.log(A)
        log_d.append(np.max(log_da, axis=0) + np.log(B[:, z]))
        log_psi.append(np.argmax(log_da, axis=0))

    # termination
    log_ps = np.max(log_d[-1])
    qs = [-1] * N
    qs[-1] = int(np.argmax(log_d[-1]))
    for i in range(N - 2, -1, -1):
        qs[i] = log_psi[i + 1][qs[i + 1]]

    return qs, np.exp(log_ps)


# PARAMETERS
corpus = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]

# List with the order of POS for the following matrices
post_list = post_list_func(corpus)

# List with the order of vocabulary for the observation matrix
vocabulary_list = vocabulary_func(corpus)

# Matrix Pi - Initial state distribution
pi = pi_func(corpus, post_list)

# Matrix A - Transition matrix
transition_matrix = transition_matrix_func(corpus, post_list)

# Matrix B - Observation matrix
observation_matrix = observation_matrix_func(corpus, post_list, vocabulary_list)

# RESULTS
print(
    "After training our model, we tested it with the following 3 sentences from the Brown corpus:\n"
)

corpus_test = nltk.corpus.brown.tagged_sents(tagset="universal")[10150:10153]

list_of_sentence_pos_real = []
list_of_sentence_word = []
for sentence in corpus_test:
    sub_list_pos = []
    sub_list_word = []
    for word, pos in sentence:
        sub_list_pos.append(pos)
        sub_list_word.append(word)
    list_of_sentence_pos_real.append(sub_list_pos)
    list_of_sentence_word.append(sub_list_word)

print(list_of_sentence_word)
print(list_of_sentence_pos_real)

print(
    "\nThen, if we try to infer the sequence of states (POS label) for these sentences using our model,\nwe obtain the following:\n"
)

# List of lists containing the encoding of words within our vocabulary for each sentence.
list_of_encoded_sentences_voc = []
for sentence in corpus_test:
    sub_list = []
    for word, pos in sentence:
        if word in vocabulary_list:
            index = vocabulary_list.index(word)
        else:
            index = vocabulary_list.index(None)
        sub_list.append(index)
    list_of_encoded_sentences_voc.append(sub_list)

# "Now we use our list_of_encoded_sentences_voc to pass its n sentences through the Viterbi function."

list_of_encoded_sentences_pos = []
for encoded_sentence_voc in list_of_encoded_sentences_voc:
    list_of_encoded_sentences_pos.append(
        viterbi(encoded_sentence_voc, pi, transition_matrix, observation_matrix)[0]
    )

list_of_sentence_pos_result = []
for encoded_sentence_pos in list_of_encoded_sentences_pos:
    sub_list = []
    for code in encoded_sentence_pos:
        sub_list.append(post_list[code])
    list_of_sentence_pos_result.append(sub_list)

words = [
    word for list_of_sentence in list_of_sentence_word for word in list_of_sentence
]
real = [
    pos for list_of_sentence in list_of_sentence_pos_real for pos in list_of_sentence
]
result = [
    pos for list_of_sentence in list_of_sentence_pos_result for pos in list_of_sentence
]

for w, r, res in zip(words, real, result):
    print(f"{w: <15}: {r: <10} {res: <10} {r == res}")

print(
    "\nFinally, we can see that with our model, 44 out of 47 POS were predicted correctly."
)
