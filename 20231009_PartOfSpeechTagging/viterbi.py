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
    return list(pos_set)


def pi_func(corpus, post_list):
    """
    This function takes the corpus and our ordered list of POS tags as parameters
    and returns the initial state distribution matrix, π.
    """
    output = np.zeros(len(post_list))
    for sentence in corpus:
        for j in range(len(post_list)):
            if sentence[0][1] == post_list[j]:
                output[j] += 1 / len(corpus)
                pass
    return output


def transition_matrix_func(corpus, post_list):
    output = np.zeros((len(post_list), len(post_list)))
    for sentence in corpus:
        for i in range(len(sentence) - 1):
            first = sentence[i][1]
            second = sentence[i + 1][1]
            print(first, second)
            pass

    return output


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


corpus = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]
post_list = post_list_func(corpus)  # to get the order of pos
pi = pi_func(corpus, post_list)
transition_matrix = transition_matrix_func(corpus, post_list)


print(post_list)
print(pi)
print(transition_matrix)
