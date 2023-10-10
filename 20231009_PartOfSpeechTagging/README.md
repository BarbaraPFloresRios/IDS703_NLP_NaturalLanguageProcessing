# [Part Of Speech Tagging](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/tree/main/20231009_PartOfSpeechTagging)
### Bárbara Flores

**Part-of-speech tagging:**

Part-of-speech tagging or POS tagging, is a natural language processing technique that involves assigning grammatical categories or "parts of speech" (such as nouns, verbs, adjectives, and adverbs) to each word in a given text. This process is crucial for understanding the syntactic structure and meaning of sentences, as it helps identify the role that each word plays within a sentence. POS tagging is widely used in various NLP applications.

**Hidden Markov Model**

A Hidden Markov Model (HMM) is a statistical model that represents a system which transitions between different states over time. The key characteristic of an HMM is that the states are not directly observable; instead, they generate observable symbols or observations. Each state has a probability distribution over possible observations, and the system transitions from one state to another based on transition probabilities. The Viterbi algorithm is used with HMMs to find the most likely sequence of hidden states given a sequence of observed symbols.

**Viterbi:**

The Viterbi algorithm is a dynamic programming algorithm used for finding the most likely sequence of hidden states in a Hidden Markov Model (HMM) given a sequence of observations. It efficiently computes the maximum probability of a particular state sequence and is particularly useful in applications where the underlying states are not directly observable but can be inferred from observed data. The algorithm maintains a trellis or matrix of probabilities, calculating the most likely path to each state at each time step. By backtracking through the trellis, the algorithm identifies the optimal sequence of hidden states that best explains the observed data. Viterbi is widely employed in various fields, including speech recognition, natural language processing, and bioinformatics, for tasks where understanding the underlying sequence of states is crucial for accurate modeling and prediction.

The Viterbi algorithm is often applied in the context of POS tagging when POS tags are treated as hidden states in an HMM. It helps determine the most probable sequence of POS tags for a given sequence of observed words in a sentence, taking into account the inherent dependencies between POS tags and the likelihood of observing specific words given their POS tags.

**Assigment:**

In this context, the task is to modify and enhance the provided [input_viterbi.py](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/20231009_PartOfSpeechTagging/input_viterbi.py), which was given in the Introduction to NLP class, taught by Patrick Wang, to create a part-of-speech hidden Markov model (HMM).

For more details about the exercise, you can refer to the [assignment_instructions.pdf](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/20231009_PartOfSpeechTagging/assignment_instructions.pdf) file.

You can find my completed work for this project in [viterbi.py](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/20231009_PartOfSpeechTagging/viterbi.py) file

For this project, we trained a Hidden Markov Model for part-of-speech tagging using the first 10,000 tagged sentences from the `Brown Corpus`.

With this corpus, the following matrices were constructed, which served as input to train our model:

- π: initial state distribution
- A: transition matrix
- B: observation matrix or emission matrix"

Then, using the provided Viterbi implementation, the sequence of states for sentences 10150-10152 from the Brown corpus was inferred. 
The obtained result by comparing the actual POS with those predicted by the model is as follows:

```python
Word           : Real POS     Pred POS     Correct Prediction?

Those          : DET          DET          True
coming         : VERB         NOUN         False
from           : ADP          ADP          True
other          : ADJ          ADJ          True
denominations  : NOUN         NOUN         True
will           : VERB         VERB         True
welcome        : VERB         VERB         True
the            : DET          DET          True
opportunity    : NOUN         NOUN         True
to             : PRT          PRT          True
become         : VERB         VERB         True
informed       : VERB         VERB         True
.              : .            .            True
The            : DET          DET          True
preparatory    : ADJ          ADJ          True
class          : NOUN         NOUN         True
is             : VERB         VERB         True
an             : DET          DET          True
introductory   : ADJ          ADJ          True
face-to-face   : ADJ          NOUN         False
group          : NOUN         NOUN         True
in             : ADP          ADP          True
which          : DET          DET          True
new            : ADJ          ADJ          True
members        : NOUN         NOUN         True
become         : VERB         VERB         True
acquainted     : VERB         VERB         True
with           : ADP          ADP          True
one            : NUM          NUM          True
another        : DET          NOUN         False
.              : .            .            True
It             : PRON         PRON         True
provides       : VERB         VERB         True
a              : DET          DET          True
natural        : ADJ          ADJ          True
transition     : NOUN         NOUN         True
into           : ADP          ADP          True
the            : DET          DET          True
life           : NOUN         NOUN         True
of             : ADP          ADP          True
the            : DET          DET          True
local          : ADJ          ADJ          True
church         : NOUN         NOUN         True
and            : CONJ         CONJ         True
its            : DET          DET          True
organizations  : NOUN         NOUN         True
.              : .            .            True
```

We can see that out of 47 words in our text, we were able to correctly label 44 with our model. 
"In particular, the cases where different values were obtained were:
```python

Word           : Real POS     Pred POS     Correct Prediction?
coming         : VERB         NOUN         False
face-to-face   : ADJ          NOUN         False
another        : DET          NOUN         False
```
