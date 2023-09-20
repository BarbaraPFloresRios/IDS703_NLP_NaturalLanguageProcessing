# Markov Text Generation
### Bárbara Flores

This is a bare-bones Markov text generator for the 'Introduction to Natural Language Processing' class.

In this project, I developed my version of a Markov text generation that relies on n-grams for prediction.

Markov text generation is a powerful technique used in natural language processing, where the probability of each word in a sentence is determined based on the context provided by the previous n-grams, which are sequences of n words. This approach enables the model to generate text that maintains contextual relevance and coherency, making it particularly effective for tasks like text generation, language modeling, and even machine translation. By utilizing n-grams, the implementation can capture intricate dependencies and patterns within the text data, resulting in more meaningful and contextually accurate text generation.

The function uses the following parameters:

- a sentence [list of tokens] that we're trying to build on,
- n [int], the length of n-grams to use for prediction, and
- a source corpus [list of tokens]
- a flag indicating whether the process should be randomize [bool]

It returns an extended sentence until the first period (.), question mark (?), or exclamation point (!) is found OR until it has 10 total tokens.


**Random process:** If *randomize* is True, the next word is drawn randomly, considering the probability of each n-gram in a weighted manner.

**Deterministic process:** If the input flag *randomize* is False, the algorithm chooses, at each step, the single most probable next token. When two tokens are equally probable, it chooses the one that occurs first in the corpus.

The algorithm uses stupid backoff and no smoothing.

The developed code is located in the file [mtg.py](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/MarkovTextGenerator/mtg.py)

For more details about the exercise, you can refer to the [hw_ngrams.pdf](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/MarkovTextGenerator/assignment_instructions.pdf) file.

