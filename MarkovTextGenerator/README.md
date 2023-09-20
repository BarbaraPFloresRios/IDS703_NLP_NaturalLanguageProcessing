# Markov text generation
### Bárbara Flores

This is bare-bones Markov text generator for the "Introduction to Natural Language Processing" class. 

In this project, I developed my version of a Markov text generation, which uses the following parameters:

- a sentence [list of tokens] that we're trying to build on,
- n [int], the length of n-grams to use for prediction, and
- a source corpus [list of tokens]
- a flag indicating whether the process should be randomize [bool]

It returns an extended sentence until the first ., ?, or ! is found OR until it has 10 total tokens.

For more details about the exercise, you can refer to the [hw_ngrams.pdf](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/MarkovTextGenerator/assignment_instructions.pdf) file.
