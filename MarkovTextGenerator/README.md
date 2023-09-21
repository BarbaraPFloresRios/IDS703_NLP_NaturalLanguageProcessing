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


**Random process:** 

If *randomize* is True, the next word is drawn randomly, considering the probability of each n-gram in a weighted manner.

**Deterministic process:** 

If the input flag *randomize* is False, the algorithm chooses, at each step, the single most probable next token. When two tokens are equally probable, it chooses the one that occurs first in the corpus.

The algorithm uses stupid backoff and no smoothing.

The developed code is located in the file [mtg.py](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/MarkovTextGenerator/mtg.py)

For more details about the exercise, you can refer to the [hw_ngrams.pdf](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/MarkovTextGenerator/assignment_instructions.pdf) file.

**Limitations**

A n-gram-based language model, like the Markov approach with n-grams, is a basic natural language processing model. In these models, probabilities of word or character occurrences are calculated based on sequences of the previous n words. While they are simple and efficient, these models have limitations such as fixed context and difficulties in capturing long-range dependencies. Nowadays, there are more advanced language models like Transformer-based models, including BERT, GPT-3, and their variants, which have surpassed traditional n-gram models in various natural language understanding and generation tasks. These advanced models employ attention mechanisms and deep learning techniques to capture complex language patterns, making them the state-of-the-art choice for many NLP applications.

Additionally, the present work does not aim to find the most efficient code execution method; rather, it intends to provide an example of code execution using algorithms that are as simple to understand as possible.

**Test cases**

In order to test the code, some test cases provided in class were executed. Finally, some personal test cases are also run.

For the random cases, a **seed** was defined to ensure result reproducibility.

- Clases Test Cases
```python
printing some test cases ...


sentence: ['she', 'was', 'not']
n: 3
corpus: austen-sense
randomize: False 
output: ['she', 'was', 'not', 'in', 'the', 'world', '.'] 


sentence: ['she', 'was', 'not']
n: 1
corpus: austen-sense
randomize: False 
output: ['she', 'was', 'not', ',', ',', ',', ',', ',', ',', ','] 


sentence: ['robot']
n: 3
corpus: austen-sense
randomize: False 
output: ['robot', ',', 'and', 'the', 'two', 'miss', 'steeles', ',', 'as', 'she'] 


sentence: ['robot']
n: 2
corpus: austen-sense
randomize: False 
output: ['robot', ',', 'and', 'the', 'same', 'time', ',', 'and', 'the', 'same'] 
```


- Personal Test Cases
```python
sentence: ['she', 'was', 'not']
n: 3
corpus: austen-sense
randomize: True 
output: ['she', 'was', 'not', 'an', 'inch', 'to', 'its', 'safety', '.'] 


sentence: ['robot']
n: 2
corpus: austen-sense
randomize: True 
output: ['robot', 'of', 'her', 'what', 'be', 'your', 'invitation', 'to', 'make', 'her']
```

For the following cases, we will use the Zen of Python as the corpus to train our model.
<em>

The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!<em>

```python
sentence: ['C+', 'Python']
n: 2
corpus: zen_of_python
randomize: False 
output: ['C+', 'Python', 'is', 'better', 'than', 'ugly.', 'Explicit', 'is', 'better', 'than'] 


sentence: ['easy']
n: 2
corpus: zen_of_python
randomize: True 
output: ['easy', 'to', 'explain,', 'it', 'may', 'not', 'be', 'obvious', 'at', 'first'] 


sentence: ['never', 'is']
n: 1
corpus: zen_of_python
randomize: True 
output: ['never', 'is', 'better', 'is', 'guess.', 'implementation', 'than', 'better', 'than', 'Sparse'] 
```
