"""Demonstrate the "unigram" language model."""
from collections import Counter
import math
from typing import Optional, List

# generate English document
text = "Four score and seven years ago, our fathers brought forth, upon this continent, a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal."
text = (text + " ") * 100  # make the document very long

# tokenize - split the document into a list of little strings
tokens = [char for char in text]

# encode as {0, 1}
vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> List[int]:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = [0 for _ in range(len(vocabulary))]
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx] = 1
    return embedding


encodings = [onehot(vocabulary, token) for token in tokens]


# define model
class UnigramModel:
    """The unigram language model."""

    def __init__(self) -> None:
        """Initialize."""
        self.p: Optional[List[float]] = None

    def train(self, encodings: List[List[int]]) -> "UnigramModel":
        """Train the model on data."""
        counts = [1 for _ in range(len(vocabulary))]  # add-1 smoothing!!
        for encoding in encodings:
            for v in range(len(encoding)):
                counts[v] += encoding[v]
        self.p = [count / len(encodings) for count in counts]
        return self

    def apply(self, encodings: List[List[int]]) -> float:
        """Compute the log probability of a document."""
        if self.p is None:
            raise ValueError("This model is untrained")
        return sum(
            math.log(dot_product(encoding, self.p)) for encoding in encodings
        )


def dot_product(a: List[int], b: List[float]) -> float:
    """Compute the dot product of vectors a and b."""
    return sum(a_i * b_i for a_i, b_i in zip(a, b))


# train model
model = UnigramModel()
model.train(encodings)

# compute probability
test_data = "The quick brown fox jumps over the lazy dog."
tokens = [char for char in test_data]
encodings = [onehot(vocabulary, token) for token in tokens]
log_p = model.apply(encodings)

# print
print(f"learned p value: {model.p}")
print(f"log probability of document: {log_p}")
print(f"probability of document: {math.exp(log_p)} (this is due to underflow)")
