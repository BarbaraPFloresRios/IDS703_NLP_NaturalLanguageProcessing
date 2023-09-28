"""Pytorch."""
import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
import matplotlib.pyplot as plt
import math


FloatArray = NDArray[np.float64]


def onehot(
    vocabulary: List[Optional[str]], token: Optional[str]
) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def logit(x: FloatArray) -> FloatArray:
    """Compute logit (inverse sigmoid)."""
    return np.log(x) - np.log(1 - x)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize vector so that it sums to 1."""
    return x / torch.sum(x)


def loss_fn(p: float) -> float:
    """Compute loss to maximize probability."""
    return -p


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct initial s - corresponds to uniform p
        s0 = logit(np.ones((V, 1)) / V)
        self.s = nn.Parameter(torch.tensor(s0.astype("float32")))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        p = normalize(torch.sigmoid(self.s))

        # compute log probability of input
        return torch.sum(input, 1, keepdim=True).T @ torch.log(p)


def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype("float32"))

    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations =  100
    learning_rate =  0.1
    loss_history = []  #We create a list to store the values of the loss function as we train the model. 

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(num_iterations):
        p_pred = model(x)
        loss = -p_pred
        loss_history.append(loss.item()) #We save the value of the loss function in our 'loss_history' list.
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

    # Calculating real values of the minimum possible loss and optimal probabilities.
    # Real token probabilities (known)
    p_real = np.sum(encodings, axis=1) / np.sum(encodings)

    # Minimum possible loss (known)
    log_p_real = np.log(encodings.T @ p_real).sum()
    loss_real = -log_p_real

    # Final token probabilities 
    p_final = normalize(torch.sigmoid(model.s)).detach().numpy().flatten() 
    
    # Final value of our loss function, obtained after n iterations.
    log_p_final = -loss.item() 
    loss_final = loss.item() # Final value of our loss function, obtained after n iterations.
    

    # printing results  
    print("Given our token, the optimal (known) probabilities of our vocabulary are as follows:\n")
    for i in range(len(vocabulary)):
        print (f"{vocabulary[i]}: {p_real[i]:.5f}", end='\t')
        if (i+1)%5 == 0:
            print()

    print(f"\n\nThe log probability of document (known) is : {log_p_real:.5f}")
    print(f"The probability of document (known) is : {math.exp(log_p_real):.5f}")

    print(f"\nAfter training our model with n = {num_iterations} iterations and a learning rate of {learning_rate}, we obtained the following results:")

    for i in range(len(vocabulary)):
        print (f"{vocabulary[i]}: {p_final[i]:.5f}", end='\t')
        if (i+1)%5 == 0:
            print()

    print(f"\n\nThe log probability of document (known) is : {log_p_final:.5f}")
    print(f"The probability of document (known) is : {math.exp(log_p_final):.5f}")  
    print("\nIt is worth mentioning that when dealing with a larger corpus (n = 673,022),\nwhen calculating the probability of the document, we obtain such a small\nvalue that tends to 0. This is why we work with logarithmic probabilities.")
    

    # display results
    plt.plot(range(len(loss_history)), loss_history, marker='o', linestyle='-')
    plt.axhline(y=loss_real, color='red', linestyle='--', label=f'y = {loss_real}')
    plt.show()



    vocabulary_text = [str(i) for i in vocabulary] # in order to transform None to "None"
    plt.clf()
    plt.bar( vocabulary_text,  p_real, color='pink')
    plt.show()


  
    plt.clf()
    plt.bar( vocabulary_text,  p_final, color='purple')
    plt.show()


    print(len(text))
if __name__ == "__main__":
    gradient_descent_example()


