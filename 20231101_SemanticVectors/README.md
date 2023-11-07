# [Semantic Vectors](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/tree/main/20231101_SemanticVectors)
#### Daniela Jiménez
#### Bárbara Flores



In this project, we modify and enhance the provided [input_document_vectors_experiment.py](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/20231101_SemanticVectors/input_document_vectors_experiment.py), which was originally given during the Introduction to NLP class, taught by Patrick Wang. 

Our goal is to explore how dense word/document embeddings can be utilized for document classification and compare the performance of various models. The task at hand is to differentiate between documents authored by two distinct writers: `Lewis Carroll` and `Jane Austen`. We will compare four different types of dense document vectors, two of which were provided in the assignment instructions, and the remaining two were developed as part of this work.

For more details about the exercise, you can refer to the [assignment_instructions.pdf](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/20231101_SemanticVectors/%20assignment_instructions.pdf) file.

You can find our completed work for this project in [document_vectors_experiment.py](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/20231101_SemanticVectors/document_vectors_experiment.py) file

COMENTAR LO DE LA BOLSA DE WORDS

The models that were compared in this exercise were:

1. **Raw Counts (Token Counts)**:
   
   In this implementation, documents are represented as vectors of token counts. Each document is converted into a vector where each dimension represents the frequency of a token in the vocabulary. A logistic regression classifier is then used for classification.

2. **TF-IDF (Term Frequency-Inverse Document Frequency)**:

   In this implementation, the TF-IDF representation is used instead of token counts. Documents are transformed into vectors weighted by TF-IDF, which takes into account the importance of each word in the context of all the documents. A logistic regression classifier is used for classification.

3. **LSA (Latent Semantic Analysis):**

   In this implementation, Latent Semantic Analysis (LSA) is applied to the token count representation. LSA is a dimensionality reduction technique that seeks to extract underlying semantic information in the documents. TruncatedSVD is used to perform dimensionality reduction, reducing the document vectors to 300 dimensions. A logistic regression classifier is then trained on the data transformed by LSA.


4. **Word2Vec:**
   In this implementation, the pre-trained Google News Word2Vec model is used to represent the documents. Each word in a document is represented as a dense vector and then averaged to obtain a representation for the entire document, resulting in document vectors of 300 dimensions. A logistic regression classifier is used for classification.

These four models allow for a comparison of different approaches to document classification, ranging from simple token count-based representations to more advanced representations based on word semantics.


### Results

```python
raw counts (train): 0.9887267904509284
raw_counts (test): 0.9686567164179104
tfidf (train): 0.9993368700265252
tfidf (test): 0.9716417910447761
lsa (train): 0.9661803713527851
lsa (test): 0.9492537313432836
word2vec (train): 0.9348474801061007
word2vec (test): 0.9194029850746268
```
