# [Semantic Vectors](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/tree/main/20231101_SemanticVectors)
#### Daniela Jiménez
#### Bárbara Flores


### Project Introduction
In this project, we modify and enhance the provided [input_document_vectors_experiment.py](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/20231101_SemanticVectors/input_document_vectors_experiment.py), which was originally given during the Introduction to NLP class, taught by Patrick Wang. 

Our goal is to explore how dense word/document embeddings can be utilized for document classification and compare the performance of various models. The task at hand is to differentiate between documents authored by two distinct writers: `Lewis Carroll` and `Jane Austen`. We will compare four different types of dense document vectors, two of which were provided in the assignment instructions, and the remaining two were developed as part of this work.

For more details about the exercise, you can refer to the [assignment_instructions.pdf](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/20231101_SemanticVectors/%20assignment_instructions.pdf) file.

You can find our completed work for this project in [document_vectors_experiment.py](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/20231101_SemanticVectors/document_vectors_experiment.py) file

### Models

In the following section, we will explore four distinct models for document classification. Each of these models leverages various techniques to represent and classify documents. Central to these techniques is the concept of the 'bag of words,' which forms the foundation of traditional text analysis. While some models directly rely on the 'bag of words' approach by counting the occurrences of individual tokens, others transcend this concept to capture more nuanced aspects of document semantics. Let's delve into these models and discover how they harness the 'bag of words' and other sophisticated techniques to classify documents.

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

The percent correct for each method were:


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


1. **Raw Counts:**
   
   In our baseline case, we employed Raw Counts, where documents are represented as vectors of token counts. We achieved an accuracy of approximately 97% on our test data, which can be considered a robust performance for our model. Raw Counts provide a fundamental approach to document representation by considering the raw frequency of words in documents.

2. **TF-IDF:**
   
   When we transition to the TF-IDF model, we observe slightly improved results with an accuracy of 97.1% on our test data. TF-IDF represents an enhancement over Raw Counts as it accounts for the importance of words within the entire document collection. This method assigns higher weights to words that are common in one class of documents but distinctively frequent in another, thus enhancing the model's ability to discriminate between documents. The focus on term importance results in superior classification performance compared to Raw Counts, which solely considers word frequency. TF-IDF proves effective in capturing the discriminative power of words in the classification process.

3. **LSA:**
   
   Latent Semantic Analysis (LSA), which reduces token count dimensions to 300, exhibits slightly lower performance compared to Raw Counts, likely due to its focus on capturing underlying semantic information while potentially overlooking fine-grained details. This reduced dimensionality may not effectively capture textual nuances. LSA has limitations, such as sensitivity to term variability, as it treats terms as separate entities without considering semantic similarities, impacting its handling of word form variations and synonyms.

4. **Word2Vec:**

   In our evaluation, Word2Vec yields a lower classification performance with an accuracy of approximately 91%. This decrease in performance can be attributed to Word2Vec's reliance on pre-trained word vectors, which may not be perfectly aligned with the distinctive writing styles of Lewis Carroll and Jane Austen. It's important to consider that these authors wrote their works almost two centuries ago, while we are employing a pre-trained model derived from contemporary news data. This temporal misalignment could contribute to Word2Vec's challenges in capturing the unique textual characteristics and semantics of these historical writings.
