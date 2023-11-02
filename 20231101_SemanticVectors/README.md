# [Semantic Vectors](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/tree/main/202311_SemanticVectors)
### Bárbara Flores

**Semantic Vectors:**

Semantic vectors" are numerical representations that capture the meaning of words, phrases, or documents in a vector space. These vectors are used in natural language processing to understand and measure semantic similarity between words or texts, which is crucial in tasks such as information retrieval, text classification, and machine translation.

**LSA**
LSA (Latent Semantic Analysis) is a technique in natural language processing and data analysis used to uncover semantic similarity patterns in a set of documents. It relies on the concept that words appearing in similar contexts tend to have similar meanings. By applying LSA, one can reveal semantic relationships and hidden structures within large text collections.

LSA involves dimensionality reduction using mathematical techniques like Singular Value Decomposition (SVD), transforming word and document representations into a lower-dimensional semantic space. This enables the discovery of semantic similarities between words and documents, making it valuable for tasks like information retrieval, text summarization, document clustering, and sentiment analysis.

Commonly used in natural language processing, LSA enhances the understanding of meaning in a text corpus. However, it's worth noting that LSA has limitations in capturing complex concepts and may not capture language nuances as effectively as newer approaches like neural network-based language models such as Word2Vec, GloVe, or BERT.

**Assigment:**

In this context, the task is to modify and enhance the provided [input_document_vectors_experiment.py](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/20231101_SemanticVectors/input_document_vectors_experiment.py), which was given in the Introduction to NLP class, taught by Patrick Wang, to explore how dense word/document embeddings can be used for document classification. The task is to to distinguish between documents from two different authors. We will implement two types of dense document vectors:

- Using Latent Semantic Analysis (LSA) on raw token counts.
- Summing pretrained Word2Vec embeddings.

For more details about the exercise, you can refer to the [assignment_instructions.pdf](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/20231101_SemanticVectors/%20assignment_instructions.pdf) file.

You can find my completed work for this project in [document_vectors_experiment.py](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/20231101_SemanticVectors/document_vectors_experiment.py) file
