# Gradient Descent
### Bárbara Flores


**Gradient descent:** 

Gradient descent is a fundamental optimization algorithm used in machine learning and various fields of computational mathematics. It serves as the backbone for training machine learning models and finding optimal solutions to complex problems. At its core, gradient descent is an iterative method employed to minimize a given cost or loss function. This optimization technique plays a crucial role in improving model accuracy and efficiency.


The primary objective of gradient descent is to locate the minimum of a cost function by iteratively adjusting the model's parameters. It achieves this by taking small steps in the direction of the steepest descent of the cost function. The "gradient" in gradient descent represents the rate of change of the cost function concerning the model parameters. This gradient points towards the direction of the steepest ascent, and the algorithm inverts this direction to minimize the function.


**PyTorch:** 

For this task, we will use the PyTorch library to optimize our model with gradient descent.


PyTorch is an open-source machine learning library that provides a flexible and dynamic framework for building and training deep neural networks. It is widely used in both research and production due to its intuitive design and support for GPU acceleration. PyTorch offers tools for automatic differentiation, making it easier to implement custom neural network architectures and optimize them using gradient-based methods like gradient descent. It has a rich ecosystem of libraries and community support, making it a popular choice for developing state-of-the-art machine learning models.

We will specifically use the Adam algorithm. The Adam algorithm (Adaptive Moment Estimation) is a popular and efficient optimization method used in deep learning and is available in PyTorch as part of the optimization package. Adam combines the advantages of stochastic gradient descent (SGD) with moments and adaptability in the learning rate to converge faster and more stably during the training of neural networks.

**Assigment:** 

In this context, the task is to modify and enhance the provided [input_unigram_pytorch.py](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/GradientDescent/input_unigram_pytorch.py), which was given in the Introduction to NLP class, to visualize the application of gradient descent in optimizing a unigram language model.

You can find my completed work for this assignment in [unigram_pytorch.py](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/GradientDescent/unigram_pytorch.py) file

For this task, I selected a learning rate of 0.1 and n = 100, and the result obtained was quite close to the expected one, within a reasonable timeframe

**Known Values:** 
Given our token, the optimal (known) probabilities of our vocabulary are:

```python
a: 0.06010	b: 0.01179	c: 0.01849	d: 0.03317	e: 0.09897	
f: 0.01817	g: 0.01419	h: 0.04794	i: 0.05427	j: 0.00141	
k: 0.00413	l: 0.03065	m: 0.02172	n: 0.05712	o: 0.06243	
p: 0.01181	q: 0.00090	r: 0.04940	s: 0.04869	t: 0.06686	
u: 0.02187	v: 0.00869	w: 0.01880	x: 0.00125	y: 0.01735	
z: 0.00010	 : 0.16154	None: 0.05821

"The log probability of document (known) is : -1956525.85084"
"The probability of document (known) is : 0.00000"
```

**Model Values:** 

After training our model with n = 100 iterations and a learning rate = 0.1, we obtained the following probabilities of our vocabulary:
```python
a: 0.06013	b: 0.01190	c: 0.01853	d: 0.03317	e: 0.09847	
f: 0.01822	g: 0.01413	h: 0.04793	i: 0.05428	j: 0.00145	
k: 0.00411	l: 0.03067	m: 0.02178	n: 0.05706	o: 0.06237	
p: 0.01191	q: 0.00113	r: 0.04939	s: 0.04870	t: 0.06664	
u: 0.02193	v: 0.00854	w: 0.01883	x: 0.00134	y: 0.01740	
z: 0.00077	 : 0.16103	None: 0.05818	

"The final log probability of the document obtained with the model is : -1956867.25000"
"The final probability of the document obtained with the model is : 0.00000"
```

It is worth mentioning that when dealing with a larger corpus (n = 673,022), when calculating the probability of the document, we obtain such a small value that tends to 0. This is why we work with logarithmic probabilities.

**Visualizations:** 

In the following graph, we can observe how the loss function evolves as we increase the iterations;
the line gradually approaches the known minimum possible loss.

timeframe.<img width="948" alt="Screen Shot 2023-09-27 at 23 44 36" src="https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/assets/143648839/334e9617-21a8-4a2c-9422-4420dc6b4213">

Additionally, if we graphically compare the final token probabilities with the (known) optimal probabilities,
we can see that they get quite close with 100 iterations.


<img width="988" alt="Screen Shot 2023-09-27 at 23 45 21" src="https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/assets/143648839/c6d66c13-363d-4a76-a111-71d8f05bf481">
