# Gradient Descent
### Bárbara Flores


**Gradient descent:** 

Gradient descent is a fundamental optimization algorithm used in machine learning and various fields of computational mathematics. It serves as the backbone for training machine learning models and finding optimal solutions to complex problems. At its core, gradient descent is an iterative method employed to minimize a given cost or loss function. This optimization technique plays a crucial role in improving model accuracy and efficiency.


The primary objective of gradient descent is to locate the minimum of a cost function by iteratively adjusting the model's parameters. It achieves this by taking small steps in the direction of the steepest descent of the cost function. The "gradient" in gradient descent represents the rate of change of the cost function concerning the model parameters. This gradient points towards the direction of the steepest ascent, and the algorithm inverts this direction to minimize the function.


**PyTorch:** 

In this project, we utilize the PyTorch library for model optimization using gradient descent. PyTorch is a versatile open-source machine learning framework recognized for its dynamic design, GPU support, and extensive community backing. It simplifies custom neural network creation and gradient-based optimization, making it a preferred choice for both research and production. Specifically, we employ the Adam algorithm (Adaptive Moment Estimation) within PyTorch, a highly effective optimization technique that combines elements of stochastic gradient descent (SGD) with adaptive learning rates, ensuring quicker and more stable convergence during neural network training.

**Assigment:** 

In this context, the task is to modify and enhance the provided [input_unigram_pytorch.py](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/GradientDescent/input_unigram_pytorch.py), which was given in the Introduction to NLP class, to visualize the application of gradient descent in optimizing a unigram language model.

You can find my completed work for this assignment in [unigram_pytorch.py](https://github.com/BarbaraPFloresRios/IDS703_NLP_NaturalLanguageProcessing/blob/main/GradientDescent/unigram_pytorch.py) file

We used a text from 'Sentido y sensibilidad' by Jane Austen to train our unigram  model. I selected a learning rate of 0.1 and n = 100, and the result obtained was quite close to the expected one, within a reasonable timeframe

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
