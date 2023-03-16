# Attention
## Simple Attention
- The attention mechanism is essentially a way to non-uniformly weight the contributions of the input feature vectors so as to optimize the process of learning; it ignores some part of the input, while emphasizing others
- The way attention mechanism usually work is that the input is represented as T different components $x_1, x_2, …, x_T$, each of which has a tensor representation $x_t\in\mathbb R^D$. This can be words of a sentence or sub-images of a single image
- Each input $x_t$ is transformed into a new tensor $h_t=F(x_t)$, where $F()$ is a neural networks. This means $x_t$ does not have to be a one-dimensional tensor. However, $h_t$ generally should be one-dimensional
- So the processed sequence $h_1, h_2, …, h_T$ is the actual input to the attention mechanism. Next, an importance score $\tilde {\alpha}_i$ needs to be learned for each input $x_i$

$$ \tilde {\alpha}_i=score(F(x_i))$$

- The function $score()$ is another neural network. That means the model itself will learn how to score the inputs. Then, the importances are normalized into a probability distribution so that

$$\alpha_1, \alpha_2, … \alpha_T=sm(\tilde {\alpha}_1, \tilde {\alpha}_2, … ,\tilde {\alpha}_T)$$

- Next, the output of the attention mechanism $\bar x$ is calculated

$$\bar x=\sum_{i=1}^T\alpha_i*F(x_i)$$

## Adding Context
- The first improvement to the attention mechanism is adding context to the scores. Context means that the score for an item $x_i$ should depend on all the other items $x_{j\neq i}$. The attention mechanism uses global information to make local decisions
- The score network can be made a network that takes in two inputs: the tensor of inputs (B, T, H) and a second tensor of shape (B, H) containing a context for each sequence
- H is used to represent the number of features in the states $h_t=F(x_t)$, to make them distinct from the size D of the original inputs $x_i$
- The second context tensor of shape (B, H) has no sequence dimension T because this vector is intended to be used as the context for all T items of the first input. This means every item in a bag will get the same context for computing its score
- The simplest form of context is the average value of all the features extracted. So if there is $h_i=F(x_i)$, the average $\bar h$ can be computed from all of these to give the model a rough idea of all the options it has to choose from
- This makes the attention computation occur in three steps:
    1. Compute the average of the features extracted from each input. This treats everything as being equal in importance:
        
        $$\qquad\bar h=\frac1T\sum_{i=1}^TF(x_i)$$
        
    2. Compute the attention scores $\alpha$ for the T inputs. The only change there is that $\bar h$ is included as a second argument to each score computation:
    
        $$\alpha=sm(score(h_1,\bar h), score(h_2,\bar h),…,score(h_T,\bar h)$$
    
    1. Compute the weighted average of the extracted features (identical to before)
        
        $$\bar x=\sum_{i=1}^T\alpha_i*F(x_i)$$
        
- There are three common ways to compute the score from the representation $h$ and the context $\bar h$, commonly called the dot, general, and additive scores
## Additive Attention Score
- In this case, I have chosen the additive attention score, which works as follows: for a given item and its context, a vector $v$ is used and a matrix $W$ as the parameters for the layer, making it a small neural network

  $$\qquad score(h_t, \bar h)= tanh(W[h_t, \bar h])v^T$$

- This equation is a simple one-layer neural network. $W$ is the hidden layer, followed by tanh activation, and $v$ is the output layer with one output, which is necessary because the score should be a single value
- The context $\bar h$ is incorporated into the model by simply concatenating it with the item $h_t$, so that the item and its context are the inputs into this fully connected network
