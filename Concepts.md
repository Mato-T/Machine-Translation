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

## Self-Attention
- Regarding transformers, self-attention uses a slightly different approach. The first step in calculating self-attention is to create three vectors from each of the encoder’s input vectors. For each word, a query, a key, and a value vector is created. These vectors are created by multiplying the embedding by three weight matrices that are trained during the training process
    
    $$q_n=x_n\*W_q$$
    $$k_n=x_n\*W_k$$
    $$v_n=x_n\*W_v$$
    
- The second step in calculating self-attention is to calculate a score. The score is calculated by taking the dot product of the query vector of the currently processed word with the key vector of each respective word in the sentence. This will result in a context of relevance in relation to the currently processed word
    
    $$s_1=q_1k_1$$
    $$s_2=q_1k_2$$
    $$s_n=q_1k_n$$
    
- The third and fourth steps are to divide the scores by the square root of the dimension of the key vector. This leads to having more stable gradients. These results are passed through a softmax operation, which normalizes the scores so they are all positive and add up to 1
    
    $$\alpha=sm(\frac{s_1}{\sqrt{d_k}}, \frac{s_2}{\sqrt{d_k}}, \frac{s_n}{\sqrt{d_k}})$$
    
- This softmax score determines how much each word will be expressed at this position. The current word will have the highest softmax score, but it also gives other words some weights to indicate some relevance to the current word
- The next step is to multiply all the softmax values with their respective value vectors. Lastly, the weighted value vectors are summed up to generate the output of the self attention layer for the particular word
    
    $$z_1=\sum^n_{n=1}\alpha_nv_n$$
    
- This process is repeated for every input embedding, calculating weights in relation to the currently processed word. Note that for faster processing, the calculations are done in matrix form, meaning the the word embeddings are packed into a matrix, so that all words are processed at once
- This means when multiplying by the different weight matrices, a query, key, and value matrix is obtained, resulting in a Z matrix. These steps are summarized by the following equation
    
    $$Z=sm(\frac{QK^T}{\sqrt{d_k}})V$$
## Multiheaded Attention
- Self-attention is further refined by adding a mechanism called multiheaded atttention. It expands the model’s ability to focus on different positions
- It also gives the attention multiple representation subspaces. With multiheaded attention, there are several sets of query, key, and value weight matrices. Each of these sets is randomly initialized
- This means, the same self-attention calculation is done, just with different times (depending on the head size) with different weight matrices, ending up with many different $Z$ matrices
- The feed-forward layer, however, is not expecting matrices — it is expecting a single matrix (vector for each word). So these matrices need to be condensed down to a single matrix. This is done by concatenating the matrices and then multiply them by an additional weight matrix
- The result would be the $Z$ matrix that captures information from all the attention heads, which can be passed into the feed-forward neural network

    ![mha](https://user-images.githubusercontent.com/127037803/225584188-b714fc96-37d1-48d1-a607-9bed862ac4aa.png)
    
 # Positional Encoding
 - The traditional embedding approaches are fast and potentially accurate but often overfit the data since they lack the sequential information
- One solution might be when the embeddings themselves contain information about their relative order. This is called a positional encoding
- The location of the word can be represented as vectors, which are added to the inputs so the input contain themselves and information about where each item is in the input
- This encodes sequential information into the network’s inputs. The hidden layers still operate independently but now it can learn to extract the position information from the input
- There is a sequence of embeddings $h_1, h_2,...h_T$ where $h_i\in\mathbb R^D$ represents the embedding after feeding in token $x_i$
- A position vector $P(t)$ can be added to the embeddings to create an improved embedding $\tilde h_t$ that contains information about the original content $h_t$ and its location as the $t$ th item in the input

    $$\tilde h_t=h_t+P(t)$$

- A function can be defined for $P(t)$ using the sine and cosine function, and the input to the sine and cosine functions to represent the position of a vector $t$. Oscillations are used instead of their actual position, because the later the word appears, the higher the number would be, distorting the representation
- The sine function oscillates up and down. If $sin(t)=y$ is calculated, knowing $y$ tells something about what input $t$ might have been. For example, if $y=0$ and there multiple oscillations, there are many possible positions this input could be at
- This situation can be improved by adding a second sine call, but with a frequency component $f$. So $sin(t/f)$ is computed, once with $f=1$ and once with $f=10$

    ![pos](https://user-images.githubusercontent.com/127037803/225593420-b17dd802-0035-48bd-840e-6c0099011251.png)

- Now, with two values of $f$, it is possible to uniquely identify some positions. If $sin(t)=0$ and $sin(t/100)=0$, there are four possible positions the input could be at: $t=0, 31, 68, 94$, which are the only four positions where both cases are true
- This shows that when adding frequencies $f$ to the calculation, the exact location within the input sequence can be inferred from the combination of values
- A position encoding function $P(t)$ is defined that returns a D dimensional vector by creating sine and cosine values at different frequencies: $f_1, f_2,...f_{D/2}$ (D/2 is used because both sine and cosine values are used for the frequency)

   ![pos_f](https://user-images.githubusercontent.com/127037803/225590502-8e6d2fbd-ba6c-4855-b2e4-5cec59f19b31.png)

- In this case, the frequency $f_i$ is defined as $f_i=10000^{\frac {2i}D}$, where $D$ represents the total dimension of the positional embedding and $i$ represents the indexes of each of the positional dimensions
- This means that there are as many different frequency plots as there are position dimensions because each $i$ represents a different frequency. The more frequencies there are, the easier it becomes to identify unique positions in time
    
# Transformer Architecture
## Encoder
- The encoder starts by processing the input sequence (French sentence). To address the order of the words (because they are not processed sequentially), the transformer adds positional encoding. Next, the result of adding the positional encoding to the input embedding is then passed into the self-attention layer to create some context of the words relative to each other
- The transformer network also makes use of residual connections. Each sub-layer in each encoder has a residual connection around it, and is followed by a layer-normalization step
- The output of the top encoder is then transformed into a set of attention vectors K and V. These are to be used by each decoder in its encoder-decoder attention layer, which helps the decoder focus on appropriate places in the input sequence

## Decoder
- As done with the encoder inputs, positional encoding is added to those decoder inputs to indicate the position of each word. The inputs to the decoder are the translated sentences, shifted to the right by 1 to implement teachers forcing. The loss of the first token (due to shifting) is made up by placing an SOS token at the first position.
- In the decoder, the self-attention layer is only allowed to calculate the context from earlier positions in the translated sequence. This is done by masking future positions before the softmax step in the self-attention calculations
- The encoder-decoder attention layer works like the multiheaded self-attention, except that it takes the Query matrix from the masked self-attention layer and the Key and Value matrix from the last output of the encoder rather than generating its own
- The decoder uses residual connections as well, but due to its architecture, three residual connections are added. The decoder stack outputs a vector of floats, which are passed into the final linear layer, which is followed by a softmax layer
- The linear layer is simply a fully connected neural network that projects the vector produced by the stack of decoders into a logits vector of the size of the vocabulary — each cell corresponding to the score of a unique word
- The softmax layer then turns those scores into probability. The cell with the highest probability is chosen. Since the entire sequence is processed at once, it also generated the translation at once

    ![architecture](https://user-images.githubusercontent.com/127037803/225584793-2ede5568-3a8a-4503-a225-f555e0916354.png)

