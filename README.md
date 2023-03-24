# Project Description
## Introduction
- The Transformer architecture introduced in the paper "Attention Is All You Need" was a major leap forwards in the development of more sophisticated Natural Language Models and it paved the way for popular models like GPT and BERT. As the time of this writing, major tech-companies like Microsoft, Google, OpenAI, Meta, etc. are announcing their own models that are based on the Transformer architecture.
- For this reason, I decided to take a deeper look into the architecture behind this revolutionary model and assessed sentiment analysis as a fitting project. The dataset consists of about 1.6 million tweets extracted using the Twitter API and is found on Kaggle using this URL: https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download
- Please note that the following information should be read along with the Python code in this repository and requires a more profound understanding of neural networks, the Attention mechanism. Each concept described in this documentation is explained in more detail in the Concept.md file. For more information on the cross-entropy loss, check out the Concept.md file in the Fraud-Detection repository.
## Preprocessing
- First, lets take a look at the distribution of length of each tweet:

  ![length](https://user-images.githubusercontent.com/127037803/227503757-7387def7-cf32-49ec-8dd8-ed964a9a945e.png)
- As seen in this plot, most tweets remain under the threshold of 150 characters, however, I will leave out any sample that exceeds 100 characters. I deemed this the appropriate threshold, since it allowed me greater freedom when it came to experimenting with hyperparameters (I provide more information in the conclusion section). However, this also limits the capability to generalize well to longer sequences.
- The dataset seems to label positive sentiment as 0 and negative sentiment as 4. A better representation of the target variable is to use 0 and 1 instead. I also dropped any rows that reported missing values.
- Next, a vocabulary must be created to implement the embeddings of the tokens. Special tokens, such as for padding or start-of-sentence, are includes as weel. I also split the dataset randomly to create a training and test dataset.
- Since I use batched samples, all sentences must be padded to fit the length of the longest sentence since the Transformer expects the input to be in a unified dimension. These are all the steps required to prepare the dataset for the model.

## Building and Training
- In contrast to RNNs, the entire sequence is processed at once using positional encoding. The encoder part of the transformer processes the input sequence using self-attention and residual connections. The output of the top encoder is then transformed into a set of attention vectors K and V. These are to be used by each decoder in its encoder-decoder attention layer, which helps the decoder focus on appropriate places in the input sequence.
- The decoder component processes the input sequence similar by also using positional encoding, residual connections and self-attention. However, it masks future positions of the sequence in self-attention and uses mthe Query matrix from the masked self-attention layer nad the Key and Value matrix form the encoder, rather than generating its own. The decoder then makes a prediction about the sentiment of the tweet (for more information, check out the Concepts.md file).
- I experimented with some hyperparameters (I was limited by my computational resources, however) and got the best predictions using Cosine Annealing as learning rate scheduler and ADAM as optimizer for the model. (For more information, check out the Concepts.md file in Image-Classification repository)
- Note that I tried many different approaches and hyperparameters so I always saved the model with the best result. This is way I loaded a model from a file. The actual parameters are placed in a Markdown section above.

## Evaluation
- At first, I wanted to see how training was doing and if the loss was steadily decreasing. Look at the following:

  ![Loss](https://user-images.githubusercontent.com/127037803/227533128-23b6bb14-41b4-48d9-a91f-9c2dd929589c.png)
- The plot looks fine to me, the loss is decreasing and cosine annealing had a good impact on the performance, since about every 6 epochs, the learning rate was adjusted
- For evaluation, I ran through the test dataset and stored the predictions. These predictions where then used to for various evaluation metrics. The results are all in the mid 80% range, leaving room for improvement. For more information on what these scores convey, check out the Concepts.md file in the Fraud-Detection repository.

## Conclusion
- This was my favorite project so far since it really helped me understand the transformer architecture. Hyperparameter tuning was a fun, yet very challenging, exercise. There were so many parameters to tune, such as number of heads in attention, number of layers, neurons, etc.
- However, I really reached my limit here when it came to computational resources. I spent many many hours on training different models, failing and learning at the same time. I don't think that, for an individual, it is recommended to train Transformer model from scratch for practical reasons. It is an amazing way to understand the architecture behind the model, but if you are lacking the data and GPU power, it can be very frustrating.
- This is also the reason why I chose 100 characters as threshold since I could ran the model with different parameters to save time and may even achieve better results as I was able to try out many different things. It is great that many research teams release pre-trained models and I think individuals like me should focus on fine-tuning them rather than training them from scratch.
