# Project Description
## Introduction
- The Transformer architecture, introduced in the paper "Attention Is All You Need", was a great leap forward in the development of more sophisticated natural language models and paved the way for popular models such as GPT and BERT. At the time of writing, major technology companies such as Microsoft, Google, OpenAI, Meta, etc. are announcing their own models based on the Transformer architecture.
- For this reason, I decided to take a deeper look into the architecture of this revolutionary model and I deemed sentiment analysis as a suitable project. The dataset consists of about 1.6 million tweets extracted via the Twitter API and can be found on Kaggle using the following URL: https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download
- Please note that the following information should be read in conjunction with the Python code in this repository and requires a deeper understanding of neural networks and the attention mechanism. Each concept described in this documentation is explained in more detail in the Concept.md file. For more information on cross-entropy loss, see the Concept.md file in the Fraud-Detection repository.
## Preprocessing
- Let's first take a look at the distribution of the length of each tweet:

  ![length](https://user-images.githubusercontent.com/127037803/227503757-7387def7-cf32-49ec-8dd8-ed964a9a945e.png)
- As seen in this graph, most tweets remain below the threshold of 150 characters, though I omit any samples that contain more than 100 characters. I felt this threshold was appropriate because it allowed me greater freedom to experiment with hyperparameters (see the Conclusions section for more information). However, it also limits the ability to generalize to longer sequences.
- The dataset appears to label positive sentiment as 0 and negative sentiment as 4. A better representation of the target variable would be to use 0 and 1 instead. I also deleted any rows that contained missing values.
- Next, a vocabulary needs to be created to implement the embedding of the tokens. Special tokens, such as the ones for padding or start-of-sentence, are also included. I then randomly split the dataset to create a training dataset and a test dataset.
- Since I am using batched samples, all records must be padded to match the length of the longest sequence. This is because the transformer expects the input to be in a uniform dimension. These are all the steps required to prepare the dataset for the model.

## Building and Training
- Unlike RNNs, the entire sequence is processed at once by using positional encoding. The encoder part of the transformer processes the input sequence using self-attention and residual connections. The output of the top encoder is then converted into a set of attention vectors K and V. These are used by each decoder in its encoder-decoder attention layer, which helps the decoder focus on appropriate tokens in the input sequence.
- The decoder component processes the input sequence similarly, also using position encoding, residual connections, and self-attention. However, it masks future positions of the sequence in the self-attention layer and uses the query matrix from the masked self-attention layer and the key and value matrix from the encoder instead of generating its own. The decoder then makes a prediction about the sentiment of the tweet (see the Concepts.md file for more information).
- I experimented with some hyperparameters (I was limited by my computational resources, though) and got the best predictions using Cosine Annealing as the learning rate scheduler and ADAM as the optimizer for the model. (For more information, see the Concepts.md file in the Image Classification repository.)
- Note that I tried many different approaches and hyperparameters, so I always saved the model with the best result. This is why I loaded in a model from a file. The actual parameters are located in a markdown section above.

## Evaluation
- First, I wanted to see how the training was going and if the loss was steadily decreasing. Take a look at the following:

  ![Loss](https://user-images.githubusercontent.com/127037803/227533128-23b6bb14-41b4-48d9-a91f-9c2dd929589c.png)
- The graph looks good to me, the loss is decreasing and Cosine Annealing had a good impact on performance as about every 6 epochs the learning rate was adjusted.
- For scoring, I ran the test dataset and saved the predictions. These predictions were then used for various evaluation metrics. The evaluation results are all in the mid-80% range and leave room for improvement. More information on the evaluation metrics can be found in the Concepts.md file in the Fraud-Detection repository.

## Conclusion
- This has been my favorite project so far because it helped me understand the architecture of the transformers. Tuning the hyperparameters was a fun but very challenging exercise. There were so many parameters to adjust, such as the number of heads at attention, the number of layers, the neurons, etc.
- However, this is where I really reached my limits in terms of computational resources. I spent many, many hours training different models, failing and learning at the same time. I don't think it's advisable for an individual to train a Transformer model from scratch for practical reasons. It is an amazing way to understand the architecture behind the model, but if you lack the data and GPU power, it can be very frustrating.
- That's also why I chose 100 characters as a threshold, because I could run the model with different parameters to save time and maybe even get better results since I was able to experiment with a lot of things. It's great that many research teams are publishing pre-trained models, and I think individuals like me should focus on fine-tuning these models rather than training them from scratch.
- As for performance, I think there is still room for improvement, and honestly I thought this model should perform better. Naturally, I compared my results to those posted on Kaggle, and my results seem to outperform most of the kernels on the forum. I'm sure that with more computational resources I can increase performance, but for now I'm satisfied with the results.
