# Project Description
## Introduction
- The Transformer architecture introduced in the paper "Attention Is All You Need" was a major leap forwards in the development of more sophisticated Natural Language Models and it paved the way for popular models like GPT and BERT. As the time of this writing, major tech-companies like Microsoft, Google, OpenAI, Meta, etc. are announcing their own models that are based on the Transformer architecture.
- For this reason, I decided to take a deeper look into the architecture behind this revolutionary model and assessed sentiment analysis as a fitting project. The dataset consists of about 1.6 million tweets extracted using the Twitter API and is found on Kaggle using this URL: https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download
- Please note that the following information should be read along with the Python code in this repository and requires a more profound understanding of neural networks, the Attention mechanism. Each concept described in this documentation is explained in more detail in the Concept.md file. For more information on the cross-entropy loss, check out the Concept.md file in the Fraud-Detection repository.
## Preprocessing
- First, lets take a look at the distribution of length of each tweet:

  ![length](https://user-images.githubusercontent.com/127037803/227503757-7387def7-cf32-49ec-8dd8-ed964a9a945e.png)
- As seen in this plot, most tweets remain under the threshold of 150 characters, so I will leave out any sample that exceeds 150 characters. There is no harm in that, since the excluded samples are very few in numbers.
- The dataset seems to label positive sentiment as 0 and negative sentiment as 4. A better representation of the target variable is to use 0 and 1 instead. I also dropped any rows that reported missing values.
- Next, a vocabulary must be created to implement the embeddings of the tokens. Special tokens, such as for padding or start-of-sentence, are includes as weel. I also split the dataset randomly to create a training and test dataset.
- Since I use batched samples, all sentences must be padded to fit the length of the longest sentence since the Transformer expects the input to be in a unified dimension. These are all the steps required to prepare the dataset for the model.

## Building and Training
- In contrast to RNNs, the entire sequence is processed at once using positional encoding. The encoder part of the transformer processes the input sequence using self-attention and residual connections. The output of the top encoder is then transformed into a set of attention vectors K and V. These are to be used by each decoder in its encoder-decoder attention layer, which helps the decoder focus on appropriate places in the input sequence.
- The decoder component processes the input sequence similar by also using positional encoding, residual connections and self-attention. However, it masks future positions of the sequence in self-attention and uses mthe Query matrix from the masked self-attention layer nad the Key and Value matrix form the encoder, rather than generating its own. The decoder then makes a prediction about the sentiment of the tweet (for more information, check out the Concepts.md file).
- I experimented with some hyperparameters (I was limited by my computational resources, however) and got the best predictions using Cosine Annealing as learning rate scheduler and ADAM as optimizer for the model. (For more information, check out the Concepts.md file in Image-Classification repository)

## Evaluation
- For evaluation, I ran through the test dataset and stored the predictions. The predictions where then used to for various evaluation metrics. The 
