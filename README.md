# IMDb Sentiment Analysis with Fine-Tuned BERT

## Project Overview
In this project, I fine-tuned a pre-trained BERT model (`bert-base-uncased`) to perform sentiment analysis on IMDb movie reviews. My goal was to classify reviews as positive or negative with high accuracy using natural language processing techniques. I structured the workflow into five stages: data preprocessing, embedding extraction, model training, hyperparameter optimization, and deployment.

## Dataset
- Source: [IMDb Dataset on Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb)
- Description: I used a dataset of 50,000 movie reviews, evenly split between positive and negative sentiments (25,000 each).
- Subset Used: To keep things efficient, I worked with a balanced subset of 500 training and 500 test samples.
- Format: I stored the tokenized dataset as `./tokenized_imdb` and the embeddings as `./embeddings_imdb` in my project directory.

## Pre-trained Model
- Model: `bert-base-uncased` from the BERT family (Hugging Face Transformers)
- Why: I chose this model for its strong contextual understanding and proven performance in text classification tasks, thanks to its bidirectional architecture.
- Fine-Tuned Output: I saved my fine-tuned model locally as `./fine_tuned_bert` and uploaded it to [Hugging Face Hub](https://huggingface.co/rimbarbar/imdb-sentiment-analysis).

## Project Structure
I organized everything into a five key jupyter notebooks, making it easy for others to follow and collaborate on:
1. Preprocessing: I loaded and tokenized the IMDb dataset, reducing it to 500 train + 500 test samples.
2. Representation: I extracted BERT embeddings for each review using the CLS token (768-dimensional vectors).
3. Pre-trained Model: I fine-tuned `bert-base-uncased` for binary classification (positive/negative).
4. Optimization: I used Bayesian optimization to tune hyperparameters and boost performance.
5. Deployment: I tested the model with sample reviews and pushed it to Hugging Face Hub.

## Performance Metrics
I evaluated my model’s performance with these metrics:
- Accuracy: I measured overall prediction correctness (e.g., I achieved 84.8% in the final evaluation).
- Precision: I checked the reliability of positive predictions (e.g., 86.3%).
- Recall: I assessed how well I captured the positive class (e.g., 82.1%).
- F1-Score: I balanced precision and recall (e.g., 84.2%).
- Loss: I tracked training progress (e.g., validation loss dropped to 0.379 by epoch 3).

Detailed results per epoch are in the outputs of `LL_LLM_Project_Resub_PretrainedModel.ipynb` and 'LL_LLM_Project_Resub_Optimization.ipynb'

## Hyperparameters
I tuned these key hyperparameters using Bayesian optimization:
- Batch Size: I settled on 8 (rounded from 8.51, balancing memory and speed).
- Learning Rate: I optimized it to 4.16e-5 for smooth convergence.
- Weight Decay: I set it to 0.0777 to control regularization.
- Epochs: I used 3 to avoid overfitting on my small dataset.

For my initial training, I started with a batch size of 8, a learning rate of 5e-5, and a weight decay of 0.01.

## Relevant Links
- My Model on Hugging Face: [rimbarbar/imdb-sentiment-analysis](https://huggingface.co/rimbarbar/imdb-sentiment-analysis)
- IMDb Dataset: [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)
  
