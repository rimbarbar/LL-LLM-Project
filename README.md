# LLM Project

## Project Task
In this project, I fine-tuned a pre-trained NLP model to perform sentiment analysis on IMDb movie reviews. The goal here was to classify movie reviews as either positive or negative and recieve accurate predictions.

## Dataset
I used the IMDb dataset available on Hugging Face. It consists of 50,000 movie reviews labeled as positive or negative, with a balanced class distribution.

## Pre-trained Model
I used bert-base-uncased, a transformer-based language model from the BERT family. I chose it for its robust contextual understanding and strong performance in text classification tasks.

## Performance Metrics
The performance metrics I evaluated were:
- Accuracy: to evaluate overall correctness of predictions.
- Precision, Recall, and F1-score: to provide deeper insight into model performance, especially for handling class imbalances.
- Loss: to measure model optimization progress during training.

## Hyperparameters
The key hyperparameters tuned include:
- Batch size: 8 (optimized for training efficiency)
- Learning rate: 5e-5 (to balance convergence and stability)
- Epochs: 3 (to prevent overfitting)
## Relevant Links
- Model on Hugging Face: https://huggingface.co/rimbarbar/imdb-sentiment-analysis/tree/main
- IMDb Dataset: https://huggingface.co/datasets/stanfordnlp/imdb 
