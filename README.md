# Sentiment Analysis with RNN on IMDb Dataset  
🔗 Live App: [https://sentiment-analysis-rnngit-xesrdfyvncj5aky3r4yrl8.streamlit.app/](https://sentiment-analysis-rnngit-xesrdfyvncj5aky3r4yrl8.streamlit.app/)

## Project Overview:

This project is a sentiment analysis application built using a Simple Recurrent Neural Network (RNN) implemented in PyTorch. It classifies IMDb movie reviews as either **positive** or **negative**.

The workflow consists of:
- Loading and preprocessing a dataset of 50,000 movie reviews.
- Tokenizing and padding the text data using TensorFlow's tokenizer.
- Training an RNN model to learn patterns in text sequences.
- Saving the trained model and tokenizer.
- Creating an interactive web app using **Streamlit** to make real-time predictions on user-entered reviews.

The app allows anyone to input a movie review and instantly get a sentiment prediction.

---

## Project Structure:

- `app.py` – Streamlit web app for real-time predictions.
- `train.py` – Script for training the RNN model and saving it.
- `predict.py` – Command-line script for sentiment prediction.
- `save_tokenizer.py` – Tokenizer creation and saving.
- `model.pth` – Trained PyTorch model weights.
- `tokenizer.pkl` – Saved tokenizer used in preprocessing.
- `requirements.txt` – Python dependencies.
- `runtime.txt` – Python version for deployment.
- `dataset/IMDB Dataset.csv` – IMDb movie review dataset.

---

## Dataset:

- **Name**: IMDb Dataset of 50K Movie Reviews  
- **Source**: [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- Contains 50,000 reviews labeled as `positive` or `negative`.

---

## Output:

![alt text](image.png)