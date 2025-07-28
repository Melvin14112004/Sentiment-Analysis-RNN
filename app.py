import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
from keras.utils import pad_sequences

# Define the same model architecture used during training
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        out = self.fc(h_n.squeeze(0))
        return self.sigmoid(out)

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = SentimentRNN(vocab_size=10000, embed_dim=128, hidden_dim=64)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Streamlit UI
st.set_page_config(page_title="IMDb Sentiment Analyzer", layout="centered")
st.title("🎬 IMDb Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to predict whether it's **Positive** or **Negative**.")

user_input = st.text_area("✍️ Your Review:")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review before submitting.")
    else:
        # Preprocess input
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=200)
        input_tensor = torch.LongTensor(padded)

        # Predict
        with torch.no_grad():
            prediction = model(input_tensor)
        sentiment = "🌟 Positive" if prediction.item() >= 0.5 else "☹️ Negative"
        st.success(f"**Sentiment:** {sentiment}")

