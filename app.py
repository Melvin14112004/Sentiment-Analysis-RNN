import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer  # ✅ use this, not keras.utils
from tensorflow.keras.preprocessing.sequence import pad_sequences  # ✅ fix import source

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
