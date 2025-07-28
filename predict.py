import torch
import torch.nn as nn
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

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

model = SentimentRNN(vocab_size=10000, embed_dim=128, hidden_dim=64)
model.load_state_dict(torch.load("model.pth"))
model.eval()

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=200)
    input_tensor = torch.LongTensor(padded)
    with torch.no_grad():
        output = model(input_tensor)
    prediction = output.item()
    return "Positive" if prediction >= 0.5 else "Negative"

# Example usage:
review = input("Enter a movie review: ")
print("Sentiment:", predict_sentiment(review))
