import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

df = pd.read_csv("dataset/IMDB Dataset.csv")
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['review'])

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f, protocol=4)
