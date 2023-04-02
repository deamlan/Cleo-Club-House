import numpy as np
import pandas as pd
import nltk
import string
import re
from sentence_transformers import SentenceTransformer


data = pd.read_csv("assert.csv")
# print(data)

clean_data = []
for i in range(0, data.shape[0]):
    line = data['Questions'].iloc[i]
    line = line.lower()
    translator = str.maketrans('', '', string.punctuation)
    line = line.translate(translator)
    line = " ".join(line.split())
    clean_data.append(line)
    print(line)


embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
clean_data_embeddings = embedder.encode(clean_data)
print(clean_data_embeddings)