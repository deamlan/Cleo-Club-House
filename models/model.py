# required imports
import pandas as pd
import string
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# import numpy as np
# import re
# import nltk
#from sklearn.linear_model import LinearRegression
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import mean_squared_error
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import accuracy_score
#import matplotlib.pyplot as plt
#from sklearn.naive_bayes import GaussianNB
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC

def train_lr_model():
    data = pd.read_csv("assert.csv")
    print("Succesfully parsed csv!")

    clean_data = []
    for i in range(0, data.shape[0]):
        line = data['Questions'].iloc[i]
        line = line.lower()
        translator = str.maketrans('', '', string.punctuation)
        line = line.translate(translator)
        line = " ".join(line.split())
        clean_data.append(line)
        print(line)
    
    print("Successfully cleaned data! \n Starting embedder")
    embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
    clean_data_embeddings = embedder.encode(clean_data)
    print("Here is the embedded data:\n", clean_data_embeddings)

    print("started shaping clean data...")
    print((clean_data_embeddings).shape)
    print((clean_data_embeddings).ndim)

    df = pd.DataFrame(clean_data_embeddings)

    df['cluster']=data['cluster']
    print("df.head() :\n", df.head())

    df = df.sample(frac=1, random_state=42)   
    X = df.drop('cluster', axis=1)
    Y = df['cluster']

    # Split the dataset into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Split data into training and testing sets
    print("Started spliting training and test data ...")
    train_data, test_data, train_labels, test_labels = train_test_split(X,Y, test_size=0.1, random_state=42)
    print("train data: \n", train_data)

    # Train a Linear Regression model
    print("Training LR model...")
    lr = LogisticRegression(max_iter=1200,random_state=0)
    lr.fit(train_data, train_labels)

    # Predict target values on the test set
    y_pred = lr.predict(test_data)
    print((test_data).shape)
    
    # return model variable
    return lr