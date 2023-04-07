from flask import Flask, render_template
import models.model as model
import string
import pandas as pd
from sentence_transformers import SentenceTransformer
import random

answer_data=pd.read_csv("Book1.csv")

app = Flask(__name__)

lr = model.train_lr_model()

# index route
@app.route('/')
def index():
    return render_template('index.html')

# test api
@app.route('/test-ui')
def test_ui():
    return render_template('test-index.html')


# dummy api to test UI
@app.route('/api/v1.0/get-dummy-answer/<question>')
def get_dummy_answer(question):
    ans = '# To be edited #'
    return render_template('dummy-api.html', question=question, ans=ans)


# get_cluster API get the cluster label for the given question
@app.route('/api/v1.0/get-cluster/<question>')
def get_cluster(question):
    question = question.lower()
    translator = str.maketrans('','',string.punctuation)
    question = question.translate(translator)
    question = " ".join(question.split())
    embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
    question = embedder.encode(question)
    question2d = question.reshape(1, 768)

    cluster_arr = lr.predict(question2d)
    cluster = cluster_arr[0].astype('U')

    #choose the random question
    cluster1=cluster
    if cluster1==40:
        cluster1=cluster1-1
    else:
        cluster1=cluster1+1
    cluster_data = cluster_data[cluster_data.cluster==cluster1]
    cluster_question=cluster_data['Questions'].values
    random_index = random.randint(0, len(cluster_question)-1)

    # Select the random question
    random_question = cluster_question[random_index]

    #Select the answer
    answer=answer_data['answer'].iloc[cluster-1]
    return answer,random_question

# driver function
if __name__ == '__main__':
    app.run(debug=True)
