from flask import Flask, render_template
import models.model as model
import string
from sentence_transformers import SentenceTransformer


app = Flask(__name__)

lr = model.train_lr_model()

# index route
@app.route('/')
def index():
    return 'Work in progress'


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
    
    return cluster

# driver function
if __name__ == '__main__':
    app.run(debug=True)
