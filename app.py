from flask import Flask, render_template


app = Flask(__name__)


# index route
@app.route('/')
def index():
    return 'Work in progress'


# dummy api to test UI
@app.route('/api/v1.0/get-dummy-answer/<question>')
def get_dummy_answer(question):
    ans = '# To be edited #'
    return render_template('dummy-api.html', question=question, ans=ans)


# driver function
if __name__ == '__main__':
    app.run()
