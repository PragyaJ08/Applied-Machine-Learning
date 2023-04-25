from flask import Flask, request, render_template, url_for, redirect
import joblib, json
import score
import os

os.chdir(os.path.abspath(os.path.dirname(__file__)))

app = Flask(__name__, template_folder = './template')

filename = open("svcmodel.joblib",'rb')
svc =joblib.load(filename)

threshold=0.5

@app.route('/')
def home():
    return render_template('spam.html')

@app.route('/score', methods=['POST'])
def spam():
    sent = request.form['sent']
    label,prop=score.score(sent,svc,threshold)
    lbl="Spam" if label == 1 else "not a spam"
    dictToReturn = {'Sentence':sent,'Prediction':lbl, 'Propensity':prop}
    json_obj = json.dumps(dictToReturn) 
    return json_obj

if __name__ == '__main__': 
    app.run(host="0.0.0.0", port=5000, debug=True)