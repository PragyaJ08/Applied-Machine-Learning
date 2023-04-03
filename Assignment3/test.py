import score
import joblib
import os
import requests
import numpy
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Defining input values to test the score function on
msg = "i like cats love them"
threshold = 0.7

os.chdir('./Assignment3/')
train=pd.read_csv(r"./Data/Training Data.csv")
val=pd.read_csv(r"./Data/Validation Data.csv")
test=pd.read_csv(r"./Data/Test Data.csv")

# Defining input values to test the score function on
spam = "FreeMsg Today's the day if you are ready! I'm horny & live in your town. I love sex fun & games! Netcollex Ltd 08700621170150p per msg reply Stop to end"
ham = "Easy mate, * guess the quick drink was bit ambitious."

# Importing Model
filename = open("svcmodel.joblib",'rb')
svc =joblib.load(filename)

os.chdir('../')


# Defining Unit Tests

# check if score function returns values properly
def test_smoke(text=msg,model=svc,threshold=threshold):
    label,propensity=score.score(text,model,threshold)
    assert label!= None
    assert propensity!= None

# check if the type of data meets certain requirements
def test_format(text=msg,model=svc,threshold=threshold):
    label,propensity = score.score(text,model,threshold)
    assert type(text) == str
    assert type(threshold) == float
    assert type(label) == int
    assert type(propensity) == float 

# Checking whether propensity value between 0 or 1
def test_predict(text=msg,model=svc,threshold=threshold):
    label,propensity=score.score(text,model,threshold)
    assert label == False or label == True

# check if the label value is in {0,1}
def test_propensity(text=msg,model=svc,threshold=threshold):
    label,propensity=score.score(text,model,threshold)
    assert propensity>=0 and propensity<=1

# if threshold is 0, prediction becomes 1
def test_0_thresh(text=msg,model=svc,threshold=0):
    label,propensity=score.score(text,model,threshold)
    assert label == 1

# if threshold is 1, prediction becomes 0
def test_1_thresh(text=msg,model=svc,threshold=1):
    label,propensity=score.score(text,model,threshold)
    assert label == 0

# testing obvious spam
def test_spam(text=spam,model=svc,threshold=threshold):
    label,propensity=score.score(text,model,threshold)
    assert label == 1

# testing obvious ham
def test_ham(text=ham,model=svc,threshold=threshold):
    label,propensity=score.score(ham,model,threshold)
    assert label == 0


def test_flask():
    # Launch the Flask app using os.system
    os.system('start /b python Assignment3/app.py &')

    # Wait for the app to start up
    time.sleep(1)

    # Make a request to the endpoint
    response = requests.get('http://127.0.0.1:5000/')
    print(response.status_code)

    # Assert that the response is what we expect
    assert response.status_code == 200
    assert type(response.text) == str

    # Shut down the Flask app using os.system
    os.system('kill $(lsof -t -i:5000)')