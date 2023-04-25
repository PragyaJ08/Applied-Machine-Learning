import score
import joblib, os, json
import requests, time
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

os.chdir(os.path.abspath(os.path.dirname(__file__)))

# Defining input values to test the score function on
msg = "i like cats love them"
threshold = 0.7

train=pd.read_csv(r"./Data/Training Data.csv")
val=pd.read_csv(r"./Data/Validation Data.csv")
test=pd.read_csv(r"./Data/Test Data.csv")

# Defining input values to test the score function on
spam = "FreeMsg Today's the day if you are ready! I'm horny & live in your town. I love sex fun & games! Netcollex Ltd 08700621170150p per msg reply Stop to end"
ham = "Easy mate, * guess the quick drink was bit ambitious."

# Importing Model
filename = open("svcmodel.joblib",'rb')
svc =joblib.load(filename)


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
    os.system('start /b python Assignment4/app.py &')
    
    response = requests.get('http://127.0.0.1:5000/')
    print(response.status_code)

    assert response.status_code == 200
    assert type(response.text) == str

    json_reply = requests.post('http://127.0.0.1:5000/score', {"sent": ham})

    assert json_reply.status_code == 200
    assert type(json_reply.text) == str

    load_json = json.loads(json_reply.text)

    assert type(load_json["Sentence"]) == str
    assert load_json["Prediction"] == "Spam" or load_json["Prediction"] == "not a spam"

    prop_score = float(load_json["Propensity"])
    assert prop_score >= 0 and prop_score <= 1

    os.system('kill $(lsof -t -i:5000)')

def test_docker():
    os.system('docker build --network=host -t image_spamClassification .')

    os.system('docker run --shm-size=1G -p 5000:5000 --name spam-flask-app -it -d image_spamClassification')

    time.sleep(10)
    
    response = requests.get('http://127.0.0.1:5000/')
    print(response.status_code)

    assert response.status_code == 200
    assert type(response.text) == str

    json_response = requests.post('http://127.0.0.1:5000/score', {"sent": ham})

    assert json_response.status_code == 200
    assert type(json_response.text) == str

    load_json = json.loads(json_response.text)

    assert type(load_json["Sentence"]) == str

    assert load_json["Prediction"] == "Spam" or load_json["Prediction"] == "not a spam"

    prop_score = float(load_json["Propensity"])
    assert prop_score >= 0 and prop_score <= 1