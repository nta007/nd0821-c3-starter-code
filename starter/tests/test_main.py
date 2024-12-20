from dotenv import load_dotenv

load_dotenv()
import sys

from fastapi.testclient import TestClient

try:
    from main import app
except ImportError:
    sys.path.append('./')
    from main import app

client = TestClient(app)


def test_predict():
    data = {
        "workclass": "Private",
        "education": "Bachelors",
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native-country": "United-States",
        "age": 42,
        "fnlgt": 159449,
        "education-num": 13,
        "capital-gain": 5178,
        "capital-loss": 0,
        "hours-per-week": 40
    }
    r = client.post("/predict", json=data)

    assert r.status_code == 200
    assert r.json()['prediction'] == 1


def test_hello_world():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Hello World!"}
