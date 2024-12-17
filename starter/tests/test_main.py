from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_predict():
    data = {
        "workclass": "Private",
        "education": "11th",
        "marital-status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own",
        "race": "White",
        "sex": "Female",
        "native-country": "United-States"
    }
    r = client.post("/predict", json=data)
    print(r.json())
    assert r.status_code == 200
    assert r.json() == {"message": "success"} and 'prediction' in r.json()


def test_hello_world():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Hello World!"}
