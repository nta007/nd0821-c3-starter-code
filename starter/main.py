from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd
import pickle

from starter.ml import model
from starter.ml.data import process_data
from starter.const import cat_features


class PredictData(BaseModel):
    workclass: str
    education: str
    marital_status: str = Field(..., alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str = Field(..., alias='native-country')

    def to_dataframe(self):
        # print(self.dict(by_alias=True))
        return pd.DataFrame([self.dict(by_alias=True)])


def load_model(pkl_filepath):
    with open(pkl_filepath, 'rb') as model_file:
        return pickle.load(model_file)


encoder, lb, trained_model = load_model('./model/model.pkl')

app = FastAPI(version="0.1.0", title="Udacity DevOps ML API",
              description="API for ML model")


@app.get('/')
def hello_world():
    return {"message": "Hello World!"}


@app.post('/predict')
async def predict(pred: PredictData):
    data = pred.to_dataframe()
    X, _, _, _ = process_data(data,
                              categorical_features=cat_features,
                              label=None,
                              training=False,
                              encoder=encoder,
                              lb=lb)

    # compute prediction
    prediction = model.inference(trained_model, X)
    return {"message": "success", "prediction": prediction}


if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True)
