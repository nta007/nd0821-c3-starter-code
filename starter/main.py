import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from starter.ml import model
from starter.ml.data import process_data
from starter.const import cat_features, stored_model_filepath
from starter.ml.model import load_model


class PredictData(BaseModel):
    workclass: str
    education: str
    marital_status: str = Field(..., alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str = Field(..., alias='native-country')
    age: int
    fnlgt: int
    education_num: int = Field(..., alias='education-num')
    capital_gain: int = Field(..., alias='capital-gain')
    capital_loss: int = Field(..., alias='capital-loss')
    hours_per_week: int = Field(..., alias='hours-per-week')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_dataframe(self):
        data = self.dict(by_alias=True)
        return pd.DataFrame(data, index=[0])

    class Config:
        schema_extra = {
            "example": {
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
            }}


encoder, lb, trained_model = load_model(stored_model_filepath)
app = FastAPI(version="1.0.1", title="Udacity DevOps ML API",
              description="API for ML model", debug=True)


@app.get('/')
def hello_world():
    return {"message": "Hello World!"}


@app.post('/predict')
async def predict(pred: PredictData):
    data = pred.to_dataframe()

    X, _, _, _ = process_data(X=data,
                              categorical_features=cat_features,
                              label=None,
                              training=False,
                              encoder=encoder,
                              lb=lb)
    # compute prediction
    try:
        prediction = model.inference(trained_model, X)
        rs = int(prediction[0])
        return {"message": "success", "prediction": rs}
    except Exception as e:
        return {"message": "error", "error": str(e)}


# if __name__ == "__main__":
#     uvicorn.run('main:app',
#                 host="0.0.0.0",
#                 port=8000,
#                 reload=True,
#                 env_file='.env'
#                 )
