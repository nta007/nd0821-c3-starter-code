import pandas as pd
import pytest

from starter.const import cat_features
from starter.ml.data import process_data
from sklearn.model_selection import train_test_split

from starter.ml.model import load_model, inference


# declare fixture
@pytest.fixture()
def data():
    df = pd.read_csv("../data/census_cleaned.csv")
    return train_test_split(df, test_size=0.20)


@pytest.fixture()
def trained_model():
    encoder, lb, trained_model = load_model('../model/model.pkl')
    return trained_model


def test_inference(data, trained_model):
    train_dataset, test_dataset = data

    X_train, y_train, encoder, lb = process_data(
        X=train_dataset,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    predicts = inference(trained_model, X_train)
    assert predicts is not None and len(predicts) == len(X_train)
