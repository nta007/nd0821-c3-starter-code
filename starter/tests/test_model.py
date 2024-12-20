import sys

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

load_dotenv()
try:
    from starter.const import cat_features, stored_model_filepath, data_cleaned_filepath
    from starter.ml.data import process_data
    from starter.ml.model import load_model, inference, compute_model_metrics, train_model
except ModuleNotFoundError:
    sys.path.append('./')
    from starter.const import cat_features, stored_model_filepath, data_cleaned_filepath
    from starter.ml.data import process_data
    from starter.ml.model import load_model, inference, compute_model_metrics, train_model


# declare fixture
@pytest.fixture()
def data():
    df = pd.read_csv(data_cleaned_filepath)
    return train_test_split(df, test_size=0.20)


@pytest.fixture()
def trained_model():
    encoder, lb, trained_model = load_model(stored_model_filepath)
    return trained_model


def test_train_model(data):
    train_dataset, test_dataset = data
    X_train, y_train, encoder, lb = process_data(
        X=train_dataset,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    trained = train_model(X_train, y_train)
    assert isinstance(trained, RandomForestClassifier)


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


def test_compute_model_metrics(data, trained_model):
    train_dataset, test_dataset = data

    X_train, y_train, encoder, lb = process_data(
        X=train_dataset,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    trained = train_model(X_train, y_train)

    predicts = inference(trained, X_train)

    prec, rec, f1 = compute_model_metrics(y_train, predicts)

    assert prec is not None and rec is not None and f1 is not None
