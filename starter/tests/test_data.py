import os
import sys

import pytest
import pandas as pd

from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

print(os.environ.get('data_cleaned_filepath'))

try:
    from starter.const import cat_features, data_cleaned_filepath
    from starter.ml.data import process_data
except ModuleNotFoundError:
    sys.path.append('./')
    from starter.const import cat_features, data_cleaned_filepath
    from starter.ml.data import process_data


# declare fixture
@pytest.fixture()
def data():
    df = pd.read_csv(data_cleaned_filepath)
    return train_test_split(df, test_size=0.20)


def test_process_data(data):
    train_dataset, test_dataset = data
    X_train, y_train, encoder, lb = process_data(
        X=train_dataset,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert len(X_train) > 0 and len(X_train) == len(y_train)
