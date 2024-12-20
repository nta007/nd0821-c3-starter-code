# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split

from starter.const import cat_features, stored_model_filepath, \
    metrics_output_filepath, data_cleaned_filepath
from starter.ml import model
from starter.ml.data import process_data

# Add the necessary imports for the starter code.
import pickle

# Add code to load in the data.
data = pd.read_csv(data_cleaned_filepath)

# Optional enhancement, use K-fold cross validation instead of
# a train-test split.
train, test = train_test_split(data, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    X=train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
trained_model = model.train_model(X_train, y_train)

# save model with pickle
with open(stored_model_filepath, 'wb') as model_file:
    pickle.dump([encoder, lb, trained_model], model_file)

# trained model metrics
train_preds = model.inference(trained_model, X_train)
prec, rec, f1 = model.compute_model_metrics(y_train, train_preds)
print(f"Trained model metrics: Precision: {prec}, Recall: {rec}, F1: {f1}")


def compute_model_metrics(test_df: pd.DataFrame, stored_model) -> pd.DataFrame:
    metrics = []
    for feat in cat_features:
        for col in test_df[feat].unique():
            _df = test_df[test_df[feat] == col]
            X__test, y__test, _, _ = process_data(
                _df,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb
            )

            preds = model.inference(stored_model, X__test)
            precision, recall, fbeta = model.compute_model_metrics(
                y__test, preds)
            metric = {
                "feature": feat,
                "value": col,
                "precision": precision,
                "recall": recall,
                "f1": fbeta,
            }
            metrics.append(metric)

    metrics_df = pd.DataFrame(metrics)
    # save metrics to csv
    metrics_df.to_csv(metrics_output_filepath, index=False)
    return metrics_df


compute_model_metrics(test, trained_model)
