from os import environ

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

stored_model_filepath = environ.get('MODEL_FILEPATH',
                                    "../model/model.pkl")
metrics_output_filepath = environ.get('METRICS_FILEPATH',
                                      "../data/metrics.csv")
data_cleaned_filepath = environ.get('DATA_CLEANED_FILEPATH',
                                    "../data/census_cleaned.csv")
