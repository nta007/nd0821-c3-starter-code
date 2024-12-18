# Model Card
Udacity devops ml model card - study only

## Model Details
- Model is trained by using RandomForestClassifier, use to predict salary of people.

## Intended Use
- This model just for learning, so do not use in production, it's not recommend.

## Training Data
- The data is a set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0)). More information here: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
- Evaluation data is a part of data set, I've split into multi part, used for training, test,..

## Metrics

```
Precision: 0.7391640866873065
Recall: 0.6074097630783909
F1: 0.6668412324343196
```

## Ethical Considerations
- Data comes from public source, so it does not contains information for real life.

## Caveats and Recommendations
- If use for production or somethings important, I recommend think careful about train data.