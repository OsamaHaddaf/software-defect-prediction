# ARTI 406 - Software Defect Prediction

This is a machine learning course project that predicts whether a software module is defective or non-defective using the KC1 software defect dataset.

The project compares two models:

- Support Vector Machine (SVM)
- Random Forest

## Dataset

The dataset file should be placed here:

```text
data/raw/kc1.csv
```

The target column is:

```text
defects
```

Labels are converted to:

- `1` = defective
- `0` = non-defective

## Method

The dataset is split into:

- 80% training data
- 20% testing data

GridSearchCV is used to tune both models on the training data. The final results are calculated using the 20% test data.

## Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVM Tuned | 0.675 | 0.400 | 0.508 | 0.448 | 0.624 |
| Random Forest Tuned | 0.724 | 0.466 | 0.429 | 0.446 | 0.618 |

SVM had better recall, so it detected more defective modules. Random Forest had better accuracy and precision, so it made fewer false alarms.

## How to Run

Install the required packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the training script:

```powershell
.\.venv\Scripts\python.exe src\train_models.py
```

## Output Files

The script creates:

```text
reports/model_comparison.csv
reports/best_grid_search_parameters.csv
reports/random_forest_feature_importance.csv
figures/
```

These files can be used in the final report and presentation.
