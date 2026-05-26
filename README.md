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

The current dataset has 2109 rows, so the split creates 1687 training rows and 422 testing rows.

GridSearchCV is used to tune both models on the training data. The final results are calculated using the 20% test data.

## Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| SVM Tuned | 0.716 | 0.313 | 0.708 | 0.434 | 0.790 |
| Random Forest Tuned | 0.841 | 0.488 | 0.600 | 0.538 | 0.822 |

SVM had higher recall, so it detected more defective modules. Random Forest had better accuracy, precision, F1-score, and ROC-AUC.

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
