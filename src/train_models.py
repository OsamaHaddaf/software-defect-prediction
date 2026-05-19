from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "kc1.csv"
TARGET_COLUMN = "defects"


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Put the KC1 CSV file at data/raw/kc1.csv, then run again."
        )

    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}'. Found columns: {list(df.columns)}"
        )

    df = df.drop_duplicates()
    y = df[TARGET_COLUMN]
    if y.dtype == bool:
        y = y.astype(int)
    else:
        y = y.astype(str).str.lower().map(
            {
                "true": 1,
                "false": 0,
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0,
            }
        ).fillna(y)
        y = y.astype(int)

    X = df.drop(columns=[TARGET_COLUMN])
    X = X.select_dtypes(include=["number"])
    return X, y


def evaluate_model(name: str, model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "model": name,
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "f1": scores["test_f1"].mean(),
        "roc_auc": scores["test_roc_auc"].mean(),
    }


def main() -> None:
    X, y = load_dataset()
    print(f"Dataset shape: {X.shape[0]} rows, {X.shape[1]} features")
    print("Class distribution:")
    print(y.value_counts().sort_index())
    print()

    models = [
        (
            "SVM RBF",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SVC(
                            kernel="rbf",
                            class_weight="balanced",
                            probability=True,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    results = [evaluate_model(name, model, X, y) for name, model in models]
    results_df = pd.DataFrame(results)
    print(results_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
