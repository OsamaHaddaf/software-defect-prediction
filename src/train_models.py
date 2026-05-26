from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "kc1.csv"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "figures"
TARGET_COLUMN = "defects"
RANDOM_STATE = 42
TUNING_METRIC = "f1"


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

    y = df[TARGET_COLUMN]
    if y.dtype == bool:
        y = y.astype(int)
    else:
        y = (
            y.astype(str)
            .str.lower()
            .map({"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0})
            .fillna(y)
            .astype(int)
        )

    X = df.drop(columns=[TARGET_COLUMN]).select_dtypes(include=["number"])
    return X, y


def evaluate_on_test_set(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def save_confusion_matrix(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Non-defective", "Defective"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"{name} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=200)
    plt.close(fig)


def save_roc_curve(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"{name} ROC Curve")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name.lower().replace(' ', '_')}_roc_curve.png", dpi=200)
    plt.close(fig)


def save_feature_importance(model: RandomForestClassifier, columns: pd.Index) -> None:
    importance = pd.DataFrame(
        {
            "feature": columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance.to_csv(REPORTS_DIR / "random_forest_feature_importance.csv", index=False)

    top_features = importance.head(10).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.set_title("Top 10 Random Forest Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "random_forest_feature_importance.png", dpi=200)
    plt.close(fig)


def build_grid_searches() -> dict[str, GridSearchCV]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    svm_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(probability=True, random_state=RANDOM_STATE),
            ),
        ]
    )

    svm_grid = {
        "model__C": [0.1, 1, 10, 100],
        "model__kernel": ["linear", "rbf", "poly"],
        "model__gamma": ["scale", "auto"],
        "model__class_weight": [None, "balanced"],
    }

    rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "class_weight": [None, "balanced"],
    }

    return {
        "SVM Tuned": GridSearchCV(
            estimator=svm_pipeline,
            param_grid=svm_grid,
            scoring=TUNING_METRIC,
            cv=cv,
            n_jobs=-1,
            verbose=1,
        ),
        "Random Forest Tuned": GridSearchCV(
            estimator=rf_model,
            param_grid=rf_grid,
            scoring=TUNING_METRIC,
            cv=cv,
            n_jobs=-1,
            verbose=1,
        ),
    }


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    X, y = load_dataset()
    print(f"Dataset shape: {X.shape[0]} rows, {X.shape[1]} features")
    print("Full dataset class distribution:")
    print(y.value_counts().sort_index())
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Training set: {X_train.shape[0]} rows ({X_train.shape[0] / X.shape[0]:.0%})")
    print(f"Testing set:  {X_test.shape[0]} rows ({X_test.shape[0] / X.shape[0]:.0%})")
    print("Training class distribution:")
    print(y_train.value_counts().sort_index())
    print("Testing class distribution:")
    print(y_test.value_counts().sort_index())
    print()

    print(f"Grid search tuning metric: {TUNING_METRIC}")
    print("Grid search uses only the 80% training set.")
    print("The 20% test set is used only for the final comparison.\n")

    rows = []
    tuning_rows = []
    fitted_models = {}
    models = build_grid_searches()
    for name, model in models.items():
        print(f"Running grid search for {name}...")
        start = perf_counter()
        model.fit(X_train, y_train)
        elapsed = perf_counter() - start
        best_model = model.best_estimator_
        print(f"Finished {name} in {elapsed:.2f} seconds")
        print(f"Best CV {TUNING_METRIC}: {model.best_score_:.3f}")
        print(f"Best parameters: {model.best_params_}")

        fitted_models[name] = best_model
        tuning_rows.append(
            {
                "model": name,
                "best_cv_f1": model.best_score_,
                "best_parameters": model.best_params_,
            }
        )
        rows.append(evaluate_on_test_set(name, best_model, X_test, y_test))
        save_confusion_matrix(name, best_model, X_test, y_test)
        save_roc_curve(name, best_model, X_test, y_test)
        print(f"Saved {name} confusion matrix and ROC curve")

    results = pd.DataFrame(rows)
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(REPORTS_DIR / "best_grid_search_parameters.csv", index=False)

    rf_model = fitted_models["Random Forest Tuned"]
    save_feature_importance(rf_model, X.columns)

    print("\nFinal 20% test-set results:")
    print(results.round(3).to_string(index=False))
    print("\nBest grid-search parameters:")
    print(pd.DataFrame(tuning_rows).to_string(index=False))
    print(f"\nSaved report tables to: {REPORTS_DIR}")
    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
