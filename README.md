# ARTI 406 Software Defect Prediction

Machine learning course project for predicting defective software modules using KC1 software metrics.

## Project Topic

Software Defect Prediction using:

- Support Vector Machine (SVM)
- Random Forest

## Folder Structure

- `data/raw/` - put the original KC1 CSV dataset here
- `data/processed/` - cleaned or transformed datasets
- `notebooks/` - Jupyter notebooks for experiments
- `src/` - reusable Python code
- `figures/` - exported plots such as confusion matrices and ROC curves
- `reports/` - final report assets and result tables

## VS Code Setup

Open this folder in VS Code:

```powershell
code "C:\Users\osama\Documents\Codex\2026-05-19\files-mentioned-by-the-user-arti"
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Register the notebook kernel:

```powershell
python -m ipykernel install --user --name arti406-sdp --display-name "ARTI406 SDP"
```

## Dataset

Download or copy the KC1 dataset CSV into:

```text
data/raw/kc1.csv
```

The target column should be named `defects`. If the dataset uses `true/false`, the code converts it to `1/0`.

## First Run

After adding `data/raw/kc1.csv`, run:

```powershell
python src/train_models.py
```

The script will train SVM and Random Forest baselines and print the evaluation metrics.
# software-defect-prediction
