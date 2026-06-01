# Credit Card Fraud Detection App

This is a complete Python Machine Learning app for detecting credit card fraud.

## Features

- Train ML model
- Single transaction prediction
- Bulk CSV prediction
- Fraud probability score
- Model performance dashboard
- Confusion matrix
- Download prediction report

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier
- Streamlit
- Joblib
- Matplotlib

## Dataset

Download the Credit Card Fraud Detection dataset from Kaggle and place the file in this folder:

```bash
creditcard.csv
```

The dataset columns should be:

```text
Time, V1, V2, ..., V28, Amount, Class
```

Where:

```text
Class 0 = Genuine transaction
Class 1 = Fraud transaction
```

## Run Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train model

```bash
python train_model.py
```

This will create:

```text
fraud_model.pkl
scaler.pkl
model_metrics.pkl
```

### 3. Start app

```bash
streamlit run app.py
```

## Viva Explanation

This project uses supervised machine learning to classify credit card transactions as genuine or fraudulent.
Since the dataset is highly imbalanced, the model uses class balancing and is evaluated using precision,
recall, F1-score, ROC-AUC, and confusion matrix instead of relying only on accuracy.
# ml_project
