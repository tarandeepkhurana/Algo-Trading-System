import mlflow
import mlflow.sklearn
import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# Setup logging
log_dir = 'logs/reliance'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('lightgbm_classifier')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'lightgbm_classifier.log'))
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def tune_model():
    X_train = pd.read_csv("data/train/reliance/X_train.csv")
    y_train = pd.read_csv("data/train/reliance/y_train.csv").squeeze()
    X_test = pd.read_csv("data/test/reliance/X_test.csv")
    y_test = pd.read_csv("data/test/reliance/y_test.csv").squeeze()
    logger.debug("Data loaded successfully.")
    
    # Load param grid from params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    param_grid = params["lightgbm"]

    tscv = TimeSeriesSplit(n_splits=5)

    mlflow.set_experiment("LightGBMClassifier")

    with mlflow.start_run():

        model = LGBMClassifier(random_state=42)
        grid = GridSearchCV(model, param_grid, cv=tscv, scoring='accuracy')
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        best_threshold = thresholds[np.argmax(tpr - fpr)]

        # Logging
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc_score", auc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("test_data_length", len(y_test))
        mlflow.log_text(str(list(X_train.columns)), "features_used.txt")

        # Classification report HTML
        report_df = pd.DataFrame(report).transpose()
        report_df.to_html("artifacts/classification_report.html")
        mlflow.log_artifact("artifacts/classification_report.html")

        # Confusion matrix as image
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("artifacts/confusion_matrix.png")
        mlflow.log_artifact("artifacts/confusion_matrix.png")

        # ROC Curve plot
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("artifacts/roc_curve.png")
        mlflow.log_artifact("artifacts/roc_curve.png")

        mlflow.sklearn.log_model(best_model, "best_lightgbm_model")
        
        model_path = "models/reliance/"
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(best_model, os.path.join(model_path, "lightgbm.pkl"))
        logger.debug("Model saved successfully.")

if __name__ == "__main__":
    tune_model()
