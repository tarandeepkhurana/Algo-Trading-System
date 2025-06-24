import mlflow
import mlflow.sklearn
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# Setup logging
log_dir = 'logs/reliance'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('naive_bayes')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'naive_bayes.log'))
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def train_naive_bayes():
    # Load data
    X_train = pd.read_csv("data/train/reliance/X_train.csv")
    y_train = pd.read_csv("data/train/reliance/y_train.csv").squeeze()
    X_test = pd.read_csv("data/test/reliance/X_test.csv")
    y_test = pd.read_csv("data/test/reliance/y_test.csv").squeeze()
    logger.debug("Data loaded successfully.")
    
    mlflow.set_experiment("NaiveBayesClassifier")

    with mlflow.start_run():

        model = GaussianNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        best_threshold = thresholds[np.argmax(tpr - fpr)]

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc_score", auc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("test_data_length", len(y_test))
        mlflow.log_text(str(list(X_train.columns)), "features_used.txt")

        # Classification report as HTML
        pd.DataFrame(report).transpose().to_html("artifacts/classification_report.html")
        mlflow.log_artifact("artifacts/classification_report.html")

        # Confusion Matrix plot
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("artifacts/confusion_matrix.png")
        mlflow.log_artifact("artifacts/confusion_matrix.png")

        # ROC Curve
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

        # Save model
        model_dir = "models/reliance/"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "naive_bayes.pkl"))
        logger.debug("Model saved successfully.")
        mlflow.sklearn.log_model(model, "naive_bayes_model")

if __name__ == "__main__":
    train_naive_bayes()
