import os
import pandas as pd
import mlflow
import joblib

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("FeatureImportances")

# Load training features
X_train = pd.read_csv("data/train/hdfc/X_train.csv")
features = X_train.columns.tolist()

# Model paths
model_dir = "models/hdfc"
model_paths = {
    "DecisionTree": os.path.join(model_dir, "decision_tree.pkl"),
    "RandomForest": os.path.join(model_dir, "random_forest.pkl"),
    "XGBoost": os.path.join(model_dir, "xgboost.pkl"),
    "LightGBM": os.path.join(model_dir, "lightgbm.pkl"),
}

def log_feature_importance_html_only():
    with mlflow.start_run(run_name="feature_importance_all_models"):

        for model_name, model_path in model_paths.items():

            model = joblib.load(model_path)

            # Build DataFrame
            importances = model.feature_importances_
            df = pd.DataFrame({
                "Feature": features,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            # Save and log as HTML
            html_path = f"artifacts/{model_name}_feature_importance.html"
            df.to_html(html_path, index=False)
            mlflow.log_artifact(html_path)

            print(f"✅ Logged feature importance for {model_name} as HTML")

if __name__ == "__main__":
    log_feature_importance_html_only()
