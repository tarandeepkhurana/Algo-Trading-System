import logging
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Ensure the "logs" directory exists
log_dir = 'logs/reliance'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('data_preprocess')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocess.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def split_dataset() -> None:
    """
    Splits the dataset into train and test.
    Scale the features using StandardScaler.
    """
    try:
        file_path = "data/feature_engineered/reliance/transformed_stock_data.csv"
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from: %s", file_path)

        X = df.drop(columns='Target')
        y = df['Target']

        train_size = int(len(df) * 0.8)

        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        X_train.columns = X_train.columns.str.strip()
        X_test.columns = X_test.columns.str.strip()

        X_test = X_test.replace([np.inf, -np.inf], np.nan) #Removing the one inf value from X_test
        X_test = X_test.dropna()

        y_test = y_test.loc[X_test.index]  #Aligning y_test accordingly

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert back to DataFrames
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        train_dir = "data/train/reliance"
        test_dir = "data/test/reliance"
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        X_train_scaled_df.to_csv(os.path.join(train_dir, "X_train.csv"), index=False)
        y_train.to_csv(os.path.join(train_dir, "y_train.csv"), index=False)
        X_test_scaled_df.to_csv(os.path.join(test_dir, "X_test.csv"), index=False)
        y_test.to_csv(os.path.join(test_dir, "y_test.csv"), index=False)

        logger.debug("Data splitted and scaled successfully.")
    except Exception as e:
        logger.error("Error occurred while splitting the dataset: %s", e)
        raise

if __name__ == "__main__":
    split_dataset()