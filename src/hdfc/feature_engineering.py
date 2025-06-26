import pandas as pd
import logging
import os
import ta
import numpy as np

#Ensures logs directory exists
log_dir = 'logs/hdfc' 
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def new_features() -> None:
    """
    Create new features for training the ml models.
    """
    try:
        file_path = "data/raw/hdfc/stock_data.csv"
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from: %s", file_path)
        
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
        df["High"] = pd.to_numeric(df["High"], errors="coerce")
        df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        close_series = df['Close'].squeeze()
        volume = df['Volume'].squeeze()

        # Feature: RSI
        df['RSI'] = ta.momentum.RSIIndicator(close=close_series).rsi()

        # Feature: MACD and Signal line
        macd = ta.trend.MACD(close=close_series)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()

        # Moving Averages
        df['MA10'] = close_series.rolling(window=10).mean()
        df['MA20'] = close_series.rolling(window=20).mean()
        df['MA50'] = close_series.rolling(window=50).mean()

        # Volume Change %
        df['Volume_Change_Pct'] = volume.pct_change()

        # Daily Return %
        df['Daily_Return_Pct'] = close_series.pct_change()

        # Bollinger Band Width
        bb = ta.volatility.BollingerBands(close=close_series)
        df['BB_Width'] = bb.bollinger_hband() - bb.bollinger_lband()
        
        #Price Momentum
        df['Price_Momentum'] = df['Close'] - df['MA20']
        
        df['MA_Diff'] = df['MA20'] - df['MA50']

        # First: convert 'Close' column to a plain NumPy array
        close_values = df['Close'].values.flatten()

        # Create Next_Close using NumPy (this avoids all Pandas index issues)
        next_close_values = np.roll(close_values, -1)

        # Now calculate the target
        target_values = (next_close_values > close_values).astype(int)

        # Set the last target to NaN or drop it later (since it compares to garbage)
        target_values = (next_close_values > close_values).astype(float) 
        target_values[-1] = np.nan

        # Assign back to DataFrame
        df['Target'] = target_values

        # Drop rows with NaNs due to indicators or shifting
        df.dropna(inplace=True)

        df['Target'] = df['Target'].astype(int)

        # Select only final columns to keep
        final_features = [
            'RSI', 'MACD', 'MACD_Signal', 'MA10', 'MA20', 'MA50',
            'Volume', 'Volume_Change_Pct', 'Daily_Return_Pct', 'BB_Width',
             'Price_Momentum', 'MA_Diff', 'Target'
        ]

        final_df = df[final_features]

        data_dir = "data/feature_engineered/hdfc"
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, "transformed_stock_data.csv")
        final_df.to_csv(file_path, index=False)
        logger.debug("Data loaded properly to transformed_stock_data.csv")
    except Exception as e:
        logger.error("Error occurred while creating new features: %s", e)
        raise

if __name__ == "__main__":
    new_features()