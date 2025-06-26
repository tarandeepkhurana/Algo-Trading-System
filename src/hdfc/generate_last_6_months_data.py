import os
import logging
import pandas as pd
import ta
import numpy as np

#Ensures logs directory exists
log_dir = 'logs/hdfc' 
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('generate_last_6_months_data')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'generate_last_6_months_data.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def generate_features():
    """
    Generates last 6 months transformed data for HDFC stocks.
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

        # Drop rows with NaNs due to indicators or shifting
        df.dropna(inplace=True)
        df_clean = df[~df.isin([np.inf, -np.inf]).any(axis=1)]

        final_features = [
            'Date', 'Close', 'RSI', 'MACD', 'MACD_Signal', 'MA10', 'MA20', 'MA50',
            'Volume', 'Volume_Change_Pct', 'Daily_Return_Pct', 
            'BB_Width', 'Price_Momentum', 'MA_Diff'
        ]

        final_df = df_clean[final_features].tail(126)

        file_path = "data/last_6_months_data/hdfc"
        os.makedirs(file_path, exist_ok=True)
        final_df.to_csv(os.path.join(file_path, "hdfc_features_6_months.csv"), index=False)
        logger.debug("Last 6 months transformed data generated successfully.")

    except Exception as e:
        logger.error("Error occurred while generating last 6 months features: %s", e)
        raise

if __name__ == "__main__":
    generate_features()