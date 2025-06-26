import mlflow.sklearn
import pandas as pd
from src.hdfc.trade_logger import (process_trades_hdfc, 
                            log_summary_pnl_hdfc, log_win_ratio_hdfc)
from utils.sheet_utils import connect_all_sheets, clear_and_initialize_sheet
import os
import logging
from dotenv import load_dotenv
from utils.telegram_alert import send_telegram_alert

load_dotenv()

bot_token = os.getenv("BOT_TOKEN")
chat_id = os.getenv("CHAT_ID")

# Ensure the "logs" directory exists
log_dir = 'logs/hdfc'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger
logger = logging.getLogger('predict_last_6_months')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'predict_last_6_months.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def predict():
    """
    - Loads the model
    - Loads the last 6 months transformed data
    - Sends the action signal for HDFC stocks
    - Logs the Trade Log sheet, Summary PnL sheet and Win Ratio sheet
    """
    try:
        #Loading the best model
        model = mlflow.sklearn.load_model(
        r"C:\Users\taran\Documents\Algo_Trading_System\Algo-Trading-System\mlartifacts\664492794023519560\models\m-1021983c258449d3965917a55f164d8a\artifacts"
        )
        logger.debug("Model loaded successfully.")

        # Load your 6-month dataframe with stock prices and features
        df = pd.read_csv("data/last_6_months_data/hdfc/hdfc_features_6_months.csv")
        logger.debug("Last 6 months data loaded successfully.")
        
        last_row = df.iloc[[-1]]
        price = round(last_row['Close'].values[0], 2)
        date = last_row['Date'].values[0]

        # Extract only the feature columns used for training
        features = last_row[['RSI', 'MACD', 'MACD_Signal', 'MA10', 'MA20', 'MA50', 
                        'Volume', 'Volume_Change_Pct', 'Daily_Return_Pct', 
                        'BB_Width', 'Price_Momentum', 'MA_Diff']]
        
        # Sending Telegram alert about model's prediction
        pred_class = model.predict(features)[0]
        if pred_class == 1:
            action = "Buy"
        elif pred_class == 0:
            action = "Sell"
        
        if action in ["Buy", "Sell"]:
            send_telegram_alert(
                f"📢 {action} Signal for HDFC at ₹{price} on {date}",
                bot_token=bot_token,
                chat_id=chat_id
            )
        else:
            send_telegram_alert(
                "❌ Error: Model loading failed", 
                bot_token=bot_token, 
                chat_id=chat_id
            )
        logger.debug("ChatBot alert sent for today's action")
        
        # Connect all 3 sheets (tabs)
        sheets = connect_all_sheets("Algo_Trade_Log")
        logger.debug("Connected to all sheet tabs.")
        
        # Reset Trade Log Sheet for HDFC
        clear_and_initialize_sheet(sheets['trade_log_hdfc'])
        logger.debug("Trade Log sheet reset for HDFC.")

        # Run the trade logging
        process_trades_hdfc(model, df, sheets["trade_log_hdfc"])
        logger.debug("Trade Log sheet updated.")

        # Log Summary PnL
        log_summary_pnl_hdfc(sheets["trade_log_hdfc"], sheets["summary_pnl"])
        logger.debug("Summary PnL sheet updated.")
        
        #Log Win Ratio
        log_win_ratio_hdfc(sheets["trade_log_hdfc"], sheets["win_ratio"])
        logger.debug("Win Ratio sheet updated.")

    except Exception as e:
        logger.error("Error occurred while making predictions: %s", e)
        raise

if __name__ == "__main__":
    predict()