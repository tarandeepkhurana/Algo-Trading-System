import yfinance as yf
import logging
import os

#Ensures logs directory exists
log_dir = 'logs/reliance' 
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_fetcher')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'data_fetcher.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def fetch_data() -> None:
    """
    It downloads the Reliance stock data from yfinance library.
    """
    try:
        data_dir = 'data/raw/reliance' 
        os.makedirs(data_dir, exist_ok=True)

        df = yf.download("RELIANCE.NS", start="2022-01-01", end=None)

        df.reset_index(inplace=True) 
        
        file_path = os.path.join(data_dir, 'stock_data.csv')
        df.to_csv(file_path, index=False)  
        logger.debug("Data loaded properly to stock_data.csv")
    except Exception as e:
        logger.error("Error occurred while fetching the data: %s", e)
        raise
    
    
if __name__ == "__main__":
    fetch_data()