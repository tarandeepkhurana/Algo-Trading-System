from src.reliance.data_fetcher import fetch_data
from src.reliance.generate_last_6_months_data import generate_features
from src.reliance.predict_last_6_months import predict
import os
import logging

#Ensures logs directory exists
log_dir = 'logs/reliance' 
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('main_reliance')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'main_reliance.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def main_reliance():
    """
    Main function for RELIANCE stock which updates the results
    """
    try:
        #Fetch the updated data
        fetch_data()

        #Generate last 6 months transformed data
        generate_features()

        #Make predictions for the last 6 months data
        predict()

        logger.debug("Predictions for RELIANCE done.")
    
    except Exception as e:
        logger.error("Error occurred in main_reliance fn: %s", e)
        raise

if __name__ == "__main__":
    main_reliance()