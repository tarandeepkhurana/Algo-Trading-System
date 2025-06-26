from src.hdfc.main_hdfc import main_hdfc
from src.reliance.main_reliance import main_reliance
import os
import logging

#Ensures logs directory exists
log_dir = 'logs/main' 
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('main')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, 'main.log')
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def main():
    """
    It runs the main function of all the stocks.
    """

    try:
        print("\n" + "="*70)
        print("**RUNNING RELIANCE FLOW**".center(70))
        print("="*70)
        main_reliance()

        print("\n" + "="*70)
        print("**RUNNING HDFC FLOW**".center(70))
        print("="*70)
        main_hdfc()

        logger.debug("Main function execution completed for all stocks.")

        print("\n" + "="*70)
        print("**PREDICTION & LOGGING COMPLETED SUCCESSFULLY**".center(70))
        print("="*70)

    except Exception as e:
        logger.error("Error occurred while running the main fn: %s", e)
        raise

if __name__ == "__main__":
    main()