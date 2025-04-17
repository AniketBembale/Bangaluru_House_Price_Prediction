import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def save_data(df: pd.DataFrame, data_dir: str) -> None:
    """Save the raw dataset to the specified directory."""
    try:
        raw_data_path = os.path.join(data_dir, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        save_path = os.path.join(raw_data_path, 'raw_data.csv')
        df.to_csv(save_path, index=False)
        logger.debug('Raw data saved to %s', save_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        data_path = 'Bengaluru_House_Data.csv'
        save_dir = './data'
        df = load_data(data_url=data_path)
        save_data(df, data_dir=save_dir)
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()