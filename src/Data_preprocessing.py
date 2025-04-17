import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
import yaml
import re

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    log_file = os.path.join(log_dir, 'data_preprocessing.log')
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise



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



def convert_range_to_sqft(x):
    try:
        x = x.strip()
        if '-' in x:
            tokens = x.split('-')
            if len(tokens) == 2:
                return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)

    except:
        match = re.match(r"([\d.]+)\s*(Sq\. Meter|Sq\. Yards|Sq\. Feet|Acres|Cents|Guntha|Grounds|Perch)", x)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            conversion_factors = {
                'Sq. Meter': 10.7639,
                'Sq. Yards': 9.0,
                'Sq. Feet': 1.0,
                'Acres': 43560,
                'Cents': 435.6,
                'Guntha': 1089,
                'Grounds': 2400,
                'Perch': 272.25
            }

            return value * conversion_factors.get(unit, np.nan)

        return np.nan



def remove_outliers(df):
    try:
        df_out = pd.DataFrame()
        for key, subdf in df.groupby("location"):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out
    except Exception as e:
        logger.error(f"Error removing outliers: {e}")
        raise


def remove_bhk_outliers(df):
    try:
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk_size'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_sqft),
                    'std': np.std(bhk_df.price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk_size'):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
        return df.drop(exclude_indices, axis='index')
    except Exception as e:
        logger.error(f"Error removing BHK outliers: {e}")
        raise


def preprocess_data(df):
    try:
        logger.info("Starting data preprocessing...")

        df = df.drop(["area_type", "society", "balcony", "availability"], axis=1)
        df = df.dropna()
        df["bhk_size"] = df["size"].apply(lambda x: int(x.split(" ")[0]))
        df['total_sqft'] = df['total_sqft'].apply(convert_range_to_sqft)

        df.copy()
        df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]
        df.location = df.location.apply(lambda x: x.strip())

        location_stat = df.groupby("location")["location"].agg("count").sort_values(ascending=False)
        loc_less_10 = location_stat[location_stat <= 10].index
        df.location = df.location.apply(lambda x: "other" if x in loc_less_10 else x)


        remove_outliers(df)

        remove_bhk_outliers(df)
        df[df.bath < df.bhk_size + 2]

        df = df.drop(["size", "price_per_sqft"], axis=1)
        dum = pd.get_dummies(df.location)
        df = pd.concat([df, dum.drop("other", axis=1)], axis=1)
        df = df.drop("location", axis=1)

        logger.info("Data preprocessing completed successfully.")
        return df

    
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def train_test_split_data(df):
    try:
        x = df.drop("price", axis=1)
        y = df["price"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
        logger.info("Train-test split successful.")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error in train_test_split_data: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try: 
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_path = 'data/raw/raw_data.csv'
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
