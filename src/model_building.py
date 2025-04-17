import os
import numpy as np
import pandas as pd
import pickle
import logging
import yaml
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

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

def load_data():
    """Load training and testing data"""
    try:
        x_train = pd.read_csv("data/raw/train.csv")
        x_test = pd.read_csv("data/raw/test.csv")
        y_train = x_train.pop('price').values.ravel()
        y_test = x_test.pop('price').values.ravel()
        logger.debug(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise

def train_and_evaluate(models_config, x_train, x_test, y_train, y_test):
    results = []

    models = {
        "Linear": {
            "Model": LinearRegression(),
            "params": {}
        },
        "KNN": {
            "Model": KNeighborsRegressor(),
            "params": models_config.get("KNN", {})
        },
        "Ridge": {
            "Model": Ridge(),
            "params": models_config.get("Ridge", {})
        },
        "Lasso": {
            "Model": Lasso(),
            "params": models_config.get("Lasso", {})
        },
        "RandomForest": {
            "Model": RandomForestRegressor(),
            "params": models_config.get("RandomForest", {})
        },
        "XGBoost": {
            "Model": XGBRegressor(),
            "params": models_config.get("XGBoost", {})
        }
    }

    for model_name, config in models.items():
        try:
            logger.info(f"Training model: {model_name}")
            gs = GridSearchCV(
                estimator=config["Model"],
                param_grid=config["params"],
                scoring='r2',
                cv=5,
                verbose=2,
                n_jobs=-1,
                return_train_score=True
            )

            gs.fit(x_train, y_train)
            best_model = gs.best_estimator_
            y_pred = best_model.predict(x_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            results.append({
                'Model': model_name,
                'Best Params': gs.best_params_,
                'Train R2': gs.best_score_,
                'Test R2': r2,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse
            })

            logger.info(f"Finished: {model_name} | Test R2 Score: {r2:.4f}")

            # Save model
            model_save_path = f"models/{model_name}_model.pkl"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            with open(model_save_path, 'wb') as f:
                pickle.dump(best_model, f)
            logger.debug(f"Model saved to {model_save_path}")

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")

    # Save comparison
    results_df = pd.DataFrame(results)
    results_df.to_csv("models/model_comparison.csv", index=False)
    logger.info("Model comparison saved to models/model_comparison.csv")

def main():
    try:
        params = load_params('params.yaml')
        models_config = params["model_training"]["models"]
        x_train, x_test, y_train, y_test = load_data()
        train_and_evaluate(models_config, x_train, x_test, y_train, y_test)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
