
import os
import pickle
import sys
from src.mlproject.exception import CustomException

# function to save the object
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.mlproject.logger import logging




def evaluate_model(X_train, y_train, X_test, y_test, models: dict):
    """
    Trains and evaluates multiple regression models.

    Returns:
        dict: {model_name: RMSE}
    """

    model_report = {}

    for model_name, model in models.items():
        logging.info(f"Training model: {model_name}")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        logging.info(
            f"{model_name} | RMSE: {rmse:.2f} | MAE: {mae:.2f}"
        )

        model_report[model_name] = rmse

    return model_report
