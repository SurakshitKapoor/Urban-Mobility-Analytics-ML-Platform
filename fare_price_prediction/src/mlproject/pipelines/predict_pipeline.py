
import os
import sys
import pandas as pd

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import load_object


class PredictPipeline:
    """
    Handles loading preprocessor + model
    and generating fare predictions
    """

    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        # self.model_path = os.path.join("artifacts", "model.pkl")

        self.model_path = os.path.join("artifacts", "model_dev.pkl")   # testing
        # self.model_path = os.path.join("artifacts", "model_prod.pkl")  # full data

    
    
    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Starting prediction pipeline")

            # Load artifacts
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            logging.info("Artifacts loaded successfully")

            # Transform features
            X_transformed = preprocessor.transform(features)

            # Predict
            predictions = model.predict(X_transformed)

            logging.info("Prediction completed")

            return predictions

        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    """
    Captures user input for fare prediction
    """

    def __init__(
        self,
        passenger_type: str,
        distance_km: float,
        city_name: str,
        day_category: str,
        day: int,
        weekday: int,
        week: int,
        month_num: int
    ):
        self.passenger_type = passenger_type
        self.distance_km = distance_km
        self.city_name = city_name
        self.day_category = day_category
        self.day = day
        self.weekday = weekday
        self.week = week
        self.month_num = month_num

    def get_data_as_dataframe(self):
        """
        Returns input data as DataFrame
        in the SAME format as training features
        """
        return pd.DataFrame({
            "passenger_type": [self.passenger_type],
            "distance_km": [self.distance_km],
            "city_name": [self.city_name],
            "day_category": [self.day_category],
            "day": [self.day],
            "weekday": [self.weekday],
            "week": [self.week],
            "month_num": [self.month_num]
        })



# only for testing this pipeline locally with a sample data
if __name__ == "__main__":
    print("starting from predict_pipeline.py")

    sample_input = CustomData(
        passenger_type="new",
        distance_km=12.5,
        city_name="Jaipur",
        day_category="Weekday",
        day=15,
        weekday=2,
        week=20,
        month_num=5
    )

    df = sample_input.get_data_as_dataframe()

    predictor = PredictPipeline()
    fare = predictor.predict(df)

    print("Predicted Fare:", fare[0])
