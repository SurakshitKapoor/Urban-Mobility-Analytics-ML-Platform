

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    raw_data_path : str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        logging.info("entered into the data ingestion method")

        try:
            # loading the data
            BASE_URL = "https://raw.githubusercontent.com/SurakshitKapoor/Urban-Mobility-Analytics-ML-Platform/main/csv_files/"
            fact_trips = pd.read_csv(BASE_URL + "fact_trips.csv")
            dim_city = pd.read_csv(BASE_URL + "dim_city.csv")
            dim_date = pd.read_csv(BASE_URL + "dim_date.csv")

            # print("fact_trips info: ", fact_trips.head())
            logging.info("Data is loaded successfully!")

            # merging the data
            fact_df = fact_trips.copy()
            fact_df = fact_df.merge(dim_city, on="city_id", how="left")
            fact_df = fact_df.merge(dim_date[['date', 'month_name', 'day_type']], 
                                    on = "date", how = "left"    )

            print("fact_df shape: ", fact_df.shape)
            logging.info("tables are merged successfully!")


            # create artifacts folder 
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # raw data save
            fact_df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            # split data into training and testing part
            train_set, test_set = train_test_split(
                fact_df, test_size=0.2, random_state=42 )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Train-test split completed")

            # returning paths of train and test data for cleaning and transformations
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e, sys)
        


if __name__ == "__main__":
    print("starting app from data_ingestion.py")
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print("Train:", train_path)
    print("Test:", test_path)