
import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException

from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_training import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("===== TRAINING PIPELINE STARTED =====")

            # ----------------------------
            # 1. Data Ingestion
            # ----------------------------
            logging.info("Starting data ingestion")
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Train: {train_path}, Test: {test_path}")

            # ----------------------------
            # 2. Data Transformation
            # ----------------------------
            logging.info("Starting data transformation")
            transformer = DataTransformation()
            X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(
                train_path, test_path
            )
            logging.info("Data transformation completed")

            # ----------------------------
            # 3. Model Training
            # ----------------------------
            logging.info("Starting model training")
            trainer = ModelTrainer()
            best_model_name, best_model_score = trainer.initiate_model_trainer(
                X_train, X_test, y_train, y_test
            )

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best RMSE: {best_model_score}")

            logging.info("===== TRAINING PIPELINE COMPLETED SUCCESSFULLY =====")

        except Exception as e:
            logging.error("Training pipeline failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("starting from train_pipeline.py")
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
