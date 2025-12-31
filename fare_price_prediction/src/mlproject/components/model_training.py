

import os
import sys
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.utils import evaluate_model, save_object


class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Starting model training")

            # ----------------------------
            # 1. Define models
            # ----------------------------
            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(alpha=1.0),
                "Lasso": Lasso(alpha=0.01),
                "RandomForest": RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                ),
                "GradientBoosting": GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42
                )
            }

            # ----------------------------
            # 2. Evaluate models
            # ----------------------------
            model_report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            logging.info(f"Model report: {model_report}")

            # ----------------------------
            # 3. Select best model
            # ----------------------------
            best_model_name = min(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best model before tuning: {best_model_name}")

            # ----------------------------
            # 4. GridSearch ONLY on best model
            # ----------------------------
            if best_model_name == "RandomForest":
                param_grid = {
                    "n_estimators": [100, 200],
                    "max_depth": [8, 10, None],
                    "min_samples_split": [5, 10],
                    "min_samples_leaf": [2, 5]
                }

            elif best_model_name == "GradientBoosting":
                param_grid = {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                }

            else:
                # Linear / Ridge / Lasso → no tuning
                best_model.fit(X_train, y_train)
                save_object(
                    file_path=self.model_trainer_config.trained_model_path,
                    obj=best_model
                )
                return best_model_name

            # GridSearch
            gs = GridSearchCV(
                best_model,
                param_grid,
                cv=3,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )

            gs.fit(X_train, y_train)
            final_model = gs.best_estimator_

            logging.info(f"Best params: {gs.best_params_}")

            # ----------------------------
            # 5. Save final model
            # ----------------------------
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=final_model
            )

            logging.info("Final model saved successfully")

            y_pred = best_model.predict(X_test)
            best_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            return best_model_name, best_rmse

        except Exception as e:
            raise CustomException(e, sys)




if __name__ == "__main__":
    from src.mlproject.components.data_transformation import DataTransformation

    print("Starting model training pipeline test...")

    train_path = "artifacts/train.csv"
    test_path = "artifacts/test.csv"

    # Data transformation
    transformer = DataTransformation()
    X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(
        train_path, test_path
    )

    # Model training
    trainer = ModelTrainer()
    best_model_name, best_model_score = trainer.initiate_model_trainer(
        X_train, X_test, y_train, y_test
    )

    print(f"Best Model: {best_model_name}")
    print(f"Best RMSE: {best_model_score}")
