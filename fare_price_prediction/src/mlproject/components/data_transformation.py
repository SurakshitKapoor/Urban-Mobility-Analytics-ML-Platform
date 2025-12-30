

# import os
# import sys
# import pandas as pd
# from src.mlproject.logger import logging
# from src.mlproject.exception import CustomException
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from src.mlproject.utils import save_object

# class DataTransformationConfig:
#     preprocessor_path : str = os.path.join("artifacts", "preprocessor.pkl")


# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()


#     # method for data cleaning and transformation in a particular way
#     def clean_and_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#             Performs column renaming, datetime handling,
#             feature extraction, and drops leakage / ID columns.
#         """

#         # Rename columns for clarity
#         df = df.rename(columns={
#             'distance_travelled(km)': 'distance_km',
#             'fare_amount': 'fare',
#             'month_name': 'month',
#             'day_type': 'day_category'
#         })

#         # Convert date column
#         df['date'] = pd.to_datetime(df['date'], dayfirst=True)


#         # Date-based features
#         df['day'] = df['date'].dt.day
#         df['weekday'] = df['date'].dt.weekday
#         df['week'] = df['date'].dt.isocalendar().week.astype(int)
#         df['month_num'] = df['date'].dt.month


#         # Drop ID / leakage / non-ML columns
#         df = df.drop(
#             columns=[
#                 'trip_id',
#                 'city_id',
#                 'date',
#                 'month',
#                 'passenger_rating',
#                 'driver_rating'
#             ],
#             errors='ignore'
#         )

#         return df



#     def get_preprocessor(self, X: pd.DataFrame):
#         """
#             Creates and returns a ColumnTransformer
#             for numeric and categorical features.
#         """
#         # drop the target variable
#         X = X.drop(columns=['fare'])
        
#         # Identify feature types
#         numeric_features = X.select_dtypes(include=['int64', 'int32', 'float64']).columns.tolist()
#         categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

#         logging.info(f"Numeric features: {numeric_features}")
#         logging.info(f"Categorical features: {categorical_features}")


#         # Transformers
#         numeric_transformer = StandardScaler()

#         categorical_transformer = OneHotEncoder(
#             drop='first',
#             sparse_output=False,
#             handle_unknown='ignore'
#         )


#         # Column transformer
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ("num", numeric_transformer, numeric_features),
#                 ("cat", categorical_transformer, categorical_features)
#             ]
#         )

#         return preprocessor


#     def apply_preprocessor_and_get_df(self, X: pd.DataFrame, preprocessor):

#         """
#             Fits the preprocessor, transforms X,
#             and returns a DataFrame with proper column names.
#         """
#         # Fit & transform
#         X_prepared = preprocessor.fit_transform(X)

#         # Get column names
#         numeric_features = preprocessor.transformers_[0][2]
#         categorical_features = preprocessor.transformers_[1][2]

#         cat_columns = preprocessor.named_transformers_['cat'] \
#         .get_feature_names_out(categorical_features)

#         all_columns = list(numeric_features) + list(cat_columns)

#         # Convert to DataFrame
#         X_prepared_df = pd.DataFrame(X_prepared, columns=all_columns)

#         return X_prepared_df
    



#     def initiate_data_transformation(self, train_path, test_path):
#         try:
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)

#             logging.info("starting data cleaning and transformation!")

#             train_df = self.clean_and_engineer_features(train_df)
#             test_df = self.clean_and_engineer_features(test_df)

#             logging.info("done with data cleaning and transformation")

#             print("train_df: \n", train_df.head())


#             train_df_preprocessor = self.get_preprocessor(train_df)
#             test_df_preprocessor = self.get_preprocessor(test_df)
#             logging.info("preprocessing completed!")

#             print("train_df_preprocessor: ", train_df_preprocessor)

#             train_df_prepared = self.apply_preprocessor_and_get_df(train_df, train_df_preprocessor)
#             test_df_prepared = self.apply_preprocessor_and_get_df(test_df, test_df_preprocessor)

#             save_object(
#                 file_path=self.data_transformation_config,
#                 obj=preprocessor_obj
#             )


#         except Exception as e:
#             raise CustomException(e, sys)
        


# if __name__ == "__main__":
#     print("starting app from data_transformation.py")

#     # TEMP test paths (use real artifact paths)
#     train_path = "artifacts/train.csv"
#     test_path = "artifacts/test.csv"

#     obj = DataTransformation()
#     obj.initiate_data_transformation(train_path, test_path)



import os
import sys
import pandas as pd

from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.utils import save_object

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    # -----------------------------
    # 1. CLEAN & FEATURE ENGINEERING
    # -----------------------------
    def clean_and_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs column renaming, datetime handling,
        feature extraction, and removes leakage columns.
        """

        # rename columns
        df = df.rename(columns={
            'distance_travelled(km)': 'distance_km',
            'fare_amount': 'fare',
            'month_name': 'month',
            'day_type': 'day_category'
        })

        # Date handling
        df['date'] = pd.to_datetime(df['date'], dayfirst=True , errors='coerce')

        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['week'] = df['date'].dt.isocalendar().week.astype(int)
        df['month_num'] = df['date'].dt.month

        # Drop leakage / ID columns
        df.drop(
            columns=[
                'trip_id',
                'city_id',
                'date',
                'month',
                'passenger_rating',
                'driver_rating'
            ],
            errors='ignore',
            inplace=True
        )

        return df


    # -----------------------------
    # 2. PREPROCESSOR CREATION
    # -----------------------------
    def get_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Creates preprocessing pipeline for numeric and categorical features.
        """

        numeric_features = X.select_dtypes(include=['int64', 'int32', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

        logging.info(f"Numeric features: {numeric_features}")
        logging.info(f"Categorical features: {categorical_features}")

        numeric_transformer = StandardScaler()

        categorical_transformer = OneHotEncoder(
            drop='first',
            sparse_output=False,
            handle_unknown='ignore'
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )

        return preprocessor

    # -----------------------------
    # 3. MAIN TRANSFORMATION PIPE
    # -----------------------------
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Starting data cleaning & feature engineering")

            # data cleaning and feature engg.
            train_df = self.clean_and_engineer_features(train_df)
            test_df = self.clean_and_engineer_features(test_df)

            # Split features & target
            X_train = train_df.drop(columns=['fare'])
            y_train = train_df['fare']

            X_test = test_df.drop(columns=['fare'])
            y_test = test_df['fare']


            # Create preprocessor ONLY on training data
            preprocessor = self.get_preprocessor(X_train)

            # fit preprocessor on train and test
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)


            # Save preprocessor
            save_object(
                file_path=self.config.preprocessor_path,
                obj=preprocessor
            )

            logging.info("Preprocessor saved successfully")

            return (
                X_train_arr,
                X_test_arr,
                y_train.values,
                y_test.values
            )

        except Exception as e:
            raise CustomException(e, sys)


# -----------------------------
# 4️⃣ LOCAL TEST
# -----------------------------
if __name__ == "__main__":
    print("Running data_transformation.py")

    train_path = "artifacts/train.csv"
    test_path = "artifacts/test.csv"

    transformer = DataTransformation()
    X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(
        train_path,
        test_path
    )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("X_train : ", type(X_train))
