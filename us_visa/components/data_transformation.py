import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from us_visa.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from us_visa.entity.estimator import TargetValueMapping


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        Initializes the DataTransformation class with required artifacts and schema configuration.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file and returns a pandas DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a preprocessing pipeline for numerical and categorical features.
        """
        try:
            logging.info("Initializing data transformation pipeline.")
            
            # Define individual transformations
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()
            transform_pipe = Pipeline(steps=[('power_transformer', PowerTransformer(method='yeo-johnson'))])
            
            # Extract feature categories from schema config
            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']
            
            # Create a column transformer with different feature transformations
            preprocessor = ColumnTransformer([
                ("OneHotEncoder", oh_transformer, oh_columns),
                ("OrdinalEncoder", ordinal_encoder, or_columns),
                ("PowerTransformer", transform_pipe, transform_columns),
                ("StandardScaler", numeric_transformer, num_features)
            ])
            
            logging.info("Preprocessing pipeline created successfully.")
            return preprocessor
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Executes data transformation, including preprocessing, feature engineering, and handling class imbalance.
        """
        try:
            # Ensure validation has passed before proceeding
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            logging.info("Starting data transformation process.")
            preprocessor = self.get_data_transformer_object()
            
            # Load train and test datasets
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
            
            drop_cols = self._schema_config['drop_columns']
            
            def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
                """Adds derived features and removes unnecessary columns."""
                df['company_age'] = CURRENT_YEAR - df['yr_of_estab']
                return drop_columns(df, drop_cols)
            
            # Split input features and target variable
            input_feature_train_df = preprocess_data(train_df.drop(columns=[TARGET_COLUMN]))
            target_feature_train_df = train_df[TARGET_COLUMN].replace(TargetValueMapping()._asdict())
            
            input_feature_test_df = preprocess_data(test_df.drop(columns=[TARGET_COLUMN]))
            target_feature_test_df = test_df[TARGET_COLUMN].replace(TargetValueMapping()._asdict())
            
            logging.info("Applying preprocessing transformation.")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            # Handle class imbalance using SMOTEENN
            logging.info("Applying SMOTEENN to balance dataset.")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(input_feature_train_arr, target_feature_train_df)
            input_feature_test_final, target_feature_test_final = smt.fit_resample(input_feature_test_arr, target_feature_test_df)
            
            # Combine features and target into a single array
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            
            # Save transformed data and preprocessing object
            logging.info("Saving transformed data and preprocessing object.")
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            
            logging.info("Data transformation completed successfully.")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            raise USvisaException(e, sys)
