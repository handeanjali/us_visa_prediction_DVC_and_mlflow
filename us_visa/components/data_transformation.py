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
        Initialize the DataTransformation class with necessary artifacts and configurations.
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
        Read CSV data from the given file path.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Create and return a data transformation pipeline.
        """
        try:
            logging.info("Creating data transformation pipeline")

            # Load column details from schema
            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            # Define individual transformers
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()
            transform_pipe = Pipeline(steps=[('transformer', PowerTransformer(method='yeo-johnson'))])

            # Combine transformers using ColumnTransformer
            preprocessor = ColumnTransformer([
                ("OneHotEncoder", oh_transformer, oh_columns),
                ("Ordinal_Encoder", ordinal_encoder, or_columns),
                ("Transformer", transform_pipe, transform_columns),
                ("StandardScaler", numeric_transformer, num_features)
            ])

            logging.info("Data transformation pipeline created successfully")
            return preprocessor
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Execute the data transformation steps.
        """
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            logging.info("Starting data transformation process")

            # Load datasets
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Separate input features and target
            input_feature_train_df, target_feature_train_df = train_df.drop(columns=[TARGET_COLUMN]), train_df[TARGET_COLUMN]
            input_feature_test_df, target_feature_test_df = test_df.drop(columns=[TARGET_COLUMN]), test_df[TARGET_COLUMN]

            # Feature Engineering - Adding 'company_age'
            input_feature_train_df['company_age'] = CURRENT_YEAR - input_feature_train_df['yr_of_estab']
            input_feature_test_df['company_age'] = CURRENT_YEAR - input_feature_test_df['yr_of_estab']

            # Drop unnecessary columns
            drop_cols = self._schema_config['drop_columns']
            input_feature_train_df = drop_columns(input_feature_train_df, drop_cols)
            input_feature_test_df = drop_columns(input_feature_test_df, drop_cols)

            # Encode target labels
            target_feature_train_df = target_feature_train_df.replace(TargetValueMapping()._asdict())
            target_feature_test_df = target_feature_test_df.replace(TargetValueMapping()._asdict())

            # Get transformer object and apply transformations
            preprocessor = self.get_data_transformer_object()
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Apply SMOTEENN for handling imbalances
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(input_feature_train_arr, target_feature_train_df)
            input_feature_test_final, target_feature_test_final = smt.fit_resample(input_feature_test_arr, target_feature_test_df)

            # Combine input and target arrays
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

            # Save transformed objects
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            logging.info("Data transformation completed successfully")

            # Create and return transformation artifact
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            raise USvisaException(e, sys)
