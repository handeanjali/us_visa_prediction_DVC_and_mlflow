import json
import sys
import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

from pandas import DataFrame

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.constants import SCHEMA_FILE_PATH


class DataValidation:
    """
    Handles data validation tasks, including:
    - Checking column integrity
    - Detecting data drift
    - Generating validation reports
    """

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        Initializes DataValidation with necessary artifacts and configuration.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            logging.info("DataValidation initialized successfully.")
        except Exception as e:
            raise USvisaException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Validates if the dataframe has the expected number of columns.
        """
        try:
            expected_columns = len(self._schema_config["columns"])
            actual_columns = len(dataframe.columns)
            status = actual_columns == expected_columns
            logging.info(f"Expected columns: {expected_columns}, Actual columns: {actual_columns}, Validation status: {status}")
            return status
        except Exception as e:
            raise USvisaException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Validates if all expected numerical and categorical columns exist in the dataframe.
        """
        try:
            dataframe_columns = df.columns
            missing_columns = {
                "numerical": [col for col in self._schema_config["numerical_columns"] if col not in dataframe_columns],
                "categorical": [col for col in self._schema_config["categorical_columns"] if col not in dataframe_columns],
            }
            
            if missing_columns["numerical"]:
                logging.warning(f"Missing numerical columns: {missing_columns['numerical']}")
            if missing_columns["categorical"]:
                logging.warning(f"Missing categorical columns: {missing_columns['categorical']}")
            
            return not any(missing_columns.values())
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> DataFrame:
        """
        Reads a CSV file into a pandas DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Detects dataset drift between reference and current datasets.
        """
        try:
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])
            data_drift_profile.calculate(reference_df, current_df)
            
            json_report = json.loads(data_drift_profile.json())
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=json_report)
            
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            logging.info(f"Dataset drift detected: {drift_status}")
            return drift_status
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiates the data validation process, including column checks and drift detection.
        """
        try:
            logging.info("Starting data validation process.")
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
            
            validation_errors = []
            if not self.validate_number_of_columns(train_df):
                validation_errors.append("Missing columns in training dataframe.")
            if not self.validate_number_of_columns(test_df):
                validation_errors.append("Missing columns in testing dataframe.")
            if not self.is_column_exist(train_df):
                validation_errors.append("Missing required columns in training dataframe.")
            if not self.is_column_exist(test_df):
                validation_errors.append("Missing required columns in testing dataframe.")
            
            validation_status = len(validation_errors) == 0
            validation_message = "Data validation successful." if validation_status else "; ".join(validation_errors)
            
            if validation_status:
                drift_detected = self.detect_dataset_drift(train_df, test_df)
                validation_message = "Drift detected." if drift_detected else "No drift detected."
                logging.info(validation_message)
            else:
                logging.warning(f"Validation issues: {validation_message}")
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_message,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise USvisaException(e, sys)
