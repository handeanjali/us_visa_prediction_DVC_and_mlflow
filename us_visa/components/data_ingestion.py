import os
import sys
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.data_access.usvisa_data import USVisaData


class DataIngestion:
    """
    Handles the data ingestion process, including:
    - Exporting data from MongoDB to a CSV file
    - Splitting data into training and testing sets
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        Initialize DataIngestion with the provided configuration.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info("DataIngestion initialized with provided configuration.")
        except Exception as e:
            raise USvisaException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Exports data from MongoDB and saves it as a CSV file.
        """
        try:
            logging.info("Starting data export from MongoDB.")
            usvisa_data = USVisaData()
            dataframe = usvisa_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            logging.info(f"Data successfully exported. Shape: {dataframe.shape}")

            # Ensure the directory exists before saving the file
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)

            logging.info(f"Saving exported data to: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            
            return dataframe
        except Exception as e:
            raise USvisaException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Splits the dataframe into training and testing sets based on the configured ratio.
        """
        try:
            logging.info("Starting train-test split.")
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info(f"Data split completed. Train shape: {train_set.shape}, Test shape: {test_set.shape}")

            # Ensure the directory exists before saving files
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            
            logging.info("Saving train and test datasets.")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Train and test datasets successfully saved.")
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process: exporting data and splitting it into train/test sets.
        """
        try:
            logging.info("Initiating data ingestion process.")
            dataframe = self.export_data_into_feature_store()
            logging.info("Data successfully retrieved from MongoDB.")

            self.split_data_as_train_test(dataframe)
            logging.info("Train-test split successfully performed.")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logging.info(f"Data ingestion completed successfully. Artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact
        except Exception as e:
            raise USvisaException(e, sys)
