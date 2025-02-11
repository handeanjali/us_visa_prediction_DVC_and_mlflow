import os
import sys
from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,

)
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)


class ConfigurationManager:
    def __init__(self):
        try:
            self.config_dir = "config"
            self.artifact_dir = "artifacts"

        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Returns the configuration for Data Ingestion
        """
        try:
            data_ingestion_config = DataIngestionConfig(
                training_file_path=os.path.join(self.artifact_dir, "data_ingestion", "train.csv"),
                testing_file_path=os.path.join(self.artifact_dir, "data_ingestion", "test.csv"),
            )
            return data_ingestion_config

        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_ingestion_artifact(self) -> DataIngestionArtifact:
        """
        Returns the artifact for Data Ingestion
        """
        try:
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=os.path.join(self.artifact_dir, "data_ingestion", "train.csv"),
                test_file_path=os.path.join(self.artifact_dir, "data_ingestion", "test.csv"),
            )
            return data_ingestion_artifact

        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Returns the configuration for Data Validation
        """
        try:
            data_validation_config = DataValidationConfig(
                drift_report_file_path=os.path.join(self.artifact_dir, "data_validation", "report.yaml")
            )
            return data_validation_config

        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_validation_artifact(self) -> DataValidationArtifact:
        """
        Returns the artifact for Data Validation
        """
        try:
            data_validation_artifact = DataValidationArtifact(
                                                        validation_status=True,
                                                        message="Data validation completed successfully.",
                                                        drift_report_file_path=os.path.join(self.artifact_dir, "data_validation", "report.yaml"))
            return data_validation_artifact

        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Returns the configuration for Data Transformation
        """
        try:
            data_transformation_config = DataTransformationConfig(
                            transformed_train_file_path=os.path.join(self.artifact_dir, "data_transformation", "transformed_train.npy"),
                            transformed_test_file_path=os.path.join(self.artifact_dir, "data_transformation", "transformed_test.npy"),
                            transformed_object_file_path=os.path.join(self.artifact_dir, "data_transformation", "preprocessor.pkl")
)            
            return data_transformation_config

        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformation_artifact(self) -> DataTransformationArtifact:
        """
        Returns the artifact for Data Transformation
        """
        try:
            data_transformation_artifact = DataTransformationArtifact(
                            transformed_train_file_path=os.path.join(self.artifact_dir, "data_transformation", "transformed_train.npy"),
                            transformed_test_file_path=os.path.join(self.artifact_dir, "data_transformation", "transformed_test.npy"),
                            transformed_object_file_path=os.path.join(self.artifact_dir, "data_transformation", "preprocessor.pkl")
)

            return data_transformation_artifact

        except Exception as e:
            raise USvisaException(e, sys)

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Returns the configuration for Model Training
        """
        try:
            model_trainer_config = ModelTrainerConfig(
                trained_model_file_path=os.path.join(self.artifact_dir, "model_trainer", "model.pkl"),
            )
            return model_trainer_config

        except Exception as e:
            raise USvisaException(e, sys)
        

    
    def get_model_trainer_artifact(self) -> ModelTrainerArtifact:
        """
        Returns the configuration for Model Training
        """
        try:
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=os.path.join(self.artifact_dir, "model_trainer", "model.pkl"),
            )
            return model_trainer_artifact

        except Exception as e:
            raise USvisaException(e, sys)    
