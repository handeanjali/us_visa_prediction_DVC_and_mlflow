from us_visa.configuration.configuration import ConfigurationManager
from us_visa.components.data_validation import DataValidation
from us_visa.logger import logging

# Define the stage name for logging purposes
STAGE_NAME = "Data Validation Stage"


class DataValidationTrainingPipeline:
    """Pipeline to execute the Data Validation stage."""

    def __init__(self):
        """Initialize the Data Validation pipeline."""

        pass

    def main(self):
        """Main execution method for data validation."""

        # Initialize Configuration Manager
        config = ConfigurationManager()
        
        # Retrieve configurations and artifacts
        data_validation_config = config.get_data_validation_config()
        data_ingestion_artifact = config.get_data_ingestion_artifact()
        
        # Initialize DataValidation component
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config,
        )
        
         # Start data validation process
        data_validation.initiate_data_validation()


if __name__ == "__main__":
    try:
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
