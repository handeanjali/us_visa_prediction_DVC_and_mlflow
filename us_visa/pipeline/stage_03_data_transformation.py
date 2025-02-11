from us_visa.configuration.configuration import ConfigurationManager
from us_visa.components.data_transformation import DataTransformation
from us_visa.logger import logging

# Define the stage name
STAGE_NAME = "Data Transformation Stage"


class DataTransformationTrainingPipeline:
    """Pipeline to manage the data transformation stage."""
    
    def __init__(self):
        """Initialize the data transformation pipeline."""
        pass

    def main(self):
        """Main method to execute the data transformation process."""

        # Load configuration manager
        config = ConfigurationManager()

        # Retrieve necessary configurations and artifacts
        data_transformation_config = config.get_data_transformation_config()
        data_ingestion_artifact = config.get_data_ingestion_artifact()
        data_validation_artifact = config.get_data_validation_artifact()

        # Initialize and run data transformation
        data_transformation = DataTransformation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config,
        )
        data_transformation.initiate_data_transformation()


if __name__ == "__main__":
    try:
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
