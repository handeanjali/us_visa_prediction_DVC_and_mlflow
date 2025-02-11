from us_visa.configuration.configuration import ConfigurationManager
from us_visa.components.data_ingestion import DataIngestion
from us_visa.logger import logging


# Define the stage name
STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    """Pipeline to handle the data ingestion process."""

    def __init__(self):
        pass

    def main(self):
        """Main method to execute data ingestion."""
        # Initialize configuration manager
        config = ConfigurationManager()

        # Retrieve data ingestion configuration
        data_ingestion_config = config.get_data_ingestion_config()

        # Initialize and execute data ingestion
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion.initiate_data_ingestion()


if __name__ == "__main__":
    try:
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
