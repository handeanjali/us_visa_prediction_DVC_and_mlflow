from us_visa.configuration.configuration import ConfigurationManager
from us_visa.components.model_trainer import ModelTrainer
import logging


# Define the stage name
STAGE_NAME = "Model Training Stage"


class ModelTrainerTrainingPipeline:
    """Pipeline to manage the model training stage."""

    def __init__(self):
        """Initialize the Model Training pipeline."""
        pass

    def main(self):
        """Main method to execute the model training process."""

         # Load configuration manager
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        data_transformation_artifact = config.get_data_transformation_artifact()

        
         # Initialize and run model trainer
        model_trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=model_trainer_config,
        )
        model_trainer.initiate_model_trainer()


if __name__ == "__main__":
    try:
        logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
