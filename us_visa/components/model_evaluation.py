from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from us_visa.exception import USvisaException
from us_visa.constants import TARGET_COLUMN, CURRENT_YEAR
from us_visa.logger import logging
import sys
import pandas as pd
from typing import Optional
from us_visa.entity.estimator import USvisaModel
from us_visa.entity.estimator import TargetValueMapping
from us_visa.utils.main_utils import load_object, save_object  # Utility functions to load and save a saved model
from dataclasses import dataclass
import os

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float

class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e

    def get_best_model(self) -> Optional[USvisaModel]:
        """
        Get the best locally saved model if available.
        """
        try:
            best_model_path = self.model_eval_config.local_model_path  # Local model storage path
            if os.path.exists(best_model_path):
                return load_object(best_model_path)
            return None
        except Exception as e:
            raise USvisaException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Evaluate the trained model and compare it with the best locally saved model.
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df['company_age'] = CURRENT_YEAR - test_df['yr_of_estab']
            
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            #y = y.replace(TargetValueMapping()._asdict())
            y = y.replace(TargetValueMapping()._asdict()).infer_objects(copy=False)

            
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            best_model_f1_score = None
            best_model = self.get_best_model()
            
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                difference=trained_model_f1_score - tmp_best_model_score
            )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Run the model evaluation process and save the best model locally.
        """
        try:
            evaluate_model_response = self.evaluate_model()
            local_model_path = self.model_eval_config.local_model_path  # Local model path

            # Save the new model if it outperforms the previous one
            if evaluate_model_response.is_model_accepted:
                save_object(self.model_trainer_artifact.trained_model_file_path, local_model_path)
                logging.info(f"New best model saved at: {local_model_path}")

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)
            
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e
