import importlib
import sys, os
import types
from us_visa.utils import *
#import optuna
from us_visa.entity import config_entity
from us_visa.entity import artifact_entity
from us_visa.exception import ApplicationException
from us_visa.logger import logging
from us_visa.utils.main_utils import save_object, load_object
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from us_visa.entity.config_entity import SavedModelConfig
from us_visa.constants import *
import mlflow
import logging
from mlflow.tracking import MlflowClient
from sklearn.metrics import r2_score

class Experiments_evaluation:
    def __init__(self,experiment_name,run_name) :
        self.experiment_name=experiment_name
        self.run_name=run_name
        
        self.best_model_run_id=None
        self.best_model_uri=None
        self.model_path=None

        self.artifact_uri = None
        self.model_name=None

            
            
    def get_best_model_run_id(self,experiment_name, metric_name):
        # Get the experiment ID
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        # Retrieve runs and sort by the specified metric
        runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string='', order_by=[f"metrics.{metric_name} DESC"])
        
        if runs.empty:
            print("No runs found for the specified experiment and metric.")
            return None
        
        # Get the best run
        best_run = runs.iloc[0]
        self.best_model_run_id = best_run.run_id
        
        # Load the best model
        self.best_model_uri=(f"runs:/{self.best_model_run_id}/model")


    def download_model(self, dst_path):
        self.artifact_uri=mlflow.get_run(self.best_model_run_id).info.artifact_uri
        model_uri = f"{self.artifact_uri}/{self.model_name}"
        model = mlflow.pyfunc.load_model(model_uri)
        save_object(file_path=dst_path,obj=model)

    def create_run_report(self):
        # Create an MLflow client
        client = MlflowClient()
        run_id=self.best_model_run_id
        # Get the run details
        run = client.get_run(run_id)

        # Report Data 
        # List the contents of the artifact_uri directory
        model_name = self.model_name
        parameters = run.data.params
        metrics = str(run.data.metrics['R2_score'])  # Retrieve metrics

        return model_name,parameters,metrics

    def run_mlflow_experiment(self,R2_score,model,parameters,model_name):
        
        self.model_name=model_name
        # Create or get the experiment
        mlflow.set_experiment(self.experiment_name)
        
        # Start a run
        with mlflow.start_run(run_name=self.run_name):
            # Log metrics, params, and model
            mlflow.log_metric("R2_score", float(R2_score))
            mlflow.log_params(parameters)
            mlflow.sklearn.log_model(model, f"{model_name}")

        
        logging.info("Checking for best model from the Mlflow Logs")
        
        self.get_best_model_run_id(metric_name='R2_score', experiment_name=self.experiment_name)
        
        print(f"Best model Run id: {self.best_model_run_id}")

        return self.best_model_run_id
