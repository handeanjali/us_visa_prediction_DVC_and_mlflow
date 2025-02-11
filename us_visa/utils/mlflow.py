import mlflow

def start_mlflow_run():
    """
    Start an MLflow run.
    """
    mlflow.start_run()

def log_params(params):
    """
    Log model parameters to MLflow.
    """
    mlflow.log_params(params)

def log_metrics(metrics):
    """
    Log evaluation metrics to MLflow.
    """
    mlflow.log_metrics(metrics)

def end_mlflow_run():
    """
    End the current MLflow run.
    """
    mlflow.end_run()
