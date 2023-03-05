docker run -it -d -p 8080:8080 --name zenml \
    --mount type=bind,source=$PWD/zenml-server,target=/zenml/.zenconfig/local_stores/default_zen_store \
    zenmldocker/zenml-server
    
zenml connect --url http://localhost:8080

# Register the MLflow experiment tracker
zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow \ 
    --tracking_uri=http://localhost:5555

# Register the MLflow experiment tracker
zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow

# Register and set a stack with the new experiment tracker
zenml stack register custom_stack -e mlflow_experiment_tracker ... --set