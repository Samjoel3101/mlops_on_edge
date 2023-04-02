from zenml.pipelines import pipeline
from zenml.steps import step

from .data_puller import data_puller
from .model_train import train

from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.steps import MLFlowDeployerParameters


@step
def deployment_trigger() -> bool:
    """TODO: Only deploy if the test accuracy > 90%."""
    return True


@pipeline
def model_train_pipeline(data_puller, train, deployment_trigger, model_deployer):
    data_puller()
    model = train()
    deploy_decision = deployment_trigger()
    model_deployer(deploy_decision, model)
    


pipeline_instance = model_train_pipeline(
    data_puller = data_puller(), 
    train = train(), 
    deployment_trigger=deployment_trigger(), 
    model_deployer=mlflow_model_deployer_step(
        MLFlowDeployerParameters(workers=1)
    ))

if __name__ == "__main__":
    pipeline_instance.run()
    