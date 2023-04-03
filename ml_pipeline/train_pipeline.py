from zenml.pipelines import pipeline
from zenml.steps import step

from .steps.data_puller import data_puller
from .steps.model_train import train

from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.steps import MLFlowDeployerParameters




@pipeline
def model_train_pipeline(data_puller, train, deployment_trigger, model_deployer):
    data_puller()
    model = train()
    deploy_decision = deployment_trigger()
    model_deployer(deploy_decision, model)
    