import numpy as np

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.steps import BaseParameters, Output, StepContext, step
from zenml.services.utils import load_last_service_from_step


class MLFlowDeploymentLoaderStepParams(BaseParameters):
    pipeline_name: str
    step_name: str
    running: bool = True


# Step to retrieve the service associated with the last pipeline run
@step(enable_cache=False)
def prediction_service_loader(
    params: MLFlowDeploymentLoaderStepParams, context: StepContext
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline"""

    service = load_last_service_from_step(
        pipeline_name=params.pipeline_name,
        step_name=params.step_name,
        running=params.running,
    )
    if not service:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{params.step_name} step in the {params.pipeline_name} pipeline "
            f"is currently running."
        )

    return service

# Use the service for inference
@step
def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray,
) -> Output(predictions=np.ndarray):
    """Run a inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    prediction = service.predict(data)
    prediction = prediction.argmax(axis=-1)

    return prediction
