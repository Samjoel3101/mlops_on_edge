from zenml.steps import step


@step
def deployment_trigger() -> bool:
    """TODO: Only deploy if the test accuracy > 90%."""
    return True
