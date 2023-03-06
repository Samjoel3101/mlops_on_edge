from zenml.pipelines import pipeline

from .data_puller import data_puller
from .model_train import train


@pipeline
def model_train_pipeline(step1, step2):
    step1()
    step2()


pipeline_instance = model_train_pipeline(data_puller(), train())

if __name__ == "__main__":
    pipeline_instance.run()
    