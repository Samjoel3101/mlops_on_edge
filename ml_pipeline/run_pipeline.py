from .data_puller import data_puller
from .model_train import train

from .train_pipeline import model_train_pipeline

if __name__ == "__main__":
    model_train_pipeline.run(step1=data_puller, step2=train)
