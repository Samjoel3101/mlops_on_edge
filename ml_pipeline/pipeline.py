from zenml.pipelines import pipeline


@pipeline
def model_train_pipeline(step1, step2):
    train_dir = step1("mlops-initial", "train_dir")
    step2(train_dir)
