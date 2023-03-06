from zenml.steps import step, Output


import mlflow

import uuid
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.optimizers import Adam


BASE_DIR = "/home/samjoel/Projects/mlops_on_edge/data"


@step(experiment_tracker="mlflow_experiment_tracker")
def train() -> None:
    save_dir = "train_dir"
    batch_size = 32
    img_size = 224
    directory = os.path.join(BASE_DIR, save_dir)

    # Get data loaders
    datagen = ImageDataGenerator(rescale=1 / 255.0, zoom_range=0.2, horizontal_flip=True, validation_split=0.15)

    train_generator = datagen.flow_from_directory(
        directory,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        subset="training",
        class_mode="categorical",
    )

    validation_generator = datagen.flow_from_directory(
        directory,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
        subset="validation",
        class_mode="categorical",
    )

    num_classes = len(np.unique(train_generator.classes))

    img_size = 224
    base_model = VGG16(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))

    for layer in base_model.layers[:-4]:
        layer.trainable = False

    last_output = base_model.output
    x = GlobalAveragePooling2D()(last_output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.inputs, outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    mlflow.tensorflow.autolog(log_models=False)

    history = model.fit(
        train_generator,
        epochs=1,
        validation_data=validation_generator,
    )

    mlflow.tensorflow.log_model(model, artifact_path=f"models")

    # Get loss curve
    plt.figure(figsize=(20, 8))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig("loss_curve.png")

    mlflow.log_artifact("loss_curve.png", "model-stats")

    # Get accuracy curve
    plt.figure(figsize=(20, 8))
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig("accuracy_curve.png")

    mlflow.log_artifact("accuracy_curve.png", "model-stats")
