import os
import tensorflow as tf
import time
from DeepClassifier.utils.comman import save_json
from DeepClassifier.entity.config_entity import EvaluationConfig
from pathlib import Path
from urllib.parse import urlparse
import mlflow.keras



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        print("**************************************************************************")
        #print(self.config.all_params)

    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=[224, 224, 3][:-1],
            batch_size=16,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory="artifact\data_ingestion\PetImages",
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path:Path)-> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model("artifact/training/model.h5")
        self._valid_generator()
        self.scores = self.model.evaluate(self.valid_generator)

    def save_scores(self):
        scores = {"Loss": self.scores[0], "Accuracy": self.scores[1]}
        save_json(path= Path("artifact/evaluation/scores.json"), data= scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri("https://dagshub.com/Shubhamlad1/Deep_CNN_Classifier.mlflow")
        mlflow.set_experiment("my-experiment")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            #mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"Loss": self.scores[0], "Accuracy": self.scores[1]}
            )

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")