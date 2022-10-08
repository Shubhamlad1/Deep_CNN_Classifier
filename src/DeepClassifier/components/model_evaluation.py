import os
import tensorflow as tf
import time
from DeepClassifier.utils.comman import save_json
from DeepClassifier.entity.config_entity import EvaluationConfig
from pathlib import Path



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

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