import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from DeepClassifier.constants import *
from pathlib import Path
from DeepClassifier.entity import PrepareCallbackConfig


class PrepareCallback:
    def __init__(self, config: PrepareCallbackConfig):
        self.config = config

    @property
    def _create_tb_callbacks(self):
        time_stamp= time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir= os.path.join(self.config.tensorboard_root_log_dir, f"tb_logs_at_{time_stamp}")

        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    @property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(filepath= self.config.checkpoint_model_filepath,
                    save_best_only= True)

    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]