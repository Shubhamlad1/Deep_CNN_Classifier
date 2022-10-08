from DeepClassifier.constants import PARAM_FILE_PATH, CONFIG_FILE_PATH
from DeepClassifier.utils import read_yaml, create_directories
from DeepClassifier.entity import (DataIngestionConfig, 
                                    PrepareCallbackConfig, 
                                    PrepareBaseModelConfig, 
                                    TrainingConfig,
                                    EvaluationConfig
                                    )
from pathlib import Path
from DeepClassifier import logger
import os

class ConfigurationManager:
    def __init__(
        self,
        params_filepath = PARAM_FILE_PATH,
        config_filepath = CONFIG_FILE_PATH,
        ):
        
        self.config= read_yaml(config_filepath)
        self.params= read_yaml(Path('params.yaml'))
        create_directories([self.config['artifacts_roots']])

    def get_ingetion_config(self):
        config= self.config['data_ingestion']
        create_directories([config["root_dir"]])

        data_ingestion_config= DataIngestionConfig(
                root_dir= config["root_dir"],
                source_URL= config["source_URL"],
                local_data_file= config["local_data_file"],
                unzip_dir= config["unzip_dir"]
        )
        logger.info("<<<<<<<<<<<<<<<<<<<<<Data Ingetion Configuration Complted>>>>>>>>>>>>>>>>>")

        return data_ingestion_config

    def prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config= self.config["prepare_base_model"]
        logger.info(f"Creating Folder {config['root_dir']}")
        create_directories([config['root_dir']])
        print(self.params)

        prepare_base_model_config= PrepareBaseModelConfig(
                root_dir= Path(config.root_dir),
                base_model_path= Path(config.base_model_path),
                updated_base_model_path= Path(config.updated_base_model_path),
                params_image_size= self.params["IMAGE_SIZE"],
                params_learning_rate=self.params["LEARNING_RATE"],
                params_include_top= self.params["INCLUDE_TOP"],
                params_weights= self.params["WEIGHTS"],
                params_classes= self.params["CLASSES"]
            )

        logger.info(f"Configuration for prepare base model completed")

        return prepare_base_model_config

    def get_prepare_callback_config(self) -> PrepareCallbackConfig:
        config = self.config['prepare_callbacks']
        check_point_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories(
            [Path(config['tensorboard_root_log_dir']),
            Path(check_point_dir)]
        )

        get_prepare_callback_config= PrepareCallbackConfig(
                root_dir= Path(config.root_dir),
                tensorboard_root_log_dir= Path(config.tensorboard_root_log_dir),
                checkpoint_model_filepath= Path(config.checkpoint_model_filepath)
            )

        logger.info(f"Configuration for prepare callbacks completed")

        return get_prepare_callback_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config['training']
        prepare_base_model= self.config["prepare_base_model"]
        params=self.params

        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "PetImages")
        create_directories([Path(training["root_dir"])])
        

        get_training_data_config= TrainingConfig(
                root_dir= Path(training.root_dir),
                trained_model_path = Path(training["trained_model_path"]),
                updated_base_model_path = Path(prepare_base_model["updated_base_model_path"]),
                training_data=Path(training_data),
                params_epochs=params.EPOCHS,
                params_batch_size= params.BATCH_SIZE,
                params_is_augmentation= params.AUGMENTATION,
                params_image_size= params.IMAGE_SIZE
            )

        logger.info(f"Configuration for Training Data is completed")

        return get_training_data_config


    def get_evaluation_config(self) -> EvaluationConfig:
        evaluation_config = self.config["evaluation"]
        training_config = self.config["training"]
        create_directories([Path(evaluation_config["root_dir"])])
        training_data= os.path.join(self.config.data_ingestion.unzip_dir, "PetImages")

        eval_config = EvaluationConfig(
                path_of_model= "artifact/training/model.h5",
                score_file_path= Path(evaluation_config["score_file_path"]),
                training_data= training_data,
                all_params= self.params,
                params_batch_size= self.params.BATCH_SIZE,
                params_images_size= [224, 224, 3]
            )

        return eval_config