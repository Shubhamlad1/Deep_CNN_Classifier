from DeepClassifier.config.configuration import ConfigurationManager
from DeepClassifier import logger
from DeepClassifier.components import PrepareCallback
from DeepClassifier.components import TrainingData


STAGE_NAME= "Prepare_Base_Model"

def main():  
    config= ConfigurationManager()

    #----object creation for class Configmanager--------

    prepare_callback_config= config.get_prepare_callback_config()
    prepare_callbacks= PrepareCallback(config=prepare_callback_config)
    callback_list= prepare_callbacks.get_tb_ckpt_callbacks()

    training_config= config.get_training_config()
    training= TrainingData(config=training_config)
    training.get_base_model()
    training.train_valid_generator()
    training.train(callback_list=callback_list)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>> Stage Name: {STAGE_NAME} Started")
        main()
        logger.info(f">>>>>>>>> Stage Name: {STAGE_NAME} Completed/n")
        logger.info("---------------------------------------------------------\n")
    except Exception as e:
        logger.info(e)
        raise e