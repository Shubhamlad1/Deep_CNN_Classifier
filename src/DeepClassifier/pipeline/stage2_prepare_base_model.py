from DeepClassifier.config.configuration import ConfigurationManager
from DeepClassifier.components import DataIngestion
from DeepClassifier import logger
from DeepClassifier.components import PrepareBaseModel


STAGE_NAME= "Prepare_Base_Model"

def main():  
    config= ConfigurationManager()

    #----object creation for class Configmanager--------
    
    prepare_base_model_config= config.prepare_base_model_config()
    prepare_base_model= PrepareBaseModel(config=prepare_base_model_config)
    prepare_base_model.get_base_model()
    prepare_base_model.update_base_model()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>> Stage Name: {STAGE_NAME} Started")
        main()
        logger.info(f">>>>>>>>> Stage Name: {STAGE_NAME} Completed/n")
        logger.info("---------------------------------------------------------\n")
    except Exception as e:
        logger.info(e)
        raise e