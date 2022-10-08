from DeepClassifier.config.configuration import ConfigurationManager
from DeepClassifier.components import DataIngestion
from DeepClassifier import logger


STAGE_NAME= "Data_Ingestion_Stage"

def main():  
    config= ConfigurationManager()

    #----object creation for class Configmanager--------
    
    data_ingestion_config= config.get_ingetion_config()
    data_ingestion= DataIngestion(config=data_ingestion_config)
    data_ingestion.download_file()
    data_ingestion.unzip_and_clean()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>> Stage Name: {STAGE_NAME} Started")
        main()
        logger.info(f">>>>>>>>> Stage Name: {STAGE_NAME} Completed")
        logger.info("---------------------------------------------------------\n")
    except Exception as e:
        logger.info(e)
        raise e
