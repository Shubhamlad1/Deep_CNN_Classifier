from DeepClassifier import logger
from DeepClassifier.config.configuration import ConfigurationManager
from DeepClassifier.components import Evaluation
from DeepClassifier.entity import EvaluationConfig

STAGE_NAME= "Model Evaluation"

def main():  
    config= ConfigurationManager()

    #----object creation for class Configmanager--------

    config = ConfigurationManager()
    val_config = config.get_evaluation_config()
    evaluation = Evaluation(EvaluationConfig)
    evaluation.evaluation()
    evaluation.save_scores()
    evaluation.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>>> Stage Name: {STAGE_NAME} Started")
        main()
        logger.info(f">>>>>>>>> Stage Name: {STAGE_NAME} Completed/n")
        logger.info("---------------------------------------------------------\n")
    except Exception as e:
        logger.info(e)
        raise e