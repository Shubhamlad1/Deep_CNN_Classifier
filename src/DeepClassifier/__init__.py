import sys 
import os
import logging

logging_str= "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"

logdir= "log"
log_filepath= os.path.join(logdir, "running_logs.log")
os.makedirs(logdir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)


logger= logging.getLogger('deepClassifireLogger')
