import os
import urllib.request as request
from zipfile import ZipFile
from DeepClassifier.entity import DataIngestionConfig
from pathlib import Path
from DeepClassifier import logger
from DeepClassifier.utils import get_size
from tqdm import tqdm


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config= config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            logger.info(f"File {self.config.local_data_file} not exist in directory")
            logger.info(">>>>>>>>>>>Downloading Started.....<<<<<<<<<<<<<<<<<")
            filename, headers= request.urlretrieve(
                url= self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"Donloading Completed File Size: {get_size(Path(self.config.local_data_file))}")
    def _get_updated_list_of_file(self, list_of_files):
        return [f for f in list_of_files if f.endswith(".jpg") and ("Cat" in f or "Dog" in f)]

    def _preprocess(self, zf: ZipFile, f: str, Working_dir: str):
        target_file_path= os.path.join(Working_dir, f)
        if not os.path.exists(Path(target_file_path)):
            logger.info(f"Extrating the Files................")
            zf.extract(f, Working_dir)

        if os.path.getsize(target_file_path) == 0:
            logger.info(f"file name {target_file_path} removed because of zero size")
            os.remove(target_file_path)


    def unzip_and_clean(self):
        with ZipFile(file=self.config.local_data_file, mode="r") as zf:
            list_of_file= zf.namelist()
            updated_list_of_files= self._get_updated_list_of_file(list_of_file)
            for f in tqdm(updated_list_of_files):
                self._preprocess(zf, f, self.config.unzip_dir)