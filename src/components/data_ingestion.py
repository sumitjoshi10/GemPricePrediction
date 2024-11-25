import os
import sys
from src.logger import logging
from src.exception import CustomeException
import pandas as pd
from sklearn.model_selection import train_test_split 

from dataclasses import dataclass

from src.utils.utils import read_data


@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join("artifacts","raw.csv")
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            data = read_data()
            logging.info("Data Reading Competed")
            
            # To Create a artifacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True) 
            
            # Save the Raw data to the raw_data_path
            data.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            # Train Test Split the data
            train_set, test_set = train_test_split(data,test_size=0.2, random_state=42)
            
            # Saving the Train and Test Data into the path
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Data Ingestion is completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomeException(e,sys)
        

if __name__ == "__main__":
    data_ing = DataIngestion()
    print(data_ing.initiate_data_ingestion())