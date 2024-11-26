import os
import sys
from src.logger import logging
from src.exception import CustomeException
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig

from src.utils.utils import load_object

import pandas as pd

@dataclass
class PredictPipelineConfig:
    preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
    model_path = os.path.join("artifacts","model.pkl")


class PredictPipeline:   
    def __init__(self):
        logging.info("initialize the object")
        self.predict_pipeline_config = PredictPipelineConfig()

    def predict(self,features):
        try:
            logging.info("Loading Preprocessor and perform Preprocessing")       
            preprocessor=load_object(file_path = self.predict_pipeline_config.preprocessor_path)
            
            scaled_feature=preprocessor.transform(features)
            logging.info("Preprocessoring Completed")
            
            logging.info("Loading Model and perform Prediction")
            model=load_object(file_path = self.predict_pipeline_config.model_path)
    
            pred=model.predict(scaled_feature)
            logging.info(f"Prediction Completed and Price is {pred} Lakhs")

            return pred

        except Exception as e:
            raise CustomeException(e,sys)


class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity
            
    def get_data_as_dataframe(self):
        try:
            logging.info("Reading the value from the Webpage")
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
                }
            logging.info("Sucessfully Read the Value")
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomeException(e,sys)