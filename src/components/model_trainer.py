import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomeException
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from src.utils.utils import save_object,evaluate_model,save_csv

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    best_model_file_path = os.path.join('artifacts','model_score.csv')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # models={
            #     'LinearRegression':LinearRegression(),
            #     'Lasso':Lasso(),
            #     'Ridge':Ridge(),
            #     'Elasticnet':ElasticNet()
            # }
            model_params = {
                "LinearRegression":{
                    "model" : LinearRegression(),
                    "params":{
                    }
                },
                "Lasso":{
                    "model" : Lasso(),
                    "params":{

                    }
                },
                "Ridge": {
                    "model": Ridge(),
                    "params": {
                        "alpha": [0.1,1.0,2.0],
                    }
                },
                 
                'Elasticnet':{
                    "model" :ElasticNet(),
                    "params":{
                        "alpha": [1.0,2.0],                    # Regularization strength (L1 + L2)
                        "l1_ratio": [0.4,0.5,0.6],  
                    }
                }
            }
            
            # model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,model_params)
            print()
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            save_csv(
                file_path=self.model_trainer_config.best_model_file_path,
                csv_file=model_report
            )

            # Will get the best model name from the below.
            best_model_name = max(model_report, key=lambda k: model_report[k]['r2_score'])
            best_model_details = model_report[best_model_name]
            
            
            best_model = best_model_details["best_model"]
            best_model_score = best_model_details["r2_score"]
            
            if best_model_details["r2_score"]<0.6:
                raise CustomeException("No best model found")

            print(f'Best Model Found , Model Name : {best_model} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomeException(e,sys)