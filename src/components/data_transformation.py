import sys
import os
from src.logger import logging
from src.exception import CustomeException

from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OrdinalEncoder

from sklearn.compose import ColumnTransformer

from src.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
    
    
class DataTransformation:
    def __init__(self):
        self.data_transforamtion_congif = DataTransformationConfig()
    
    def get_preprocessor_obj(self):
        try:
            logging.info("Numberical and Categorical Column Preporcessing Started")
             # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color','clarity']
            numerical_cols = ['carat', 'depth','table', 'x', 'y', 'z']
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info(f"Numerical Column: {numerical_cols}")
            logging.info(f"Categorical Column: {categorical_cols}")
            
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("ordinalencoder",OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Numberical and Categorical Column Preporcessing Completed")
            
            preprocessor = ColumnTransformer([
                ("numerical_pipeline",numerical_pipeline,numerical_cols),
                ("categorical_pipeline",categorical_pipeline,categorical_cols)
            ])
            
            
            return preprocessor
            
        except Exception as e:
            raise CustomeException(e, sys)
    
    def initiate_data_transformation(self,train_path, test_path):
        try:
            logging.info("Reading the Train and Test CSV File")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            target_column = "price"
            drop_column = [target_column, "id"]
            
            input_feature_train_df = train_df.drop(columns=drop_column,axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=drop_column,axis=1)
            target_feature_test_df = test_df[target_column]
            
            preprocessor_obj = self.get_preprocessor_obj()
            
            logging.info("Applying Preporcessing in Train and Test Data Set")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
           
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                self.data_transforamtion_congif.preprocessor_path,
                preprocessor_obj
            )
            logging.info("Saved Preprocessor Object")
            
            return(
                train_arr,
                test_arr,
                self.data_transforamtion_congif.preprocessor_path
            )
            
        except Exception as e:
            raise CustomeException(e, sys)
         