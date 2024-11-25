import os
import sys
from src.logger import logging
from src.exception import CustomeException
import pandas as pd
from pathlib import Path

import dill


def read_data():
    file_path = "experiments/Gem_dataset.csv"
    data=pd.read_csv(Path(file_path))
    return data

def save_object (file_path , obj):
    try:
        file_dir = os.path.dirname(file_path)
        
        os.makedirs(file_dir,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomeException(e , sys)