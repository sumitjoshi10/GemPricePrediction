import os
import sys
from src.logger import logging
from src.exception import CustomeException
import pandas as pd
from pathlib import Path
# import json

import dill

from sklearn.metrics import r2_score,adjusted_rand_score
from sklearn.model_selection import ShuffleSplit,GridSearchCV


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

def save_csv(file_path, csv_file):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        df = pd.DataFrame.from_dict(csv_file,orient='index')
        print(df)
        df.to_csv(file_path,index=True)
    except Exception as e:
        raise CustomeException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test, model_params):
    try:
        
        scores = {}
        cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
        
        for model_name, mp in model_params.items():
            logging.info(f"Starting the Hyper parameter Tunnig for {model_name}")
            clf = GridSearchCV(mp["model"],mp["params"], cv = cv, return_train_score= False)
            clf.fit(X_train,y_train)
            
             # Store results
            # best_params = clf.best_params_
            best_score = clf.best_score_
            
            # Calculate R² score on the test set
            best_model = clf.best_estimator_
            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
          
            # scores.append({
            #     "model" : model_name,
            #     "best_score" : best_score,
            #     "best_params": best_params,
            #     "test_r2_score": r2,
            #     "avg_best_score": avg_score
            # })
            scores[model_name]={
                "best_model": best_model,
                "best_score" : best_score,
                "r2_score": r2
            }
            logging.info(f"Completed Hyper parameter Tunnig for {best_model} having scores as {best_score} and test R2 score as {r2}")
                   
        logging.info("Hyper parameter Tunning Complete")
        return scores
        
    except Exception as e:
        raise CustomeException(e,sys)