import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import dill

from src.utils.utils import load_object

from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from src.logger import logging
from src.exception import CustomeException

from dataclasses import dataclass

@dataclass
class ModelEvaluationConfig:
    model_path = os.path.join("artifacts","model.pkl")

class ModelEvaulation:
    def __init__(self):
        logging.info("evaluation started")
        self.model_evaluation_config = ModelEvaluationConfig()
    
    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))# here is RMSE
        mae = mean_absolute_error(actual, pred)# here is MAE
        r2 = r2_score(actual, pred)# here is r3 value
        logging.info("evaluation metrics captured")
        return rmse, mae, r2
    
    
    def initiate_model_evaulation(self,train_array,test_array):
        try:
            X_test,y_test=(test_array[:,:-1], test_array[:,-1])

            model=load_object(file_path=self.model_evaluation_config.model_path)

            mlflow.set_registry_uri(uri="http://127.0.0.1:8080")  #Uncomment this line if you want if different Cloud eg S3 Bucket
             
            logging.info("model has register")
            
            tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type_store)

            with mlflow.start_run():

                prediction=model.predict(X_test)

                (rmse,mae,r2)=self.eval_metrics(y_test,prediction)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                # Model registry does not work with file store
                if tracking_url_type_store != "file":   #if not in local then it will not creat a file

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")
                
                logging.info("model log successfull")

        except Exception as e:
            raise CustomeException(e,sys)