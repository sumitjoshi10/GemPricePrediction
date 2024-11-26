from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaulation


data_ingestion = DataIngestion()
train_file_path, test_file_path = data_ingestion.initiate_data_ingestion()

data_trainformation = DataTransformation()
train_arr, test_arr , preprocessor_file_path = data_trainformation.initiate_data_transformation(train_file_path,test_file_path)

model_trainer = ModelTrainer()
model_trainer.initate_model_training(train_arr,test_arr)

model_evaluation = ModelEvaulation()
model_evaluation.initiate_model_evaulation(train_arr,test_arr)