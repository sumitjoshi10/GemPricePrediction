from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


data_ingestion = DataIngestion()
train_file_path, test_file_path = data_ingestion.initiate_data_ingestion()

data_trainformation = DataTransformation()
data_trainformation.initiate_data_transformation(train_file_path,test_file_path)