import os
from pathlib import Path

import logging

# from src.logger import logging

logging.basicConfig(level=logging.INFO, format='[$(asctime)s]: %(message)s')

logging.info("Files and Folder Creation Started")

list_of_files = [
    
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",
    "src/utils/__init__.py"
    "src/utils/utils.py",
    "src/logger.py",
    "src/exception.py",
    "test/unit/__init__.py",
    "test/integration/__init__.py",
    "init_setup.sh", # Setuo Shell Script. Will write the All required Linux Command and run it in single shot
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cgf",
    "pyproject.toml",
    "tox.ini", # to test the code in local environment
    "experiments/experiments.ipynb"   
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for the file {file_name}")
        
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path,"w") as f:
            pass
            logging.info(f"Creating empty file: {file_path}")
            
    else:
        logging.info(f"{file_name} already exist")


logging.info(" Files and Folder Created")