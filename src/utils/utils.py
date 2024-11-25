import os
import sys
from src.logger import logging
from src.exception import CustomeException
import pandas as pd
from pathlib import Path


def read_data():
    file_path = "experiments/Gem_dataset.csv"
    data=pd.read_csv(Path(file_path))
    return data