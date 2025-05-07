# Read the data from data source that is created by big data/ cloud team using data ingestion,
# Our aim :- To split and train the data

import os
import sys
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig, DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as dataframe")

             # Create the 'artifacts' directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")

            # Split the dataset into train and test sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":

    # 1. Data ingestion is the first step in the pipeline
    # It is responsible for fetching the data from the source and saving it in the required format
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # 2. Data transformation is the second step in the pipeline
    # It is responsible for transforming the data into the required format
    # It is responsible for preprocessing the data and saving the preprocessor object and splitting the data into train and test sets
    # It is responsible for saving the transformed data in the required format
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
    # get the transformed data train and test data
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # 3. Model trainer is the third step in the pipeline
    # It is responsible for training the model and saving the trained model
    # It is responsible for evaluating the model and saving the best model
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_array, test_array))



