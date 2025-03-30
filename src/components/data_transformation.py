"""
Data Transformation Script

The main purpose of data transformation is to perform:
- Data Cleaning (handling missing values)
- Feature Engineering (scaling, encoding)
- Preparing data for model training

This module creates a preprocessor pipeline to transform numerical and categorical features.
"""

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Applies transformers to columns of an array or pandas DataFrame.
from sklearn.compose import ColumnTransformer

#Replace missing values using a descriptive statistic (e.g. mean, median, or most frequent) along each column, or using a constant value.
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os

#dataclasses are used to automatically creates init, repr, eq, etc. methods for classes that are mainly used to store data.
@dataclass
class DataTranformationConfig:
    """
    Configuration class for Data Transformation.
    It stores the path where the preprocessor object will be saved.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    Class responsible for creating data transformation pipeline.
    It handles missing values, encoding categorical variables, and scaling numerical features.
    """
    
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessor pipeline that transforms
        numerical and categorical features.

        Returns:
            preprocessor (ColumnTransformer): Transformer object with pipelines for numeric and categorical data.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    {"imputer", SimpleImputer(strategy="median")},
                    {"scaler", StandardScaler()}
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline, numerical_columns),
                    ("cat_pipeline",cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
