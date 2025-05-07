import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Initiates the model training process.   
        Arguments:
            train_array (numpy.ndarray): Training data array.
            test_array (numpy.ndarray): Testing data array.
            preprocessor_path (str): Path to the preprocessor object.
        """

        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models=models)
            
            # Get the best model score from the report
            best_model_score = max(sorted(model_report.values()))

            # Get the best model name from the report
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score <= 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found is {best_model_name} with score {best_model_score}")

            #save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)

            # r2 score is used to evaluate the performance of regression models
            # It represents the proportion of variance in the dependent variable that can be explained by the independent variables.
            r2__score = r2_score(y_test, predicted)
            logging.info(f"R2 score of the best model is {r2__score}")
            return r2__score
        
            

        except Exception as e:
            raise CustomException(e, sys)