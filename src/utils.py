import os
import sys

# import numpy as np 
# import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluates multiple regression models and returns their performance scores.
    
    Arguments:
        X_train (numpy.ndarray): Training feature data.
        y_train (numpy.ndarray): Training target data.
        X_test (numpy.ndarray): Testing feature data.
        y_test (numpy.ndarray): Testing target data.
        models (dict): Dictionary of models to evaluate.
        params (dict): Dictionary of hyperparameters for the models.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para= params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)


