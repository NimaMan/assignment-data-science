
from math import sqrt 
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def evaluate_forecasts(actual, predicted):
    scores = dict()
    scores["mse"] = mean_squared_error(actual, predicted, squared=False)
    scores["mae"] = mean_absolute_error(actual, predicted)
    scores["mape"] = mean_absolute_percentage_error(actual, predicted)
    
    return scores

