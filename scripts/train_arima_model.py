
import pmdarima as pm

def fit_arima(train_data, n_jobs=-1, alpha=0.1, verbose=True):
    # return a fitted ARIMA model using the Auto 
    model = pm.auto_arima(train_data, seasonal=True, m=12, trace=verbose, n_jobs=n_jobs, alpha=alpha )

    return model 


def forecast_next_day(model, num_steps=24):
    fc, conf_int = model.predict(n_periods=num_steps, return_conf_int=True)
    return fc, conf_int


def get_forecasts_confidence_intervals(model, test_data, num_steps=24):

    forecasts = []
    confidence_intervals = []

    for new_ob in test_data:
        fc, conf = forecast_next_day(model, num_steps)
        forecasts.append(fc)
        confidence_intervals.append(conf)

        # Updates the existing model with a small number of MLE steps
        model.update(new_ob)
    
    return forecasts, confidence_intervals

