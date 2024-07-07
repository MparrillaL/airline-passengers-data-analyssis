import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def fit_arima_model(data):
    # Fit ARIMA model
    model = ARIMA(data['passengers'], order=(5,1,0))
    model_fit = model.fit()
    
    # Forecast
    forecast = model_fit.forecast(steps=12)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['passengers'], label='Observed')
    plt.plot(data.index[-12:], forecast, label='Forecast', color='red')
    plt.title('ARIMA Forecast')
    plt.xlabel('Year')
    plt.ylabel('Passengers')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    data_file = 'C:/Users/manue/OneDrive/Escritorio/ESTUDIO/PROGRAMACIÓN/PYTHON/airlinedata/data/airline-passengers.csv'
    df = pd.read_csv(data_file, parse_dates=['month'], index_col='month')
    fit_arima_model(df)