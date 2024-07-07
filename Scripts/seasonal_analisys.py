import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns

def seasonal_analysis(data):
    # Seasonal decomposition
    result = seasonal_decompose(data['passengers'], model='multiplicative', period=12)
    
    # Plot decomposed components
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    result.observed.plot(ax=ax1)
    ax1.set_ylabel('Observed')
    result.trend.plot(ax=ax2)
    ax2.set_ylabel('Trend')
    result.seasonal.plot(ax=ax3)
    ax3.set_ylabel('Seasonal')
    result.resid.plot(ax=ax4)
    ax4.set_ylabel('Residual')
    plt.tight_layout()
    plt.show()
    
    # Heatmap for seasonal analysis
    data['year'] = data.index.year
    heatmap_data = data.pivot_table(values='passengers', index='month', columns='year', aggfunc='mean')

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".0f", linewidths=0.5, linecolor='white')
    plt.title('Seasonal Heatmap of Airline Passengers')
    plt.xlabel('Year')
    plt.ylabel('Month')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

if __name__ == "__main__":
    data_file = 'C:/Users/manue/OneDrive/Escritorio/ESTUDIO/PROGRAMACIÓN/PYTHON/airlinedata/data/airline-passengers.csv'
    df = pd.read_csv(data_file, parse_dates=['month'], index_col='month')
    seasonal_analysis(df)
