import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import os

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/economic_data_all_countries.csv')
RF_OUT_PATH = os.path.join(os.path.dirname(__file__), '../data/rf_forecasts.csv')
ARIMA_OUT_PATH = os.path.join(os.path.dirname(__file__), '../data/arima_forecasts.csv')

# Load data
df = pd.read_csv(DATA_PATH)

# --- Random Forest Forecasts ---
rf_results = []
for (country, indicator), group in df.groupby(['country', 'indicator_name']):
    group = group.sort_values('year')
    if len(group) > 10:
        X = np.array(group['year'].values).reshape(-1, 1)
        y = np.array(group['value'].values)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        future_years = np.arange(group['year'].max()+1, group['year'].max()+6).reshape(-1, 1)
        preds = rf.predict(future_years)
        for year, pred in zip(future_years.flatten(), preds):
            rf_results.append({
                'country': country,
                'indicator_name': indicator,
                'year': year,
                'rf_forecast': pred
            })
if rf_results:
    pd.DataFrame(rf_results).to_csv(RF_OUT_PATH, index=False)
    print(f"Random Forest forecasts saved to {RF_OUT_PATH}")
else:
    print("No Random Forest forecasts generated.")

# --- ARIMA Forecasts ---
arima_results = []
for (country, indicator), group in df.groupby(['country', 'indicator_name']):
    group = group.sort_values('year')
    y = group['value'].dropna().values
    if len(y) > 5:
        try:
            model = ARIMA(y, order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=5)
            forecast_years = np.arange(group['year'].max()+1, group['year'].max()+6)
            for year, pred in zip(forecast_years, forecast):
                arima_results.append({
                    'country': country,
                    'indicator_name': indicator,
                    'year': year,
                    'arima_forecast': pred
                })
        except Exception as e:
            print(f"ARIMA failed for {country}-{indicator}: {e}")
if arima_results:
    pd.DataFrame(arima_results).to_csv(ARIMA_OUT_PATH, index=False)
    print(f"ARIMA forecasts saved to {ARIMA_OUT_PATH}")
else:
    print("No ARIMA forecasts generated.")

# --- (Optional) Add more analytics, SHAP, LLM insight updates, etc. here ---
# You can expand this script as needed for further automation.
