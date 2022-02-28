# Necessary Packages
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm


from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.seasonal import seasonal_decompose
# Bitcoin data

data = pd.read_csv('go.csv')
a = len(data['price'])
th = 124
l = []
print(data[155:156])
while th + 1 < 272:
#divide into train and validation set
    train = data[0:th]['price']
    valid = data[th:th+1]['price']
    #preprocessing (since arima takes univariate series as input)
    #data.drop('date',axis=1,inplace=True)
    #print(data)
    from pmdarima import auto_arima
    model = auto_arima(train, start_p=1, start_q=1, max_p=3, max_q=3, max_d=2,suppress_warnings=True)
    model.fit(train)
    trace=True,
    error_action='ignore',
    forecast = model.predict(n_periods=len(valid))
    print(forecast,valid)
    l.extend(list(forecast))
    th += 1
print(l)
print(len(l))
#plot the predictions for validation set
plt.figure(figsize=(10, 6))
plt.plot(data[th:], label='Valid')
plt.plot(l, label='Prediction')
plt.show()


