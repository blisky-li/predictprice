import numpy as np #导入包
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.api as sm
import xlrd
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('b2.csv',encoding='utf-8',index_col='date')
#print(df)
from statsmodels.tsa import stattools
ts = df['return']
print(ts[:1])
from statsmodels.tsa.stattools import adfuller #ADF单位根检
from arch import arch_model
import math
n = 4
l = []
l2 = []
k = [[1,1]]
s = []
b = []
r = []
for i in k:
    while n < len(df['return']):

        try:
            basic_gm = arch_model(df['return'][:n], p = 1, q = 1, mean ='constant', vol ='GARCH', dist ='normal')
            gm_result = basic_gm.fit(disp ='off')
            #print(gm_result.summary())
            gm_forecast = gm_result.forecast(horizon = 1)
            a = gm_forecast.variance[-1:].values[0][0]
            #print(gm_forecast.variance[-1:])
            q_parametric = basic_gm.distribution.ppf(0.05, None)
            #print('5% parametric quantile: ', q_parametric)
            l.append(a)
            mean = df['predict'][n+1:n+2].values/df['price'][n:n+1].values  - 1

            # Calculate the VaR
            VaR_parametric = math.sqrt(a) * q_parametric
            r.append(mean[0])
            b.append(VaR_parametric)
            t = mean+VaR_parametric
            s.append(t[0])
            print(q_parametric,VaR_parametric,-mean+VaR_parametric)
            n += 1
        except:
            n+=1
            f = r[-1]
            r.append(f)
            l.append(0)
            ls = s[-1]
            ls2 = b[-1]
            b.append(ls2)
            s.append(ls)
            print(ls)
print(s)

plt.figure(figsize=(10, 6))
#plt.plot(r,color='red')
plt.plot(b,color='red')
#plt.plot(s,color='blue')
plt.show()
# Save VaR in a DataFrame
#VaR_parametric = pd.DataFrame(VaR_parametric, columns=['5%'], index=variance_forecast.index)

# Plot the VaR
#plt.plot(VaR_parametric, color='red', label='5% Parametric VaR')
