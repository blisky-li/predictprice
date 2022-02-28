from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

def func(x, a, b):
    return a * np.exp(b * x)

xdata = np.array([1,2,3,4])
ydata = np.array([1,3.11,9.75,30.54])
plt.plot(xdata,ydata,'b-')
plt.show()
popt, pcov = curve_fit(func, xdata, ydata)
#popt数组中，三个值分别是待求参数a,b,c
y2 = [func(i, popt[0],popt[1]) for i in xdata]
plt.plot(xdata,y2,'r--')
print(popt)
plt.show()
