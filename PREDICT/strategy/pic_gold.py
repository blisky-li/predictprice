import xlrd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.array([0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16])
y = np.array([1131,1028,876,758,646,552,516,449])
plt.plot(x,y)
plt.ylim(0)
plt.xlabel("transaction fee")
plt.ylabel("return on investment/dollars")
plt.title("Sensitivity Analysis on Transaction Fees of gold")
plt.show()
