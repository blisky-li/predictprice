import xlrd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.array([0.016,0.018,0.02,0.022,0.024])
y = np.array([540774,315728,215487,113655,80216])
plt.plot(x,y)
plt.ylim(0)
plt.xlabel("transaction fee")
plt.ylabel("return on investment/dollars")
plt.title("Sensitivity Analysis on Transaction Fees of bitcoin")
plt.show()