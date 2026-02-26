import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

'''
DEV NOTE:

You must have the summary.csv file 
'''

def inverse(V, m, c):
    '''Model equation for an inverse relationship'''
    return (m/V) + c

def Linear(V, m, c):
    '''Model equation for a linear relationship'''
    return (m*V) + c

#Load file
df = pd.read_csv('summary.csv')

#Create 3 dataframes, one for each gas
nit = df[df['Gas'] == 'nitrogen'].copy()
hel = df[df['Gas'] == 'helium'].copy()
co2 = df[df['Gas'] == 'carbonDioxide'].copy()

#Select gas to look at
UsedGas = nit

x = UsedGas['Volume']
y = UsedGas['Freq']

#Remove data which doesn't fit trend
xIndex = x.idxmin()
yValue = y.loc[xIndex]
UsedGas = UsedGas[UsedGas['Freq'] <= yValue]

x = UsedGas['Volume']
y = UsedGas['Freq']

# # dfResults = pd.DataFrame(x)
# # dfResults.to_csv('debugging.csv', index=False)

#Find inverse curve of best fit
params, _ = curve_fit(inverse, x, y, p0=[514, 11])
m, c = params
xFunc1 = np.linspace(x.min(), x.max(), 1000)
yFunc1 = inverse(xFunc1, m, c)

#Find linear line of best fit
params, _ = curve_fit(Linear, x, y, p0=[-0.17, 31])
m, c = params
xFunc2 = np.linspace(x.min(), x.max(), 1000)
yFunc2 = Linear(xFunc2, m, c)

#Plot LoBFs
plt.plot(xFunc1, yFunc1, color='r')
plt.plot(xFunc2, yFunc2, color='b')

#Plot data points
plt.scatter(UsedGas['Volume'], UsedGas['Freq'])

plt.show()