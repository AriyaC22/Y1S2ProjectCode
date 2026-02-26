import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

'''
DEV NOTE:
1. The plotting of this could be much better but I couldnt be bothered to imporve it as we aren't plotting this anyways
'''

filePath = 'data/nitrogen/nit44______8.csv'

# Read CSV and skip the first row (first line is always text)
df = pd.read_csv(filePath, skiprows=[1])

#Define x and y axis
dfx = df['x-axis'] #time
dfy = df['1']  #Voltage

#Find the time value of the peak voltage measurement
peakIndex = np.argmax(dfy)
peakTime = dfx[peakIndex]

# Assumption: more than one oscilations occurs within 0.4s (aka frequency > 2.5 Hz)
timeStart = peakTime
timeEnd = peakTime + 0.4

#Select data between PeakTime and 0.4s ahead
mask = (dfx >= timeStart) & (dfx <= timeEnd)

# Set start of x data to t=0 to stop bugs with curve_fit
#Also use numpy arrays bc easier
xWindow = dfx[mask].to_numpy()
xWindow = xWindow - xWindow[0]

yWindow = dfy[mask].to_numpy()

#Define model function of curve fit
def DecayCos(x, Amp, lambd, freq):
    return Amp * np.exp(-lambd * x) * np.cos(2 * np.pi * freq * x)

#Provide guess values so function doesn't RunTimeError
AmpGuess = yWindow.max()
LamGuess = 9  
FreGuess = 22  
p0=[AmpGuess, LamGuess, FreGuess] #Guess parameters

#Fit function to model equation
params, _ = curve_fit(DecayCos, xWindow, yWindow, p0=p0)
amp, lam, freq = params

#Make function into data points
xFunc = np.linspace(xWindow.min(), xWindow.max(), 1000)
yFunc = DecayCos(xFunc, amp, lam, freq)

#Plot
fig, ax = plt.subplots()

ax.plot(xWindow, yWindow)
ax.plot(xFunc, yFunc)

#Don't need this but nice to see
print(f"A: {amp}\nlambda: {lam}\nfreq: {freq}")


ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (V)')
ax.set_title('Plotting a model of a damped oscillator to the data collected on an oscilloscope')
ax.legend(["Oscilloscope reading", "Model of damped Oscillator"], loc="upper right")
ax.text(0.2, 0.98, f"$\\lambda$: {lam:.3f}\nf: {freq:.3f}", transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')

plt.show()