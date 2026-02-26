import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

'''
DEV NOTE:

- You must have the summary.csv file created by SummaryProgram.py

- Equation under analysis is:
    f^2 + lambda^2 = (A^2 * gamma * P / m)(1/[V+sigma])

- Rearranged equation:
    1/(f^2 + lambda^2)= V/C + sigma/C
    Where C = (A^2 gamma P)/(m)

Readers of code note:
1. Any errors calculated using an equation that looks like:
        np.sqrt( ( ()/() )**2 + ... )
    Is found using the propagation of errors for many variables. This equation can be found in the lab moodle under that title.
    
TO DO:
+ Account for error pm 2ml Volume.
+ Remove data points less than Lowest y at 40ml.
+ Find gamma from the gradient
- Implement error of freq and lambda (you idiot)
- comment your code (you idiot again)
'''

'''Constants'''
DIAM:float = np.mean([30.98E-3, 30.96E-3, 30.90E-3]) #m
DERR:float = 0.02E-3/np.sqrt(3) #m
MASS:float = np.mean([85.67E-3, 95.25E-3]) #kg
MERR:float = 0.01E-3/np.sqrt(2) #kg

AREA:float = np.pi*(DIAM**2)/4 #m^2
ARER:float = np.pi*DIAM*DERR/2 #m^2
PRES:float = MASS*9.81/AREA + 101000#N m^-2
PERR:float = np.sqrt( ( (9.81 * MERR)/(AREA) )**2 + ( -(9.81 * MASS * ARER)/(AREA**2) )**2 )#N m^-2


print(DIAM, MASS, AREA, PRES)
print(DERR, MERR, ARER, PERR)

'''Functions'''
def GetXY(gasDF:pd.DataFrame) -> pd.DataFrame:
    '''Create a Dataframe from a copy of the summary file with columns x, y, x-error, and y-error'''
    df = pd.DataFrame({
        "x":gasDF['Volume']*1E-6,
        "y":1/((gasDF['Freq'])**2 + ((gasDF['Lambda'])/(2 * np.pi))**2),
        "xerr": 2E-6,
        "yerr": np.sqrt( ( (2 * gasDF['Freq'] * gasDF['FreqError'])/(gasDF['Freq']**2 + gasDF['Lambda']**-2)**2 )**2 + ( (2 * gasDF['Lambda']**-3 * gasDF['LambdaError'])/(gasDF['Freq']**2 + gasDF['Lambda']**-2)**2 ) )
    })

    # This removes y values below the lowest y value at x = 40 
    xIndex = df["x"].idxmin()
    yValue = df["y"].loc[xIndex]
    df = df[df['y'] >= yValue]

    df = df[df['yerr'] < 0.0005]
    return df

def ModleEquation(x:any, C:any, s:any) -> any:
    '''This it the model equation for scipy.optimize.curve_fit'''
    return (x*C) + s

def Getgamma(C:float, cErr:float) -> tuple[float, float]:
    '''Returns the gamma value and error in gamma from the C value and its error'''
    gamma = ( 4 * np.pi**2 * MASS)/(AREA**2 * PRES* C)
    gamEr = np.sqrt( ( (MASS * cErr)/(AREA**2 * PRES) )**2 + ( (C * MERR)/(AREA**2 * PRES) )**2 + ( (-2 * C * MASS * ARER)/(AREA**3 * PRES) )**2 + ( (-1 * C * MASS * PERR)/(AREA**2 * PRES**2) )**2 )
    return gamma, gamEr

def PlotGas(title:str, dframe:pd.DataFrame, Fig, dicName:str=''):
    '''Will plot the x and y data points with their errors as well as a line of best fit, ultimately providing the gamma and error in gamma value in the console.'''
    #This means additions to the dictionary are globally made
    global data2Calculate

    # Allows a dictionary name to be specified to prevent over-writing data
    if dicName == '': dicName = title 

    #Titles the Figure, only relevant to the last instance a given axes is called on.
    Fig.set_title(title)

    # Find the line of best fit and its errors for the data (and constants)
    params, covariance = curve_fit(ModleEquation, dframe['x'], dframe['y'])
    c, s = params
    cErr, sErr = covariance[0][0], covariance[1][1]

    #Plot line of best fit between the range of the x values
    xFunc = np.linspace(np.min(dframe['x']), np.max(dframe['x']), 1000)
    yFunc = ModleEquation(xFunc, c, s)
    Fig.plot(xFunc, yFunc, color='red', linewidth=3, zorder=2)

    #Plot the data itself on the same axes
    # Fig.errorbar(dframe['x'], dframe['y'], xerr=dframe['xerr'], yerr=dframe['yerr'], fmt='o', color='blue', zorder=1)
    Fig.errorbar(dframe['x'], dframe['y'], fmt='o', color='blue', zorder=1)

    # Use the values to return and print the gamma values
    gamma, gamErr = Getgamma(c, cErr)

    # Add all found values to the dictionary
    data2Calculate[dicName] = [dframe['x'], dframe['y'], c, s, gamma, cErr, sErr, gamErr]

    print(f"{dicName}:\n\t>>>Gamma:{round(gamma,3)}\n\t>>>Gamma± {round(gamErr,3)}")

def PlotGammaInfo():
    '''After using a number of PLotGas Methods use this to summerise the gamma data in axes[1][1]'''
    #Remove the graph outline in axes
    axes[1][1].axis('off')

    #Create a list of strings in the wanted format of the dictionary data
    entries = [] 
    for key, values in data2Calculate.items(): 
        gamma = values[4] 
        gammaErr = values[7] 
        entries.append(f"{key}:\nγ = {gamma:.3f} ± {gammaErr:.3f}") 

    #Settings for the formatting
    rowPerCol = 4
    colSpacing = 0.45 
    columns = [ entries[i:i + rowPerCol] for i in range(0, len(entries), rowPerCol) ] 
    
    # Plot the columns according to their index / entry
    for col_index, col_entries in enumerate(columns): 
        textstr = "\n\n".join(col_entries) 
        axes[1][1].text(0.05 + col_index * colSpacing, 0.95, textstr, transform=axes[1][1].transAxes, verticalalignment='top', fontsize=10)

    data2Calculate.clear()


def SplitDF(XYdf: pd.DataFrame, mSlpit:float, cSplit:float) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Will split an x and y dataframe into 2 along the line y = mx + c'''

    # Plot the vlaue of y for the given line with each x value
    XYdf['lineY']= (mSlpit*XYdf['x'] )+ cSplit

    #create data frames for above and below the y value of the same row.
    abovedf = XYdf[XYdf['y']>=XYdf['lineY']]
    belowdf = XYdf[XYdf['y']<XYdf['lineY']]

    return abovedf, belowdf

'''Data cleaning'''
df = pd.read_csv('summary.csv') #file name to read

nit = df[df['Gas'] == 'nitrogen'].copy() #Only nitrogen data
hel = df[df['Gas'] == 'helium'].copy() #Only helium data
co2 = df[df['Gas'] == 'carbonDioxide'].copy() #Only carbon Dioxide data

nitXY = GetXY(nit) #Convert nitrogen df into an XY df
helXY = GetXY(hel) #Convert helium df into an XY df
co2XY = GetXY(co2) #Convert CO2 df into an XY df

data2Calculate = dict() #Dictionary to hold data for each gas

'''Plotting Scatter Graph'''
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 4)) #Define the plt.FIgure

x = np.linspace(4E-5, 8E-5, 1000)

PlotGas("Nitrogen", nitXY, axes[0][0]) #Plot nitrogen on top left axes
# axes[0][0].plot(x, (1.864E5/4021)*x + (1.4E5/4.021E8))
PlotGas("Helium", helXY, axes[0][1]) #Plot helium on top right axes
PlotGas("Carbon Dioxide", co2XY, axes[1][0]) #Plot co2 on bottom left axes
PlotGammaInfo()

for ax in axes.flatten():
    ax.set_xlabel('Volume')
    ax.set_ylabel('Inverse (freq^2 + decay^-2)')

plt.tight_layout()
fig.canvas.manager.window.showMaximized()
plt.show()


'''DEV NOTE 2:
- Nitrogen seems to have 2 lines split with a line in the points:
    (3.991E-5, 0.002108) and (8.012E-5, 0.003972). So, m = 46.35662770455111, c = 0.0002579069883113653
- Nitrogen data is messy below the line:
    (4.000E-5, 0.001839) and (7.999E-5, 0.003593). So, m = 43.86096524131033, c = 8.456139034758643e-05

- Helium seems to have 2 lines split with a line in the points:
    (4.213E-5, 0.001988) and (8E-5, 0.003707). So, m = 45.392130974386035, c = 7.562952204911663e-05

Carbon Dioxide seems to have 1 clear line so no changes needed.'''


'''Data Cleaning 2'''
nitTop, nitBot = SplitDF(nitXY, 46.35662770455111, 0.0002579069883113653)
nitBot, _ = SplitDF(nitBot, 43.86096524131033, 8.456139034758643e-05)
helTop, helBot = SplitDF(helXY, 45.392130974386035, 7.562952204911663e-05)

'''Plotting Scatter Graph'''
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 4))

PlotGas("Nitrogen", nitTop, axes[0][0], dicName="Nitrogen Top")
PlotGas("Nitrogen", nitBot, axes[0][0], dicName="Nitrogen Bottom")
PlotGas("Helium", helTop, axes[0][1], dicName="Helium Top")
PlotGas("Helium", helBot, axes[0][1], dicName="Helium Bottom")
PlotGas("Carbon Dioxide", co2XY, axes[1][0])
PlotGammaInfo()

for ax in axes.flatten():
    ax.set_xlabel('Volume')
    ax.set_ylabel('Inverse (freq^2 + decay^-2)')

plt.tight_layout()
fig.canvas.manager.window.showMaximized()
plt.show()

