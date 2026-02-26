import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit

#Just provides the folder name and output file name
#I'll update this when more folders become used



dataGroups = ['data/nitrogen/', 'data/helium/', 'data/carbonDioxide/']
outputCSV = 'summary.csv'

def DecayCos(x, amp, lambd, fre):
    '''Still the model equation'''
    return amp * np.exp(-lambd * x) * np.cos(2 * np.pi * fre * x)

def fit_file(path):
    '''This function does the same as the "IndividualPlot.py" program bar plotting the data. Also it provides error'''
    df = pd.read_csv(path, skiprows=[1])
    dfx = df['x-axis']
    dfy = df['1']

    peakIndex = np.argmax(dfy)
    peakTime = dfx[peakIndex]

    timeStart = peakTime
    timeEnd = peakTime + 0.4

    mask = (dfx >= timeStart) & (dfx <= timeEnd)

    xWindow = dfx[mask].to_numpy()
    xWindow = xWindow - xWindow[0]

    yWindow = dfy[mask].to_numpy()

    def DecayCos(x, Amp, lambd, freq):
        return Amp * np.exp(-lambd * x) * np.cos(2 * np.pi * freq * x)

    AmpGuess = yWindow.max()
    LamGuess = 9  
    FreGuess = 22  
    p0=[AmpGuess, LamGuess, FreGuess]

    try:
        params, covariance = curve_fit(DecayCos, xWindow, yWindow, p0=p0)
        _, lam, fre = params
        lamErr = np.sqrt(covariance[1,1])
        freErr = np.sqrt(covariance[2,2])
    except RuntimeError:
        #On fit fail
        lam, fre = np.nan, np.nan
        lamErr, freErr = np.nan, np.nan
    return lam, fre, lamErr, freErr 

#Holds the data before exporting (probs takes loads of memory sry)
results = []

print("Waiting...")

# Loop over all CSV files in the folder
for folder in dataGroups:
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            path = os.path.join(folder, file)
            # Extract 4th and 5th characters from file name
            vol = file[3:5]
            lam, fre, lamErr, freErr = fit_file(path)
            folderName = folder[5:-1]
            results.append({'Gas':folderName, 'Volume': vol, 'Lambda': lam,  'LambdaError': lamErr, 'Freq': fre, 'FreqError': freErr})
            print(".",end="")

# Convert results to DataFrame and save
dfResults = pd.DataFrame(results)
dfResults.to_csv(outputCSV, index=False)

print("\nIt is done :3")
