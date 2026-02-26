import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

'''
DEV NOTE:

- You must have the summary.csv file created by SummaryProgram.py

- Equation under analysis is:
     (f**2 + lambda**2)V = (2 A**2 gamma P)/(m)

If define: (f**2 + lambda**2)V = Phi

So: gamma = (Phi m)/(2 A**2 P)

TO DO:
- Get Pressure error equation correct 
- Account from \pm 2 ml in volume measurement
'''

'''Constants'''
DIAM = np.mean([30.98E-3, 30.96E-3, 30.90E-3]) #m
DERR = 0.02E-3/np.sqrt(3) #m
MASS = 85.67E-3 #kg
MERR = 0.01E-3 #kg

AREA = np.pi*(DIAM**2)/4
ARER = np.pi*DIAM*DERR/2
PRES = MASS*9.81/AREA
PERR = np.sqrt( ( (9.81 * MERR)/(AREA) )**2 + ( -(9.81 * MASS * ARER)/(AREA**2) )**2 )

'''Functions'''
def GetPhi(freq, lamb, volu):
    '''The aim if for this function to be constant as (2 A**2 gamma P)/(m)'''
    return (freq**2 + lamb**2) * volu


'''Data cleaning'''
df = pd.read_csv('summary.csv')

nit = df[df['Gas'] == 'nitrogen'].copy()
hel = df[df['Gas'] == 'helium'].copy()
co2 = df[df['Gas'] == 'carbonDioxide'].copy()

nitPhi = GetPhi(nit['Freq'], nit['Lambda'], nit['Volume'])
helPhi = GetPhi(hel['Freq'], hel['Lambda'], hel['Volume'])
co2Phi = GetPhi(co2['Freq'], co2['Lambda'], co2['Volume'])


'''Ploting histogram'''
def PlotHistograms(nitPhi, helPhi, co2Phi, nitBin=round(np.sqrt(len(nitPhi))), helBin=round(np.sqrt(len(helPhi))), co2Bin=round(np.sqrt(len(co2Phi)))):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 4))

    axes[0][0].hist(nitPhi, bins=nitBin, density=True, color='red', edgecolor='black', alpha=0.8)
    mu, std = norm.fit(nitPhi)
    xmin, xmax = np.min(nitPhi), np.max(nitPhi)
    x = np.linspace(xmin, xmax, nitBin*5)
    axes[0][0].plot(x, norm.pdf(x, mu, std), 'k', linewidth=2)
    axes[0][0].text(0.5, 0.8, f'Mean Value: {mu}\nStandard Deviation:{std}', horizontalalignment='center', transform=axes[0][0].transAxes, fontsize=8)
    axes[0][0].set_title('Nitrogen Phi value')
    gamma, gerr = GetGamma(mu, std)
    print(f"- Nitrogen\n\tMean: {round(mu)}\n\tStDv: {round(std)}\n\tGamma: {round(gamma)}\n\tGamma error: {round(gerr)}\n")

    axes[0][1].hist(helPhi, bins=helBin, density=True, color='orange', edgecolor='black', alpha=0.8)
    mu, std = norm.fit(helPhi)
    xmin, xmax = np.min(helPhi), np.max(helPhi)
    x = np.linspace(xmin, xmax, helBin*5)
    axes[0][1].plot(x, norm.pdf(x, mu, std), 'k', linewidth=2)
    axes[0][1].text(0.5, 0.8, f'Mean Value: {mu}\nStandard Deviation:{std}', horizontalalignment='center', transform=axes[0][1].transAxes, fontsize=8)
    axes[0][1].set_title('Helium Phi value')
    gamma, gerr = GetGamma(mu, std)
    print(f"- Helium\n\tMean: {round(mu)}\n\tStDv: {round(std)}\n\tGamma: {round(gamma)}\n\tGamma error: {round(gerr)}\n")

    axes[1][0].hist(co2Phi, bins=co2Bin, density=True, color='green', edgecolor='black', alpha=0.8)
    mu, std = norm.fit(co2Phi)
    xmin, xmax = np.min(co2Phi), np.max(co2Phi)
    x = np.linspace(xmin, xmax, co2Bin*5)
    axes[1][0].plot(x, norm.pdf(x, mu, std), 'k', linewidth=2)
    axes[1][0].text(0.5, 0.8, f'Mean Value: {mu}\nStandard Deviation:{std}', horizontalalignment='center', transform=axes[1][0].transAxes, fontsize=8)
    axes[1][0].set_title('Carbon-Dioxide Phi value')
    gamma, gerr = GetGamma(mu, std)
    print(f"- Carbon Dioxide\n\tMean: {round(mu)}\n\tStDv: {round(std)}\n\tGamma: {round(gamma)}\n\tGamma error: {round(gerr)}\n")
    axes[1][1].axis('off')

    for ax in axes.flatten():
        ax.set_xlabel('Phi Value')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def GetGamma(Phi, PhiErr):
    gamma = (MASS * Phi)/(2 * AREA**2 * PRES)
    gam_err_squ = ( (Phi * MERR)/(2 * AREA**2 * PRES) )**2 + ( (MASS * PhiErr)/(2 * AREA**2 * PRES) )**2 +( -(MASS * Phi * ARER)/( AREA**3 * PRES) )**2 +( -(MASS * Phi * PERR)/(2 * AREA**2 * PRES) )**2 
    gamErr = np.sqrt(gam_err_squ)
    return gamma, gamErr

'''Plot and Narrow data to show trend value'''
PlotHistograms(nitPhi, helPhi, co2Phi) #This can be made better by removing data above 100,000

nitPhi = nitPhi[nitPhi <= 100000]
helPhi = helPhi[helPhi <= 100000]
co2Phi = co2Phi[co2Phi <= 100000]
PlotHistograms(nitPhi, helPhi, co2Phi) #This can be made better by removing data above 32,500

nitPhi = nitPhi[nitPhi <= 32500]
helPhi = helPhi[helPhi <= 32500]
co2Phi = co2Phi[co2Phi <= 32500]
PlotHistograms(nitPhi, helPhi, co2Phi)#, nitBin=50, helBin=50, co2Bin=50)

nitPhi = nitPhi[nitPhi <= 20000]
helPhi = helPhi[helPhi <= 29000]
co2Phi = co2Phi[co2Phi <= 21000]
PlotHistograms(nitPhi, helPhi, co2Phi, nitBin=25, helBin=25, co2Bin=25)
