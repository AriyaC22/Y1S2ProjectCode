import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#Nit 1.40
#Hel 1.66
#CO2 1.28
 
df = pd.read_csv('summary.csv')

df.rename(columns={'Gas':'gas', 'Volume':'vol', 'Lambda':'lam', 'LambdaError':'lamEr', 'Freq':'fre', 'FreqError':'freEr'}, inplace=True)

df['f0^2'] = df['fre']**2 + (df['lam']/(2*np.pi))**2
df['f0^2Er'] = np.sqrt( (2 * df['fre'] * df['freEr'])**2 + ((df['lam'] * df['lamEr'])/(2 * np.pi**2 ))**2 )
df['1/v'] = 1/(df['vol']*1e-6) 

def model(V, slope):
    return slope*V

nit = df[df['gas'] == 'nitrogen'].copy() #Only nitrogen data
hel = df[df['gas'] == 'helium'].copy() #Only helium data
co2 = df[df['gas'] == 'carbonDioxide'].copy() #Only carbon Dioxide data

nit = nit[(nit['f0^2'] >= (466/25000) * nit['1/v'])]
hel = hel[(hel['f0^2'] >= (530/25000) * hel['1/v'])]
co2 = co2

paramN, covarN = curve_fit(model, nit['1/v'], nit['f0^2'], sigma=nit['f0^2Er'], absolute_sigma=True)
paramH, covarH = curve_fit(model, hel['1/v'], hel['f0^2'], sigma=hel['f0^2Er'], absolute_sigma=True)
paramC, covarC = curve_fit(model, co2['1/v'], co2['f0^2'], sigma=co2['f0^2Er'], absolute_sigma=True)

MASS = 92.64e-3
MASSER = 0.1e-3

DIAM = 31.0e-3
DIAMER = 0.1e-3

AREA = np.pi * (DIAM/2)**2
AREAER = (np.pi/2) * DIAM * DIAMER

PRES = ((MASS * 9.81)/AREA ) + 101325
PRESER = np.sqrt( ((9.81 * MASSER)/(AREA))**2 + ((9.81 * MASS * AREAER)/(AREA**3))**2 )


gammaN = (4 * np.pi**2 * MASS * paramN)/(AREA**2 * PRES)
gammaErN = gammaN * ( (MASSER/MASS) + (covarN[0][0]/paramN) + 2*(AREAER/AREA) + (PRESER/PRES) )

gammaH = (4 * np.pi**2 * MASS * paramH)/(AREA**2 * PRES)
gammaErH = gammaH * ( (MASSER/MASS) + (covarH[0][0]/paramH) + 2*(AREAER/AREA) + (PRESER/PRES) )

gammaC = (4 * np.pi**2 * MASS * paramC)/(AREA**2 * PRES)
gammaErC = gammaC * ( (MASSER/MASS) + (covarC[0][0]/paramC) + 2*(AREAER/AREA) + (PRESER/PRES) )

print(round(gammaN[0],2), round(gammaErN[0],2))
print(round(gammaH[0],2), round(gammaErH[0],2))
print(round(gammaC[0],2), round(gammaErC[0],2))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 4))

axes[0][0].set_title("Nitrogen")
axes[0][1].set_title("Helium")
axes[1][0].set_title("Carbon Dioxide")

x = np.linspace(np.min(nit['1/v']), np.max(nit['1/v']), 1000)
y = model(x, paramN)
axes[0][0].plot(x, y, color='red', linewidth=1, zorder=2)
axes[0][0].errorbar(nit['1/v'], nit['f0^2'], fmt='.', color='blue', zorder=1)

x = np.linspace(np.min(hel['1/v']), np.max(hel['1/v']), 1000)
y = model(x, paramH)
axes[0][1].plot(x, y, color='red', linewidth=1, zorder=2)
axes[0][1].errorbar(hel['1/v'], hel['f0^2'], fmt='.', color='blue', zorder=1)

x = np.linspace(np.min(co2['1/v']), np.max(co2['1/v']), 1000)
y = model(x, paramC)
axes[1][0].plot(x, y, color='red', linewidth=1, zorder=2)
axes[1][0].errorbar(co2['1/v'], co2['f0^2'], fmt='.', color='blue', zorder=1)

axes[1][1].axis('off')
entries = [f"Nitrogen:\n$\\gamma$ = {gammaN[0]:.3f} ± {gammaErN[0]:.3f}", f"Helium:\n$\\gamma$ = {gammaH[0]:.3f} ± {gammaErH[0]:.3f}", f"Carbon Dioxide:\n$\\gamma$ = {gammaC[0]:.3f} ± {gammaErC[0]:.3f}"] 

for i in range(0, len(entries), 4):
    axes[1][1].text(0.05 + 0.45*(i//4), 0.95, "\n\n".join(entries[i:i+4]), transform=axes[1][1].transAxes,va='top', fontsize=10)

for ax in axes.flatten():
    ax.set_xlabel('$Volume^{-1}$')
    ax.set_ylabel('Resonant frequency, $f_0$ (Hz)')

plt.tight_layout()
plt.show()