#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:15:01 2021

@author: leonardobossi1
"""

import numpy as np
import matplotlib as mpl
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def linear (x, a, b):
    y = a*x + b
    return y

def csenergy(angle, e0): 
    energymeasure = e0 / ( 1 + (e0 / 511) * (1 - np.cos(np.deg2rad(angle))))
    
    return energymeasure

def errorprop(a, b, siga, sigb, value):
    asserror = value * np.sqrt(siga / a ** 2 + sigb/ b ** 2)

    return asserror

data = np.loadtxt("Experiment_Data.csv", delimiter = ",", skiprows= 7, unpack = True )
data2 = np.genfromtxt("Experiment_Data.csv", delimiter = ",", max_rows= 7,skip_header=1, dtype="str")

# Reading the data (from provided data ile)
channel,nosource,na22calib,mn54calib,cs137calib,am241calib,c20cs137,b20cs137,c30cs137,b30cs137,c45cs137,b45cs137 = data

# Removing the background noise
na22calib -= nosource
mn54calib -= nosource
cs137calib -= nosource
am241calib -= nosource

# Energy peaks for the different radioactive sources
na22e = [511.006, 1274.542]
mn54e = 834.838
cs137e = [33, 661.657]
am241e = 59

nachann1 = np.argmax(na22calib)
nachann2 = np.argmax(na22calib[200:]) + 200
mnchann = np.argmax(mn54calib)
cschann1 = np.argmax(cs137calib)
cschann2 = np.argmax(cs137calib[:100])
amchann = np.argmax(am241calib)

channels = [nachann1, nachann2, mnchann, cschann1,cschann2,amchann]
energies = [na22e[0], na22e[1], mn54e, cs137e[1], cs137e[0], am241e]

popt, pcov = curve_fit(linear, channels, energies)

yvals = list(map(lambda x: linear(x, *popt), channel))

# Calibrating the channels to energy values using the peaks 
plt.plot(channel, yvals, label = str(popt[0]) + "x + " + str(popt[1]))
plt.scatter(channels, energies)
plt.title("Calibration of Energy against Channel")
plt.ylabel("Energy (keV)")
plt.xlabel("Channel")
plt.grid()
plt.legend()

plt.figure()
plt.scatter(channel, na22calib)
plt.title("Spectrum of Na-22")
plt.xlabel('Bin no.')
plt.ylabel('Total counts')
plt.grid()

plt.figure()
plt.scatter(channel, mn54calib)
plt.title("Spectrum of Mn-54")
plt.xlabel('Bin no.')
plt.ylabel('Total counts')
plt.grid()

plt.figure()
plt.scatter(channel, cs137calib)
plt.title("Spectrum of Cs-137")
plt.xlabel('Bin no.')
plt.ylabel('Total counts')
plt.grid()

plt.figure()
plt.scatter(channel, am241calib)
plt.title("Spectrum of Am-241")
plt.xlabel('Bin no.')
plt.ylabel('Total counts')
plt.grid()


#%%

# Subtracting the scattered data from the respective background measurements
cs20 = c20cs137-b20cs137
cs30 = c30cs137-b30cs137
cs45 = c45cs137-b45cs137

# Calculating the transmission ratio
r20 = []
for i in range(len(c20cs137)):
    if b20cs137[i] != 0:
        value = c20cs137[i]/b20cs137[i]
    elif b20cs137[i] == 0:
        value = 0
    r20.append(value)


r30 = []
for i in range(len(c30cs137)):
    if b30cs137[i] != 0:
        value = c30cs137[i]/b30cs137[i]
    elif b30cs137[i] == 0:
        value = 0
    r30.append(value)

r45 = []
for i in range(len(c45cs137)):
    if b45cs137[i] != 0:
        value = c45cs137[i]/b45cs137[i]
    elif b45cs137[i] == 0:
        value = 0
    r45.append(value)
   

#expected energy responses
expected20 = csenergy(20, cs137e[1])
expectedchannel20 = (expected20 - popt[1])/popt[0]
expected30 = csenergy(30, cs137e[1])
expectedchannel30 = (expected30 - popt[1])/popt[0]
expected45 = csenergy(45, cs137e[1])
expectedchannel45 = (expected45 - popt[1])/popt[0]

plt.figure()
plt.scatter(channel, cs137calib)
#plt.scatter(channel, nosource, label='background')# angle = 20 deg
plt.axvline(cschann1,color = "r", label = ("Value from compton eq."))
plt.title("Cs-137 at 0 degrees of incidence")
plt.legend()
plt.xlabel("Channel")
plt.ylabel("Counts")
plt.grid()
plt.show()

plt.figure()
plt.scatter(channel, r20, label='Transmission ratio')
#plt.scatter(channel, c20cs137)
#plt.scatter(channel, b20cs137, label='background')# angle = 20 deg
plt.axvline(expectedchannel20,color = "r", label = ("Value from compton eq."))
plt.title("Cs-137 at 20 degrees of incidence")
plt.legend()
plt.xlabel("Channel")
plt.ylabel("Ratio")
plt.grid()
plt.show()

plt.figure()
plt.scatter(channel, cs30, label='scatter data - background')
#plt.scatter(channel, c30cs137)
#plt.scatter(channel, b30cs137  , label='background')# angle = 30 deg
plt.axvline(expectedchannel30, color = "r", label = ("Value from compton eq."))
plt.title("Cs-137 at 30 degrees of incidence")
plt.legend()
plt.xlabel("Channel")
plt.ylabel("Total counts")
plt.grid()
plt.show()

plt.figure()
plt.scatter(channel, cs45, label='scatter data - background')
#plt.scatter(channel, c45cs137)
#plt.scatter(channel, b45cs137, label = 'background')  # angle = 40
plt.axvline(expectedchannel45,color = "r", label = ("Value from compton eq."))
plt.title("Cs-137 at 45 degrees of incidence")
plt.legend()
plt.xlabel("Channel")
plt.ylabel("Total counts")
plt.grid()
plt.show()

maxenergy20 = popt[0]*np.argmax(cs20) + popt[1]
maxenergy30 = popt[0]*np.argmax(cs30) + popt[1]
maxenergy45 = popt[0]*np.argmax(cs45) + popt[1]


#%%
# Investigating and simulating the difference in peak values for different scattering
# angles    

plt.figure()
plt.scatter(channel, cs30, label='30 degree scattering angle')
#plt.scatter(channel, c30cs137)
#plt.scatter(channel, b30cs137  , label='background')# angle = 30 deg
plt.title("Cs-137 at 30 degrees of incidence")


plt.scatter(channel, cs20, color = 'r', label='20 degree scattering angle')
#plt.scatter(channel, c45cs137)
#plt.scatter(channel, b45cs137, label = 'background')  # angle = 40
plt.title("Cs-137 at 20 degrees of incidence")
plt.legend()
plt.xlabel("Channel")
plt.ylabel("Total counts")
plt.grid()

import MC as mc




    




