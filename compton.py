import numpy as np
import matplotlib as mpl
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def linear (x, a, b):
    y = a*x + b
    return y

def csenergy(angle, e0):
    energymeasure = e0/(1+(e0/511)*(1 - np.cos(np.deg2rad(angle))))
    return energymeasure

data = np.loadtxt("Experiment_Data.csv", delimiter = ",", skiprows= 7, unpack = True )
data2 = np.genfromtxt("Experiment_Data.csv", delimiter = ",", max_rows= 7,skip_header=1, dtype="str")

channel,nosource,na22calib,mn54calib,cs137calib,am241calib,c20cs137,b20cs137,c30cs137,b30cs137,c45cs137,b45cs137 = data

na22calib -= nosource
mn54calib -= nosource
cs137calib -= nosource
am241calib -= nosource

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

plt.plot(channel, yvals, label = str(popt[0]) + "x + " + str(popt[1]))
plt.scatter(channels, energies)
plt.title("Calibration of Energy against Channel")
plt.ylabel("Energy (keV)")
plt.xlabel("Channel")
plt.grid()
plt.legend()
plt.show()

plt.scatter(channel, na22calib)
plt.title("Spectrum of Na-22")
plt.xlabel("Channel")
plt.ylabel("Counts")
plt.show()

plt.scatter(channel, mn54calib)
plt.title("Spectrum of Mn-54")
plt.xlabel("Channel")
plt.ylabel("Counts")
plt.show()

plt.scatter(channel, cs137calib)
plt.title("Spectrum of Cs-137")
plt.xlabel("Channel")
plt.ylabel("Counts")
plt.show()

plt.scatter(channel, am241calib)
plt.title("Spectrum of Am-241")
plt.xlabel("Channel")
plt.ylabel("Counts")
plt.show()

#%%

cs20 = c20cs137-b20cs137
cs30 = c30cs137-b30cs137
cs45 = c45cs137-b45cs137

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
   

#expected energyresponses
expected20 = csenergy(20, cs137e[1])
expectedchannel20 = (expected20 - popt[1])/popt[0]
expected30 = csenergy(30, cs137e[1])
expectedchannel30 = (expected30 - popt[1])/popt[0]
expected45 = csenergy(45, cs137e[1])
expectedchannel45 = (expected45 - popt[1])/popt[0]

plt.scatter(channel, cs137calib)
#plt.scatter(channel, nosource, label='background')# angle = 20 deg
plt.axvline(cschann1,color = "r", label = ("Value from compton eq."))
plt.title("Cs-137 at 0 degrees of incidence")
plt.legend()
plt.xlabel("Channel")
plt.ylabel("Counts")
plt.grid()
plt.show()
plt.scatter(channel, r20)
#plt.scatter(channel, c20cs137)
#plt.scatter(channel, b20cs137, label='background')# angle = 20 deg
plt.axvline(expectedchannel20,color = "r", label = ("Value from compton eq."))
plt.title("Cs-137 at 20 degrees of incidence")
plt.legend()
plt.xlabel("Channel")
plt.ylabel("Ratio")
plt.grid()
plt.show()
plt.scatter(channel, cs30)
#plt.scatter(channel, c30cs137)
#plt.scatter(channel, b30cs137  , label='background')# angle = 30 deg
plt.axvline(expectedchannel30, color = "r", label = ("Value from compton eq."))
plt.title("Cs-137 at 30 degrees of incidence")
plt.legend()
plt.xlabel("Channel")
plt.ylabel("Ratio")
plt.grid()
plt.show()
plt.scatter(channel, cs45)
#plt.scatter(channel, c45cs137)
#plt.scatter(channel, b45cs137, label = 'background')  # angle = 40
plt.axvline(expectedchannel45,color = "r", label = ("Value from compton eq."))
plt.title("Cs-137 at 45 degrees of incidence")
plt.legend()
plt.xlabel("Channel")
plt.ylabel("Ratio")
plt.grid()
plt.show()

maxenergy20 = popt[0]*np.argmax(cs20) + popt[1]
maxenergy30 = popt[0]*np.argmax(cs30) + popt[1]
maxenergy45 = popt[0]*np.argmax(cs45) + popt[1]
