import numpy as np
import matplotlib as mpl
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def linear (x, a, b):
    y = a*x + b
    return y

data = np.loadtxt("Experiment_Data.csv", delimiter = ",", skiprows= 7, unpack = True )
data2 = np.genfromtxt("Experiment_Data.csv", delimiter = ",", max_rows= 7,skip_header=1, dtype="str")

channel,nosource,na22calib,mn54calib,cs137calib,am241calib,c20cs137,b20cs137c,c30cs137d,b30s137e,c45cs137f,b45cs137g = data

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

xvals = np.linspace(0, 500)
yvals = list(map(lambda x: linear(x, *popt), xvals))

plt.plot(xvals, yvals, label = str(popt[0]) + "x + " + str(popt[1]))
plt.scatter(channels, energies)
plt.title("Calibration of Energy against Channel")
plt.ylabel("Energy (keV)")
plt.xlabel("Channel")
plt.grid()
plt.legend()
plt.show()

plt.scatter(channel, na22calib)
plt.title("Spectrum of Na-22")
plt.show()

plt.scatter(channel, mn54calib)
plt.title("Spectrum of Mn-54")
plt.show()

plt.scatter(channel, cs137calib)
plt.title("Spectrum of Cs-137")
plt.show()

plt.scatter(channel, am241calib)
plt.title("Spectrum of Am-241")
plt.show()