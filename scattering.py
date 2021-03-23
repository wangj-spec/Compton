
import numpy as np
import random as rnd
from scipy.stats import norm
import matplotlib.pyplot as plt

# Classical electron radius, constant used in the Klein-Nishina cross sectional area.
re = 2.8179 * 10 ** -15

def photon_E(E_initial, theta, e_rest=511):
    '''
    Params:
        E_inital:: float
            Energy of incoming gamma ray
        theta:: float
            scatter angle
        e_rest:: float
            Rest mass energy for an electron in keV
    Returns:
        E_final:: float
           Expected energy for given scattering angle.
    '''
    
    if theta > np.pi:
        raise Exception('Scattering angle defined from 0 to pi radians')
    
    E_final = E_initial / (1 + (E_initial * (1 - np.cos(theta)) / e_rest))

    return E_final


def analogsignal(angle, e0, gain, e_energy=511):
    '''
    Params:
        angle:: float
            Scattering angle
        e0:: float
            Initial energy of gamma ray
        gain:: float
            Gain in the detector
        e_energy:: float
            Electron rest mass energy
    Returns:
        analogsignal::float
            Expected peak signal for given scattering angle and energy.
    '''
    
    # Calculating the scattering energy from the Compton relation
    energymeasure = e0 / (1 + (e0 / e_energy) * (1 - np.cos(np.deg2rad(angle))))

    # Computing the analog signal by multiplying the gain.
    analogsignal = energymeasure * gain

    return analogsignal


def theta_detector(r, diam=50.8e-3):
    '''
    Params:
        r::float
            Distance to the detector
        diam::float
            Aperature diameter
    Returns:
        dtheta::float
            angluar range spanned by the detector.
    '''

    dtheta = 2 * np.arcsin(diam / (2 * r))

    return dtheta


def cross_diff(theta, E_initial, re=re):
    '''
    Params:
        theta::float
            Scattering angle
        E_initial::float
            Incoming gamma ray energy in keV
        re::float
            Classical electron radius
    Returns:
        cross_diff::float
            Differential cross section for given angle.
    '''
    E_scatter = photon_E(E_initial, theta)
    P = E_scatter / E_initial

    cross_diff = 1 / 2 * (re ** 2) * (P ** 2) * (P + 1 / P - (np.sin(theta)) ** 2)

    return cross_diff



def linear_interpolation(array1, array2, x):
    '''
    Params:
        array1::arraylike
        array2::arraylike
            data points for x and y available for interpolation.
        x:: float
            x point to be linearly interpolated         
    Returns:
        value::float
            Linearly interpolated value
    '''
    
    for i in range(len(array1) - 1):
        current = array1[i]
        next_val = array1[i+1]
        if next_val > x:
            value = array2[i] + (array2[i+1] - array2[i]) * (x - current) / (next_val - current)
        else:
            continue
        
    if x >= array1[-1]:
        
        value = array2[-1] + (array2[-1] - array2[-2]) * (x - array1[-1]) / (array1[-1]-array1[-2])

    return value


def generate_noise(N, analogsig, gain, angledist, det_res=0.075, max_signal=5, bit_depth=9, probdect=1, relprob=0):
    bins = {}

    for i in range(2 ** bit_depth):
        bins[i] = 0

    for i in range(N):

        detectionprob = rnd.random()
        probscatt_n = rnd.random()

        seed = rnd.random()
        noise_signal = norm.ppf(seed, loc=0, scale=det_res * max_signal / 2)  # Gaussian noise

        if detectionprob <= probdect:   #probability of normal absorption within detector


            tot_signal = analogsig + noise_signal

            bin_val = np.floor((2 ** bit_depth) * tot_signal / max_signal)

            bins[bin_val] += 1

        elif detectionprob > probdect:

            if probscatt_n <= relprob: # compton scattering within detector
                seed = rnd.random()

                angles = angledist[0]
                probability = angledist[1]
                angle = linear_interpolation(probability, angles,seed)

                energy = analogsig / gain
                scattered = analogsignal(angle, energy, gain)
                signal = analogsig - scattered
                tot_signal = signal + noise_signal

                if tot_signal < 0:
                    continue  # not physical result

                bin_val = np.floor((2 ** bit_depth) * tot_signal / max_signal)

                bins[bin_val] += 1

            else: # backscattering
                seed = rnd.random()
                angles = angledist[0]
                probability = angledist[1]
                angle = linear_interpolation(probability, angles, seed)

                if angle < 110:
                    continue # will not reach crystal

                backsignal = backsignal = analogsignal(angle, analogsig / gain, gain)
                tot_signal = backsignal + noise_signal
                if tot_signal < 0:
                    continue  # not physical result
                bin_val = np.floor((2 ** bit_depth) * tot_signal / max_signal)

                bins[bin_val] += 1

    return bins

def mcintegral(theta_m, theta_p, energy ,N = 10000 ):
    int_vol = 2 * np.pi * (theta_p - theta_m)
    int_vals = []

    for i in range(N):
        dtheta = theta_p - theta_m
        theta = theta_m + rnd.random() * dtheta

        int_vals.append(cross_diff(theta, energy)*np.sin(theta))

    err = int_vol * np.std(int_vals) / np.sqrt(N)

    integral_est = (int_vol / N) * sum(int_vals)
    return integral_est, err

def find_ratio(channel_p, channel_m, counts):
    peak_counts = np.sum(counts[channel_m:channel_p])
    all_counts = np.sum(counts)

    return peak_counts / all_counts


data = np.loadtxt("Experiment_Data.csv", delimiter = ",", skiprows= 7, unpack = True )
data2 = np.genfromtxt("Experiment_Data.csv", delimiter = ",", max_rows= 7,skip_header=1, dtype="str")

# Reading the data (from provided data ile)
channel,nosource,na22calib,mn54calib,cs137calib,am241calib,c20cs137,b20cs137,c30cs137,b30cs137,c45cs137,b45cs137 = data

# Removing the background noise
na22calib -= nosource
mn54calib -= nosource
cs137calib -= nosource
am241calib -= nosource

am241ratio =  find_ratio(28,15, am241calib)
cs137ratio =  find_ratio(210,160, cs137calib)
mn54ratio = find_ratio(260, 210, mn54calib)

peakratios = [am241ratio, cs137ratio, mn54ratio]
energies = [59, 661.657, 834.838] # peak energies


plt.figure()
plt.plot(energies, peakratios)
plt.title("Probability of detection for Gamma rays up ")
plt.xlabel("Energy of peak in keV")
plt.ylabel("Peak to total ratio")
plt.grid()
plt.show()

integral = []
angles = []
values = np.arange(0,180, 0.5)
for i in values: # cumulative distribution function made from Kein Nishina area for Caesium
    theta_m  = 0
    theta_p = i*np.pi/180
    integral_est, err = mcintegral(theta_m, theta_p, 661.657)
    integral.append(integral_est)
    angles.append(i)

for i in range(len(angles)):
    if i == 0: # edge case 0 (beginning)
        angles2 = [0]
        errors = [0]
        areas = [integral[i]]
    else :
        if i % 5 == 0: #binning values every 5 points
            average = np.mean(integral[i-5:i])
            error = np.std(integral[i-5:i])
            areas.append(average)
            errors.append(error)
            angles2.append(angles[i-2])

value180cs = linear_interpolation(angles2, areas, 180) # finding a value for caesium radiation

# Experimental parameters

source_energy = 511 # Defines the energy of incident gamma ray 
e_energy = 511


integral = []
angles = []
values = np.arange(0,180, 0.5)

for i in values: # cumulative distribution function made from Kein Nishina area
    theta_m  = 0
    theta_p = i*np.pi/180
    integral_est, err = mcintegral(theta_m, theta_p, source_energy)
    integral.append(integral_est)
    angles.append(i)

# Binning the values from the cumulative distribution
    
for i in range(len(angles)):
    if i == 0: # edge case 0 (beginning)
        angles2 = [0]
        errors = [0]
        areas = [integral[i]]
    else :
        if i % 5 == 0: #binning values every 5 points
            average = np.mean(integral[i-5:i])
            error = np.std(integral[i-5:i])
            areas.append(average)
            errors.append(error)
            angles2.append(angles[i-2])

value180 = linear_interpolation(angles2, areas, 180)
areas.append(value180)
errors.append(0)
angles2.append(180)

# Normalising the values from the distribution
areas = areas/areas[-1]
errors = errors/areas[-1]

# Plotting resulting binned distribution
plt.scatter(angles2, areas, color =  "k")
plt.title("CDF for Kein-Nishina cross section")
plt.xlabel("Angle (degrees")
plt.ylabel("Probability")
plt.show()

#%%
nain = -np.log(1 - 0.5286549105)/(value180cs*0.05) # finding number density of electrons in NaI(Tl) using Cs

probabs = 1-np.exp(-nain*value180*0.05) # absolute probability of scattering
probdect =  linear_interpolation(energies, peakratios, source_energy)# probability of detection occurring
probrel = probabs/(1-probdect) # relative probability of scattering if detection doesn't occur.

# Exeperimental parameters for detector used (aluminimum)
electron_density = 17.41e28
scattererlength = 4e-4
diameter = 15e-3 #detector diameter
radius = 0.3 #radius at which detector is placed

gain1 = 2.85e-3

N = int(20000) # Initial intensity 
scatter_angle = 30
theta = scatter_angle * np.pi/180 # Converting to radians
dtheta = theta_detector(radius, diameter)

# Obtaining the total cross section taking into account the width of detector
sigma, sigmasigma = mcintegral(theta-dtheta/2,theta+dtheta/2 , 662)

# Expected signal for peak scattered energy
analsig30 = analogsignal(scatter_angle, source_energy, gain1, e_energy)

# Simulating spectrum with backscattering and compton edge
botheffects0 = generate_noise(N, source_energy * gain1, gain1, [angles2, areas],probdect= probdect, relprob= probrel )


plt.figure()
plt.scatter(botheffects0.keys(), botheffects0.values(), label = "No scatter")
plt.title("Simulated Compton edge")
plt.xlabel("Channel")
plt.ylabel("Counts")
plt.grid()
plt.show()

#%%
# Testing relative counts between different scattered angles using the Kein-Nishina distribution.

angle1 = 30
angle2 = 20

theta1 = angle1*np.pi/180 # 10 degrees in radians
dtheta1 = theta_detector(radius, diameter)
sigma1, sigmasigma1 = mcintegral(theta-dtheta/2,theta+dtheta/2 )

theta2 = angle2*np.pi/180 # 10 degrees in radians
dtheta2 = theta_detector(radius, diameter)
sigma2, sigmasigma2 = mcintegral(theta-dtheta/2,theta+dtheta/2 )

n1 = 10000
n2 = int(np.floor(np.exp(electron_density*scattererlength*(sigma2 - sigma1))*n1))

analsig1 = analogsignal(angle1, source_energy, gain1, e_energy)
analsig2 = analogsignal(angle2, source_energy, gain1, e_energy)

spectrum1 = generate_noise(n1, analsig1, gain1)
spectrum2 = generate_noise(n2, analsig2, gain1)

plt.figure()
plt.scatter(spectrum1.keys(), spectrum1.values(), label = str(angle1) + " degrees")
plt.scatter(spectrum2.keys(), spectrum2.values(), label = str(angle2) + " degrees")
plt.grid()
plt.legend()
plt.title("Checking relative count rates")
plt.show()

def localmaxima(arrayx, arrayy):
    arrayx = list(arrayx)
    arrayy = list(arrayy)
    n = 0 # initialise counter
    maxima = []
    for i in range(len(arrayy)):
        if i == 0:  # edge case 0 (beginning)
            previousmean = np.inf
            currentmean = np.mean(arrayy[i:i+10])
            nextmean = np.mean(arrayy[i+10:i+20])
        else:
            if i % 10 == 0:  # binning values every 5 points
                previousmean = currentmean
                currentmean = nextmean
                nextmean = np.mean(arrayy[i+10:i+20])
                if previousmean < currentmean and currentmean > nextmean:
                    n += 1
                    maximumindex = i - 10 + np.argmax(arrayy[i - 10: i + 20])
                    maxima.append((arrayx[maximumindex], arrayy[maximumindex]))
    return n, maxima
