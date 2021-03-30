
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


def theta_detector(r=0.3, diam=15e-3):
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
            break
        else:
            continue
        
    if x >= array1[-1]:
        
        value = array2[-1] + (array2[-1] - array2[-2]) * (x - array1[-1]) / (array1[-1]-array1[-2])

    return value


def generate_noise(N, analogsig, gain, angledist, det_res=0.075, max_signal=5, bit_depth=9, probdect=1, relprob=0):
    """
    Simulates a detector using the Monte-Carlo method, and returns an observed energy spectrum
    :param N:: int
            Number of incident photons of energy source energy
    :param analogsig:: float
            Expected analog signal due to source energy
    :param gain:: float
            Voltage to Energy ratio within detector
    :param angledist:: 2d array
            Cumulative distribution function of angle, of form [angles, probability]
    :param det_res:: float
            Detector resolution at 662 keV
    :param max_signal:: float
            Max voltage expected within detector, expects 5V
    :param bit_depth:: int
            Max no. of bits used to define channels, expected 9 => 512 channels
    :param probdect:: float
            Probability of absorbing incident photon within the crystal with no other events occurring
    :param relprob:: float
            Probability of incident photon compton scattering with crystal if it is not absorbed
    returns:
            bins:: dictionary
            Dictionary with channels as keys and count numbers as values
    """
    bins = {}

    for i in range(2 ** bit_depth):
        bins[i] = 0

    for i in range(N):

        detectionprob = rnd.random()
        probscatt_n = rnd.random()



        if detectionprob <= probdect:   #probability of normal absorption within detector

            seed = rnd.random()
            noise_signal = norm.ppf(seed, loc=0, scale=det_res * analogsig / 2)  # Gaussian noise

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

                seed2 = rnd.random()
                noise_signal = norm.ppf(seed2, loc=0, scale=det_res * max_signal / 2)  # Gaussian noise

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

                if angle < 90:
                    continue # will not reach crystal

                backsignal = analogsignal(angle, analogsig / gain, gain)

                seed2 = rnd.random()
                noise_signal = norm.ppf(seed2, loc=0, scale=det_res * backsignal / 2)

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


# Obtaining the cumulative distribution for probability of scattering using 
# Klein Nishina
source_energy= 511 

def cumulative_distribution(source_energy, e_energy=511):
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
    
    value180 = linear_interpolation(angles2, areas, 180) # Obtaining the total cross section from 0 to 180 degrees
    areas.append(value180)
    errors.append(0)
    angles2.append(180)

    # Normalising the values from the distribution
    areas = areas/areas[-1]
    errors = errors/areas[-1]
    
    return angles2, areas, errors, value180
    
# Obtaining the angles, corresponding cumulative distribtuion value (binned), 
#errors and the total cross section for all scattering angles 

angles2, areas, errors, value180 = cumulative_distribution(511)

# Plotting resulting binned distribution
plt.figure()
plt.scatter(angles2, areas, color =  "k")
plt.title("CDF for Kein-Nishina cross section")
plt.xlabel("Angle (degrees")
plt.ylabel("Probability")
plt.show()

#%%
nain = 5.8684093929225495e29 # Number density of electrons (estimated using Klein Nishina and Cs-137 data)

probabs = 1-np.exp(-nain * value180 * 0.05) # absolute probability of scattering
probdect = linear_interpolation(energies, peakratios, source_energy)# probability of detection occurring
probrel = probabs/(1-probdect) # relative probability of scattering if detection doesn't occur.

gain1 = 2.85e-3

N = int(20000) # Initial intensity 

# Simulating spectrum with backscattering and compton edge
botheffects0 = generate_noise(N, source_energy * gain1, gain1, [angles2, areas],probdect= probdect, relprob= probrel )

plt.figure()
plt.scatter(botheffects0.keys(), botheffects0.values(), label = "No scatter")
plt.title("Simulated Compton edge")
plt.xlabel("Channel")
plt.ylabel("Counts")
plt.grid()
plt.show()


def localmaxima(arrayx, arrayy):
    arrayx = list(arrayx)
    arrayy = list(arrayy)
    n = 0 # initialise counter
    maxima = []
    for i in range(len(arrayy)):
        if i == 0:  # edge case 0 (beginning)
            previousmean = np.inf
            currentmean = np.mean(arrayy[i:i+5])
            nextmean = np.mean(arrayy[i+5:i+10])
        else:
            if i % 5 == 0:  # binning values every 5 points
                previousmean = currentmean
                currentmean = nextmean
                nextmean = np.mean(arrayy[i+5:i+10])
                if previousmean < currentmean and currentmean < nextmean:
                    n += 1
                    maximumindex = i - 5 + np.argmax(arrayy[i - 5: i + 10])
                    maxima.append((arrayx[maximumindex], arrayy[maximumindex]))
    return n, maxima

