#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 18:04:42 2021

@author: leonardobossi1
"""

import numpy as np
import random as rnd
from scipy.stats import norm
import matplotlib.pyplot as plt

re = 2.8179 * 10 ** -15 # Classical electron radius, constant used in the Klein-Nishina cross sectional area.
bit_depth = 9
max_signal = 5

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
        raise Exception('Scattering angle defined from 0 to 180 degrees')
    
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
    energymeasure = e0 / (1 + (e0 / e_energy) * (1 - np.cos(angle)))
    
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


def spec_sim(N, analogsig, gain, angledist, det_res=0.075, max_signal=5, bit_depth=9, probdect=1, relprob=0, anglecheck = False, anglebin = 0):
    """
    Simulates a detector using the Monte-Carlo method, and returns an observed energy spectrum
    Params:
        N:: int
            Number of incident photons of energy source energy
        analogsig:: float
            Expected analog signal due to source energy
        gain:: float
            Voltage to Energy ratio within detector
        angledist:: 2d array
            Cumulative distribution function of angle, of form [angles, probability]
        det_res:: float
            Detector resolution at 662 keV
        max_signal:: float
            Max voltage expected within detector, expects 5V
        bit_depth:: int
            Max no. of bits used to define channels, expected 9 => 512 channels
        probdect:: float
            Probability of absorbing incident photon within the crystal with no other events occurring
        relprob:: float
            Probability of incident photon compton scattering with crystal if it is not absorbed
        anglecheck:: bool
            If True, will return angle distributions for the backscatter and the compton plateau
        anglebin:: float
            How wide the bins on the angle distribution will be (in degrees)
    returns:
        bins:: dictionary
            Dictionary with channels as keys and count numbers as values
    """
    
    bins = {}

    for i in range(2 ** bit_depth):
        bins[i] = 0

    if anglecheck:
        compangles = {}
        backscangles = {}
        for i in np.arange(anglebin / 2, 180, anglebin):
            compangles[i] = 0
            backscangles[i] = 0

    for i in range(N):

        detectionprob = rnd.random()
        probscatt_n = rnd.random()

        if detectionprob <= probdect:  # probability of normal absorption within detector

            seed = rnd.random()
            noise_signal = norm.ppf(seed, loc=0, scale=det_res * analogsig / 2)  # Gaussian noise

            tot_signal = analogsig + noise_signal

            bin_val = np.floor((2 ** bit_depth) * tot_signal / max_signal)

            bins[bin_val] += 1

        elif detectionprob > probdect:

            if probscatt_n <= relprob:  # compton scattering within detector
                seed = rnd.random()

                angles = angledist[0]
                probability = angledist[1]
                angle = linear_interpolation(probability, angles, seed)

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

                if anglecheck:
                    angleval = np.floor(angle*180/np.pi / anglebin) * anglebin + anglebin / 2
                    compangles[angleval] += 1

            else:  # backscattering
                
                seed = rnd.random()
                angles = angledist[0]
                probability = angledist[1]
                angle = linear_interpolation(probability, angles, seed)

                if angle < np.pi / 2:
                    continue  # will not reach crystal

                backsignal = analogsignal(angle, analogsig / gain, gain)

                seed2 = rnd.random()
                noise_signal = norm.ppf(seed2, loc=0, scale=det_res * backsignal / 2)

                tot_signal = backsignal + noise_signal
                if tot_signal < 0:
                    continue  # not physical result

                bin_val = np.floor((2 ** bit_depth) * tot_signal / max_signal)

                bins[bin_val] += 1

                if anglecheck:
                    angleval = np.floor(angle * 180 / np.pi / anglebin) * anglebin + anglebin / 2
                    backscangles[angleval] += 1

    if anglecheck:
        return bins, compangles, backscangles
    else:
        return bins


def mcintegral(theta_m, theta_p, energy ,N = 10000):
    '''
    Monte Carlo integration.
    
    Params:
        theta_m:: float
        theta_p:: float
            smaller and larger angle values being integrated over respectively.
        energy:: float
            Peak energy value of incoming photon (used to calculate differential
            cross section).
        N:: int
            Number of iterations
    Returns:
        integral_est::float
        err::float
            The estimated value for the integral and its associated error
            
        
    '''
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
    '''
    Params:
        channel_p:: int
        channel_m:: int
            Higher and lower bounds for the channels making up a peak
        counts:: int
            Counts as a function of channel number
    Returns:
        ratio:: float
            Ratio between the counts between channel_m and channel_p and the
            total counts.
    '''
    
    peak_counts = np.sum(counts[channel_m:channel_p])
    all_counts = np.sum(counts)
    ratio = peak_counts / all_counts
    
    return ratio


def cumulative_distribution(source_energy, e_energy=511):
    '''
    Params:
        source_energy:: float
            Peak energy of the source being used in keV.
        e_energy:: float
            electron rest mass energy, default to 511 keV
    Returns:
        angles2:: list
            scattering angle
        c_prob:: list
            Corresponding cumulative probability (normalised) binned for every
            5 datapoints 
        errors:: list
            Corresponding errors for values.
        value180::
            cumulative cross sectional value at an scattering angle of 180
            degrees. This is the value of the total cross sectional area as 
            it is not normalised.
            
    '''
    integral = []
    angles = [] 
    values = np.arange(0,180, 0.5)

    for i in values: # cumulative distribution function made from Kein Nishina area
        theta_m  = 0
        theta_p = i *np.pi/180
        integral_est, err = mcintegral(theta_m, theta_p, source_energy, 20000)
        integral.append(integral_est)
        angles.append(i * np.pi/180)
        
        # Binning the values from the cumulative distribution
    
    for i in range(len(angles)):
        if i == 0: # edge case 0 (beginning)
            angles2 = [0]
            errors = [0]
            c_prob = [integral[i]]
        else :
            if i % 5 == 0: #binning values every 5 points
                average = np.mean(integral[i-5:i])
                error = np.std(integral[i-5:i])
                c_prob.append(average)
                errors.append(error)
                angles2.append(angles[i-2])
    
    value180 = linear_interpolation(angles2, c_prob, np.pi) # Obtaining the total cross section from 0 to 180 degrees
    c_prob.append(value180)
    errors.append(0)
    angles2.append(np.pi)

    # Normalising the values from the distribution
    c_prob = c_prob/c_prob[-1]
    errors = errors/c_prob[-1]
    
    return angles2, c_prob, errors, value180
    

def localmaxima(arrayx, arrayy):
    '''
    Params:
        arrayx:: numpy array
        arrayy:: numpy array
            x and y values for a given dataset.
    Returns:
        n:: int
            Number of maxima detected in given x and y data.
        maxima:: list
            List of the positions of the maxima and the corresponding y values.
    '''
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
                    # Finding the indices corresponding to the maximum value in the bins
                    maximumindex = i - 10 + np.argmax(arrayy[i - 10: i + 20]) 
                    maxima.append((arrayx[maximumindex], arrayy[maximumindex]))
                    
    return n, maxima


def comptonedge(arrayx, arrayy, binsize = 6):
    arrayx = list(arrayx)
    arrayy = list(arrayy)
    compedge = []
    binnedvalues = []
    for i in range(len(arrayy)): # binning values
        if i % binsize == 0:
            mean = np.mean(arrayy[i:i + binsize])
            binnedvalues.append(mean)
    for i in range(len(binnedvalues)-1): # finding the compton valley
        if i == 0:
            previousmean = np.inf
            currentmean = binnedvalues.pop() # walking backwards from the end
            nextmean = binnedvalues.pop()
        else:
            previousmean = currentmean
            currentmean = nextmean
            nextmean = binnedvalues.pop()
            if previousmean > currentmean and nextmean > currentmean: # compton valley found
                break

    j = 0
    for i in range(len(binnedvalues)-1): # inflection point
        previousmean = currentmean
        currentmean = nextmean
        nextmean = binnedvalues.pop()
        prevgrad = currentmean - previousmean
        nextgrad = nextmean - currentmean

        if nextgrad < prevgrad: # local minimum
            j = 1
            continue
        if j == 1:
            if nextgrad > prevgrad:
                value = arrayy[int((len(binnedvalues)+1)*binsize+np.floor(binsize/2 + 1/2))]
                channel = arrayx[int((len(binnedvalues)+1)*binsize+np.floor(binsize/2 + 1/2))]
                break
            else:
                j = 0
                continue

    return  channel, value


#%%
source_energy= 662

data = np.loadtxt("Experiment_Data.csv", delimiter = ",", skiprows= 7, unpack = True )
data2 = np.genfromtxt("Experiment_Data.csv", delimiter = ",", max_rows= 7,skip_header=1, dtype="str")

# Reading the data (from provided data ile)
channel,nosource,na22calib,mn54calib,cs137calib,am241calib,c20cs137,b20cs137,c30cs137,b30cs137,c45cs137,b45cs137 = data

# Removing the background noise
na22calib -= nosource
mn54calib -= nosource
cs137calib -= nosource
am241calib -= nosource

# Finding the peak to total ratios and linearly interpolating
am241ratio =  find_ratio(28,15, am241calib)
cs137ratio =  find_ratio(210,160, cs137calib)
mn54ratio = find_ratio(260, 210, mn54calib)

peakratios = [am241ratio, cs137ratio, mn54ratio]
energies = [59, 661.657, 834.838] # peak energies

plt.figure()
plt.plot(energies, peakratios)
plt.scatter(energies, peakratios, marker='x', color='k', label='datapoints from experimental calibration data')
plt.title("Probability of detection for Gamma rays (peak counts to total counts ratio)")
plt.xlabel("Energy of peak in keV")
plt.ylabel("Peak to total ratio")
plt.grid()
plt.legend()

plt.figure()
cross_sec = []
theta_range = np.arange(0, np.pi-0.02, 0.01)

for ang in theta_range:
    cross_sec.append(mcintegral(ang - 0.0001, ang + 0.0001, source_energy, N = 2000)[0])
    

plt.plot(theta_range * 180/np.pi, cross_sec)
plt.xlabel('angle (degrees)')
plt.ylabel('cross sectional value')
plt.title('cross sectional area for a small angle range as a function of scattering angle')
plt.grid()


#%%
# Finding error of single-source calibration using MC simulation

# Obtaining the cumulative distribution for probability of scattering using 
# Klein Nishina
source_energy= 662

# Obtaining the angles, corresponding cumulative distribtuion value (binned), 
#errors and the total cross section for all scattering angles 

angles2, c_prob, errors, value180 = cumulative_distribution(source_energy)

# Plotting resulting binned distribution
plt.figure()
plt.scatter(angles2, c_prob, color =  "k", marker = '.')
plt.title("CDF for Kein-Nishina cross section")
plt.xlabel("Angle (radians)")
plt.ylabel("Cumulative probability")
plt.show()

# Defining parameters to calculate probabilties of scattering events

nain = 5.8684093929225495e29 # Number density of electrons (estimated using Klein Nishina and Cs-137 data)
probabs = 1-np.exp(-nain * value180 * 0.05) # absolute probability of scattering
probdect = linear_interpolation(energies, peakratios, source_energy)# probability of detection occurring
probrel = probabs/(1-probdect) # relative probability of scattering if detection doesn't occur.

gain1 = 2.85e-3

N = 65000 # Initial intensity 

# Simulating spectrum with backscattering and compton edge
botheffects0 = spec_sim(N, source_energy * gain1, gain1, [angles2, c_prob],probdect= probdect, relprob= probrel )

plt.figure()
plt.scatter(botheffects0.keys(), botheffects0.values(), label = "Energy peak= "+str(source_energy)+' keV')
plt.title("Simulated Compton edge")
plt.xlabel("Channel")
plt.ylabel("Counts")
plt.grid()
plt.show()

edge_coords = comptonedge(botheffects0.keys(), botheffects0.values())
print(edge_coords)

# Iterating the simulation to find the error in the peak values
iterations = 20

backscat_peak = []
comptonedge_peak = []
comptonedges = []
main_peak = []


for i in range(iterations):
    print(i)
    sim_data = generate_noise(N, source_energy * gain1, gain1, [angles2, areas], probdect=probdect, relprob=probrel)
    peak_points = localmaxima(sim_data.keys(), sim_data.values())[1]

    if len(peak_points) == 3:
        backscat_peak.append(peak_points[0])
        comptonedge_peak.append(peak_points[-2])
        main_peak.append(peak_points[-1])

    if len(peak_points) == 2:
        backscat_peak.append(peak_points[0])
        main_peak.append(peak_points[-1])

    if len(peak_points) == 1:
        main_peak.append(peak_points[-1])

    comptonedges.append(comptonedge(sim_data.keys(), sim_data.values()))
        

    
#%%
# Calculating the expected bin value for the Compton edge and the backscattering
source_energy = 511

Expected_compton = photon_E(source_energy, np.pi)
expected_signal2 = analogsignal(0, source_energy- Expected_compton, gain1) 

backscater_signal = analogsignal(0, photon_E(source_energy, np.pi), gain1) 

compton_bin = np.floor((2 ** bit_depth) * expected_signal2/ max_signal)
backscat_bin = np.floor((2 ** bit_depth) * backscater_signal/ max_signal)


# Test case
test= spec_sim(65000, source_energy * gain1, gain1, [angles2, c_prob],probdect= probdect, relprob= probrel, det_res = 0.075)

plt.figure()

plt.scatter(test.keys(), test.values(), label = "Energy peak= "+str(source_energy)+' keV')
plt.axvline(backscat_bin, color = 'k', label= 'backscatter peak')
plt.axvline(compton_bin, color = 'r', label='compton peak')
plt.legend()

    
    
    
    
