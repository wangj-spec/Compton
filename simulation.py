#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 18:04:42 2021
@author: leonardobossi1
"""

import functions as fn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

re = 2.8179 * 10 ** -15  # Classical electron radius, constant used in the Klein-Nishina cross sectional area.
bit_depth = 9
max_signal = 5


source_energy = 662

data = np.loadtxt("Experiment_Data.csv", delimiter=",", skiprows=7, unpack=True)
data2 = np.genfromtxt("Experiment_Data.csv", delimiter=",", max_rows=7, skip_header=1, dtype="str")

# Reading the data (from provided data ile)
channel, nosource, na22calib, mn54calib, cs137calib, am241calib, c20cs137, b20cs137, c30cs137, b30cs137, c45cs137, b45cs137 = data

# Removing the background noise
na22calib -= nosource
mn54calib -= nosource
cs137calib -= nosource
am241calib -= nosource

# Finding the peak to total ratios and linearly interpolating
am241ratio = fn.find_ratio(28, 15, am241calib)
cs137ratio = fn.find_ratio(210, 160, cs137calib)
mn54ratio = fn.find_ratio(260, 210, mn54calib)

peakratios = [am241ratio, cs137ratio, mn54ratio]
energies = [59, 661.657, 834.838]  # peak energies

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
theta_range = np.arange(0, np.pi - 0.02, 0.01)

for ang in theta_range:
    cross_sec.append(fn.mcintegral(ang - 0.0001, ang + 0.0001, source_energy, N=2000)[0])

plt.plot(theta_range * 180 / np.pi, cross_sec)
plt.xlabel('angle (degrees)')
plt.ylabel('cross sectional value')
plt.title('cross sectional area for a small angle range as a function of scattering angle')
plt.grid()

#%%
# Calculating the expected bin value for the Compton edge and the backscattering
gain1 = 2.85e-3
Expected_compton = fn.photon_E(source_energy, np.pi)
expected_signal2 = fn.analogsignal(0, source_energy - Expected_compton, gain1)

backscater_signal = fn.analogsignal(0, fn.photon_E(source_energy, np.pi), gain1)

compton_bin = np.floor((2 ** bit_depth) * expected_signal2 / max_signal)
backscat_bin = np.floor((2 ** bit_depth) * backscater_signal / max_signal)

angles2, c_prob, errors, value180 = fn.cumulative_distribution(source_energy)

nain = 5.8684093929225495e29  # Number dsity of electrons (estimated using Klein Nishina and Cs-137 data)
probabs = 1 - np.exp(-nain * value180 * 0.05)  # absolute probability of scattering
probdect = fn.linear_interpolation(energies, peakratios, source_energy)  # probability of detection occurring
probrel = probabs / (1 - probdect)  # relative probability of scattering if detection doesn't occur.

# Plotting the simulated spectrum for a given source energy
test = fn.spec_sim(65000, source_energy * gain1, gain1, [angles2, c_prob], probdect=probdect, relprob=probrel,
                det_res=0.085)

plt.figure()

plt.scatter(test.keys(), test.values(), label="Energy peak= " + str(source_energy) + ' keV')
plt.axvline(backscat_bin, color='k', label='backscatter peak')
plt.axvline(compton_bin, color='r', label='compton peak')
plt.xlabel('channel value')
plt.ylabel('Counts')
plt.legend()

# Plotting the number of counts due to backscattering and compton scattering
# as a function of angle obtained from the simulation.

# N = 65000 is done and values are binned for every 3 degrees

sim_data, compton_ang, backscat_ang, compbins, backbins = fn.spec_sim(65000, source_energy * gain1, gain1, [angles2, c_prob],
                                               probdect=probdect, relprob=probrel, anglecheck=True, anglebin=3
                                                , separate = True)

compton_edge = (fn.comptonedge(sim_data.keys(), sim_data.values()))

peak_points = fn.localmaxima(sim_data.keys(), sim_data.values())[1]
if len(peak_points) == 4:
    backscat_peak = (peak_points[1])

if len(peak_points) == 3:
    backscat_peak = (peak_points[0])

if len(peak_points) == 2:
    backscat_peak = (peak_points[0])

energyvalues = []
angles = np.arange(0, np.pi, 0.01)

for i in angles:
    energyvalues.append(fn.photon_E(source_energy,i))

plt.figure()
plt.title("Commpton scattering energy distribution")
plt.plot(angles, energyvalues, color = 'k')
plt.ylabel("Energy, kj")
plt.xlabel("Angle, radians")
plt.show()

plt.figure()
plt.scatter(compton_ang.keys(), compton_ang.values(), color='r', marker='x', label='Counts from Compton plateu')
plt.scatter(backscat_ang.keys(), backscat_ang.values(), color='k', marker='x', label='Counts from backscattering')
plt.xlabel('Scattering angle (degrees)')
plt.ylabel('Total counts')
plt.savefig('effect_comp.png', dpi = 500)
plt.legend()
plt.show()

plt.figure()
plt.scatter(compbins.keys(), compbins.values(), color='r', marker='x', label='Counts from Compton plateu')
plt.scatter(backbins.keys(), backbins.values(), color='k', marker='x', label='Counts from backscattering')
plt.axvline(backscat_bin, color='k', label='backscatter peak')
plt.axvline(compton_bin, color='r', label='compton edge')
plt.axvline(backscat_peak[0], color='g', label='found backscatter peak')
plt.axvline(compton_edge[0], color='b', label='found compton edge')
plt.xlabel('Channel')
plt.ylabel('Total counts')
plt.savefig('methodtest.png', dpi = 500)
plt.legend()
plt.show()


plt.figure()

plt.scatter(sim_data.keys(), sim_data.values(), color='r', marker='.')
plt.plot(channel, cs137calib, linestyle='--', color='k', label='Experimental data')
plt.xlabel('Channel')
plt.ylabel('Total counts')
plt.title('Simulated spectrum for photopeak energy of'+str(source_energy)+' keV')
# %%
# Finding error of single-source calibration using MC simulation
graderrors = []
photopeaks = np.arange(500, 901, 25)
photopeakchanns = []
photopeakerrs = []

for e in photopeaks:

    # Obtaining the cumulative distribution for probability of scattering using
    # Klein Nishina
    source_energy = e

    # Obtaining the angles, corresponding cumulative distribtuion value (binned),
    # errors and the total cross section for all scattering angles

    angles2, c_prob, errors, value180 = fn.cumulative_distribution(source_energy)

    # Plotting resulting binned distribution
    plt.figure()
    plt.scatter(angles2, c_prob, color="k", marker='.')
    plt.title("CDF for Kein-Nishina cross section")
    plt.xlabel("Angle (radians)")
    plt.ylabel("Cumulative probability")
    plt.show()

    # Defining parameters to calculate probabilties of scattering events

    nain = 5.8684093929225495e29  # Number dsity of electrons (estimated using Klein Nishina and Cs-137 data)
    probabs = 1 - np.exp(-nain * value180 * 0.05)  # absolute probability of scattering
    probdect = fn.linear_interpolation(energies, peakratios, source_energy)  # probability of detection occurring
    probrel = probabs / (1 - probdect)  # relative probability of scattering if detection doesn't occur.

    gain1 = 2.85e-3

    simgrad = 512/5*gain1 # simgrad = 2**9 / maxsignal * gain from channel = np.floor(2**9 * analogsig/maxsig)

    N = 65000  # Initial intensity

    # Iterating the simulation to find the error in the peak values
    iterations = 5

    # Creating lists for the found energy peaks and channel values
    backscat_peak = []
    comptonedge_peak = []
    main_peak = []

    comp_edge = []
    gradients = []

    for i in range(iterations):
        sim_data = fn.spec_sim(N, source_energy * gain1, gain1, [angles2, c_prob], probdect=probdect, relprob=probrel)

        peak_points = fn.localmaxima(sim_data.keys(), sim_data.values())[1]

        energy_vals = [fn.photon_E(source_energy, np.pi), source_energy - fn.photon_E(source_energy, np.pi), source_energy]
        comp_edge.append(fn.comptonedge(sim_data.keys(), sim_data.values()))

        if len(peak_points) == 4:
            backscat_peak.append(peak_points[1])
            comptonedge_peak.append(peak_points[2])
            main_peak.append(peak_points[3])

            fit_channels = [backscat_peak[-1][0], comp_edge[-1][0], main_peak[-1][0]]

            popt, pcov = curve_fit(fn.linear, energy_vals, fit_channels)
            gradients.append(popt[0])


        if len(peak_points) == 3:
            backscat_peak.append(peak_points[0])
            comptonedge_peak.append(peak_points[1])
            main_peak.append(peak_points[2])

            fit_channels = [backscat_peak[-1][0], comp_edge[-1][0], main_peak[-1][0]]

            popt, pcov = curve_fit(fn.linear, energy_vals, fit_channels)
            gradients.append(popt[0])

        if len(peak_points) == 2:
            backscat_peak.append(peak_points[0])
            main_peak.append(peak_points[1])

            fit_channels = [backscat_peak[-1][0], comp_edge[-1][0], main_peak[-1][0]]

            popt, pcov = curve_fit(fn.linear, energy_vals, fit_channels)
            gradients.append(popt[0])

        if len(peak_points) == 1:
            main_peak.append(peak_points[0])

    photopeakchanns.append(np.mean(main_peak))
    photopeakerrs.append(np.std(main_peak))

    gradients = [grad - simgrad for grad in gradients]
    graderror = np.mean(np.absolute(gradients))# average absolute error
    graderrorperc = graderror/simgrad #percentage error from simulated
    graderrors.append(graderrorperc)


pmulti, covmulti = curve_fit(fn.linear, photopeaks, photopeakchanns, sigma = photopeakerrs)

plt.figure()
plt.scatter(photopeaks ,graderrors, color = 'k', marker = 'x')
plt.ylabel("Fractional error")
plt.xlabel("Energy of photopeak")

comp_edge = np.array(comp_edge)
main_peak = np.array(main_peak)
backscat_peak = np.array(backscat_peak)


# Trying the calibration method using experimental cs137 results

peak_pointsexp = fn.localmaxima(channel, cs137calib)[1]

main_exp = peak_pointsexp[3]
backscat_exp = peak_pointsexp[1]
comp_edgeexp = fn.comptonedge(channel[0:100], cs137calib[0:100])
channel_valeexp = [backscat_exp[0], comp_edgeexp[0], main_exp[0]]

pexp, covexp = curve_fit(fn.linear, [fn.photon_E(662, np.pi), 662 - fn.photon_E(662, np.pi), 662], channel_valeexp)

print('channel/energy gradient found from the experimental data  = '\
      ''+str(pexp[0])+' channel/keV')

# Doing the same calibration but with the simulated spectrum.    
fit_channels = [np.mean(backscat_peak[:, 0]), np.mean(comp_edge[:, 0]), np.mean(main_peak[:, 0])]
channel_err = [np.std(backscat_peak[:, 0]), np.std(comp_edge[:, 0]), np.std(main_peak[:, 0])]

popt, pcov = curve_fit(fn.linear, energy_vals, fit_channels, sigma=channel_err)

print('channel/energy gradient found from the simulated data  = '\
      ''+str(popt[0])+' channel/keV with an error of'+str(np.sqrt(pcov[0])))

