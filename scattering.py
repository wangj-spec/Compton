
import numpy as np
import random as rnd
from scipy.stats import norm
import matplotlib.pyplot as plt

re = 2.8179 * 10 ** -15


def photon_E(E_initial, theta, e_rest=511):
    E_final = E_initial / (1 + (E_initial * (1 - np.cos(theta)) / e_rest))

    return E_final


def Gaussian(x, const, sigma, u):
    '''
    Returns Guassian function for a given x value and Gaussian parameters.
    sigma::float
        Standard deviation
    const::float
        constant
    u::float
        mean value
    '''
    return (const / (sigma * np.sqrt(2 * np.pi))) * np.e ** (-0.5 * ((x - u) / sigma) ** 2)


def analogsignal(angle, e0, gain, e_energy=511):
    # Calculating the scattering energy from the Compton relation
    energymeasure = e0 / (1 + (e0 / e_energy) * (1 - np.cos(np.deg2rad(angle))))

    # Computing the analog signal by multiplying the gain.
    analogsignal = energymeasure * gain

    return analogsignal


def theta_detector(r, diam=50.8e-3):
    '''
    Returns angluar range of the detector.
    '''

    dtheta = 2 * np.arcsin(diam / (2 * r))

    return dtheta


def backscatter(energy, gain):
    seed = rnd.random()
    angle = norm.ppf(seed, loc=180, scale=45)  # angle choosing in degrees
    signalmeasured = analogsignal(angle, energy, gain)

    return signalmeasured


def cross_diff(theta, E_initial, re=re):
    E_scatter = photon_E(E_initial, theta)
    P = E_scatter / E_initial

    cross_diff = 1 / 2 * (re ** 2) * (P ** 2) * (P + 1 / P - (np.sin(theta)) ** 2)

    return cross_diff

def generate_noise(N, analogsig, gain, det_res=0.075, max_signal=5, bit_depth=9, probback=0, scat_prob=0):
    bins = {}

    for i in range(2 ** bit_depth):
        bins[i] = 0

    for i in range(N):

        seed = rnd.random()
        noise_signal = norm.ppf(seed, loc=0, scale=det_res * max_signal / 2)  # Gaussian noise

        probscatt_n = rnd.random()
        backprob_n = rnd.random()

        if probscatt_n <= scat_prob:
            seed = rnd.random()
            angle = norm.ppf(seed, loc=180, scale=180)  # angle choosing in degrees [don't know the distribution yet]
            energy = analogsig / gain
            scattered = analogsignal(angle, energy, gain)
            signal = analogsig - scattered
            tot_signal = signal + noise_signal

            if tot_signal < 0:
                continue  # not physical result

            bin_val = np.floor((2 ** bit_depth) * tot_signal / max_signal)

            bins[bin_val] += 1


        elif backprob_n <= probback:  # chance of passing through the scintillating crystal
            backsignal = backscatter(analogsig / gain, gain)
            tot_signal = backsignal + noise_signal
            if tot_signal < 0:
                continue  # not physical result
            bin_val = np.floor((2 ** bit_depth) * tot_signal / max_signal)

            bins[bin_val] += 1
        else:

            tot_signal = analogsig + noise_signal

            bin_val = np.floor((2 ** bit_depth) * tot_signal / max_signal)

            bins[bin_val] += 1

    return bins


def mcintegral(theta_m, theta_p ,N = 2000 ):
    int_vol = 2 * np.pi * (np.cos(theta_m) - np.cos(theta_p))
    int_vals = []

    for i in range(N):
        dtheta = theta_p - theta_m
        theta = theta_m + rnd.random() * dtheta

        int_vals.append(cross_diff(theta, 662))

    err = int_vol * np.std(int_vals) / np.sqrt(N)

    integral_est = (int_vol / N) * sum(int_vals)
    return integral_est, err

electron_density = 17.41e28
scattererlength = 4e-4

diameter = 15e-3 #detector diameter
radius = 0.3 #radius at which detector is placed
N = int(100000)
scatter_angle = 30
theta = scatter_angle*np.pi/180 # 10 degrees in radians
dtheta = theta_detector(radius, diameter)

sigma, sigmasigma = mcintegral(theta-dtheta/2,theta+dtheta/2 )

N_angle = int(np.floor(N*(1-np.exp(-electron_density*scattererlength*sigma))))
print(N_angle)

gain1 = 2.85e-3
gain2 = 5e-3

# Experimental parameters

source_energy = 662
e_energy = 511

analsig30 = analogsignal(scatter_angle, source_energy, gain1, e_energy)


botheffects0 = generate_noise(N, source_energy*gain1, gain1 )
botheffects30 = generate_noise(N_angle, analsig30, gain1)

plt.figure()
plt.scatter(botheffects30.keys(), botheffects30.values(), label = "Scattered at 30 deg")
plt.scatter(botheffects0.keys(), botheffects0.values(), label = "No scatter")
plt.title("Including both Compton effect and backscattering")
plt.xlabel("Channel")
plt.ylabel("Counts")
plt.legend()
plt.grid()
plt.show()

#%%
#relative distributions:


electron_density = 17.41e28
scattererlength = 4e-4

diameter = 15e-3 #detector diameter
radius = 0.3 #radius at which detector is placed

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
