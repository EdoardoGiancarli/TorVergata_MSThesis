# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:18:47 2023

@author: Edoardo Giancarli
"""

import numpy as np                 # operations
from scipy import signal           # spectrograms
# import pandas as pd                # dataframe  

# from scipy.io import loadmat       # matlab datafile
# import hdf5storage as hd           # matlab datafile -v7.3  

import matplotlib.pyplot as plt                   # plotting
# import matplotlib.patches as mpatches
# from matplotlib.ticker import AutoMinorLocator
# from astroML.plotting import setup_text_plots
# setup_text_plots(fontsize=26, usetex=True)


#### Spectrogram with scipy

def freq_pow_law(t, fgw0, tau = 150000, nbreak = 5, tcoes = 0):
    
    return fgw0*(1. + (t - tcoes)/tau)**(1./(1 - nbreak))


def sign_pow_law(t, freq, fgw0, tau = 150000, nbreak = 5, tcoes = 0):
    
    return 1e-5*(freq**2)*np.cos(2*np.pi*freq*t)

# return 1e-5*(freq**2)*np.cos(2*np.pi*fgw0*((1 - nbreak)/(2 - nbreak))*((1 + (t - tcoes)/tau)**((2 - nbreak)/(1 - nbreak)) - 1))

## Generate wave signal
dx = 1/610                            # samplig time [s]
days = 0.3                            # N days
time = np.arange(0, days*86400, dx)   # time array [s] (86400 N of seconds in one day)

fgw0 = 300
freq = freq_pow_law(time, fgw0)                 # freq [Hz]
freq = freq[freq > 290]                         # freq > 290 [Hz]
time = time[:len(freq)]
signal_data = sign_pow_law(time, freq, fgw0)    # amplitude

# plt.figure()
# plt.plot(time, signal_data, color = 'OrangeRed', label = 'sig amplitude')
# plt.plot(time, freq, color = 'LawnGreen', label = 'sig frequency')
# plt.legend()
# plt.grid(True)
# plt.show()

## Set spectrogram parameters
m = 10
lfft = 2**m                                 # number of FFT points
window_length = lfft                        # chunk length
hop_length = lfft//2                        # edge

## Generating some Gaussian noise
noise = np.random.normal(loc = 0, scale = 1, size = len(signal_data))
amp = 10
data = amp*signal_data + noise

## Compute spectrogram
F, T, spectrogram = signal.spectrogram(data, fs=1/dx, window='cosine', 
                                       nperseg=window_length, noverlap=window_length-hop_length,
                                       nfft=lfft, scaling='density')


## Plot spectrogram
plt.figure(num = 3, figsize = (12, 12), tight_layout = True)
a = plt.pcolormesh(T/86400, F, np.log10(spectrogram), cmap='viridis')
plt.colorbar(a)
plt.ylim([275, 305])
plt.title('Spectrogram')
plt.ylabel('frequency [Hz]')
plt.xlabel('time [days]')
plt.show()




#### Spectrogram with gwosc and gwpy

from gwpy.timeseries import TimeSeries
from gwpy.segments import Segment
from gwpy.io import fetch_open_data

# Define the GPS start and end times
start_time = 1187008801      # GPS start time
end_time = 1187009201        # GPS end time

# Define the detector and channel name
detector = 'L1'
channel = 'L1:LOSC-STRAIN'

# Download the data
data = fetch_open_data(detector, start_time, end_time, channel)

# Extract a segment of data
segment = Segment(start_time, end_time)
data = data.crop(segment)

## Compute spectrogram
spectrogram = data.spectrogram2(fftlength=4, overlap=0.5)

## Plot spectrogram
spectrogram.plot(figsize=(10,4), cmap='viridis', logx=True, logy=True)
