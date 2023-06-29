# -*- coding: utf-8 -*-
"""
Created on Thu May  4 22:26:40 2023

@author: Edoardo Giancarli
"""

import numpy as np                              # operations
from scipy import signal                        # spectrograms
from scipy.fft import fftshift
import matplotlib.pyplot as plt                 # plot
import PySpectr_v2 as psr

from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=24, usetex=True)

#### path for LIGO-L data and simulated signal
path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/gout_58633_58638_295_300.mat"
path_bsd_gsinj = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Simulated_Signals_MSThesis/bsd_softInj_LL_longtransient_powlawC01_GATED_SUB60HZ_O3_295_300_"

#### set spectrogram parameters
m = 10
lfft = 2**m                                # number of FFT points
n_im = 11                                  # N images
amp = 1e1                                  # amp signal wrt noise

#### spectrograms
%matplotlib inline                         # to run in the console when generating the images

psr.spectrograms_injsign(n_im, path_bsd_gout, path_bsd_gsinj, amp, lfft, images = False)       # 410x266
psr.spectrograms_noise(20, path_bsd_gout, lfft, images = False)                                 # 375x231



a = psr.mat_to_dict(path_bsd_gout, key = 'gout_58633_58638_295_300',
                                          data_file = 'noise', mat_v73 = True)


tfstr = np.array(a['cont']['tfstr'][0, 0])


b = psr.mat_to_dict(path_bsd_gsinj + '1', key = 'gsinjL', data_file = 'signal')[0]

c = psr.mat_to_dict(path_bsd_gsinj + '1', key = 'sour', data_file = 'signal')


#### too see if there are problems
# a = []
# b = []
# for j in range(n_im):
    
#     bsd_gsinj, perczero_gsinj = psr.mat_to_dict(path_bsd_gsinj + str(j + 1) + '.mat', key = 'gsinjL')
#     a.append(bsd_gsinj)
#     b.append(perczero_gsinj)



# data = a[0]['y'][int(86400/0.2):]

# c = np.where(data == 0)[0]

# plt.plot(np.arange(len(data)), data)
# plt.show()

# plt.scatter(np.arange(len(c)), c)
# plt.show()
