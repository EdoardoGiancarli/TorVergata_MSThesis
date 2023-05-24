# -*- coding: utf-8 -*-
"""
Created on Sun May 14 15:07:57 2023

@author: Edoardo Giancarli
"""

import numpy as np                                 # operations
import PySpectr_v3 as psr                          # spectrograms
from scipy.fft import ifftshift, fftshift          # fft pack
import time                                        # clock time

# to run in the console when generating the images (run the command separately)
#!!!%matplotlib inline

#### noise spectrograms ################################################################################################
## path for LIGO-L data
path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/gout_58633_58638_295_300.mat"
# path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/bsd_data_LL_C01_GATED_SUB60HZ_O3_107_108_.mat"

## directory for spectrograms
directory = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Spectrograms_noise/"

## set spectrogram parameters
m = 10
lfft = 2**m                                # number of FFT points
n_im = 300                                 # N images

## spectrograms
# start cronometer
start_time = time.time()
# function
psr.spectrograms_noise(n_im, path_bsd_gout, lfft, images = True, directory = directory)  # 410x266 (if %matplotlib inline)
# psr.spectrograms_noise(n_im, path_bsd_gout, lfft, images = True, directory = directory, key = 'goutL', mat_v73=False)  # 410x266 (if %matplotlib inline)
# stop cronometer
end_time = time.time()
print("Elapsed time: {:.2f} minutes".format((end_time - start_time)/60)) # 1 min 20 s per 300 im



#### signal + noise #####################################################################################################
## directory for spectrograms
directory = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Spectrograms_im/"

## set spectrogram parameters
m = 10
lfft = 2**m               # number of FFT points
n_im = 300                 # N random set of parameters for the images

#!!!%matplotlib inline

## spectrograms
# start cronometer
start_time = time.time()
# function
par = psr.spectrograms_injsignal(n_im = n_im, lfft = lfft, images = True, directory = directory, save_to_csv = True) # 410x266
# stop cronometer
end_time = time.time()
print("Elapsed time: {:.2f} minutes".format((end_time - start_time)/60))

# circa 1 min per im if Doppler is True (lst apparent),
# circa 5 s per im if Doppler is True (lst mean) and circa 2.5 min per 30 im (circa 24 min per 300 im)
# circa 1.3 s per im and 6 min 48 s per 300 im if Doppler is False


# end