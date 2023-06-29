# -*- coding: utf-8 -*-
"""
Created on Sun May 14 15:07:57 2023

@author: Edoardo Giancarli
"""

import PySpectr_v4 as psr                          # spectrograms

# to run in the console when generating the images (run the command separately)
%matplotlib inline

#### noise spectrograms ################################################################################################
## path for LIGO-L data
# path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/gout_58633_58638_295_300.mat"
path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/bsd_data_LL_C01_GATED_SUB60HZ_O3_107_108_.mat"

## directory for spectrograms
directory = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_cnn/Spectrograms_noise/"

## set spectrogram parameters
m = 10
lfft = 2**m                # number of FFT points
n_im = 2892                  # N images

## spectrograms
# psr.spectrograms_noise(n_im, path_bsd_gout, lfft, images = True, directory = directory)  # 410x266 (if %matplotlib inline)
psr.spectrograms_noise(n_im, path_bsd_gout, lfft, images = True, directory = directory, key = 'goutL', mat_v73=False)

# 1 min 20 s per 300 im



#### signal + noise #####################################################################################################
## directory for spectrograms
directory = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_cnn/Spectrograms_im/"

## set spectrogram parameters
m = 10
lfft = 2**m               # number of FFT points
n_im = 300                # N random set of parameters for the images

%matplotlib inline

## spectrograms
par = psr.spectrograms_injsignal(n_im = n_im, lfft = lfft, images = True, directory = directory,
                                 store_results = False, save_to_csv = False)

# circa 1 min per im if Doppler is True (lst apparent),
# circa 5 s per im if Doppler is True (lst mean) and circa 2.5 min per 30 im (circa 24 min per 300 im)
# circa 1.3 s per im and 6 min 48 s per 300 im if Doppler is False


# end