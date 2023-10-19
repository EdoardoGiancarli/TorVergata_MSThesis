# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 02:23:45 2023

@author: Edoardo Giancarli
"""


import PySpectr as psr
import pathlib
from tqdm import tqdm          


##########################################################################################################
#########################################  denoiser  #####################################################

# path for store images
clean_train_dir = "path"
noisy_train_dir = "path"

clean_test_dir = "path"
noisy_test_dir = "path"

# path for interferometer data
imgdir_path = pathlib.Path("data_path")
path_bsd_gout = sorted([str(path) for path in imgdir_path.glob('*.mat')])


#### noise spectrograms ################################################################################################
# to run in the console when showing the images (run the command separately) #!!!
# %matplotlib inline

# spectrograms
img = 17

# train
for i, bsdpath in tqdm(enumerate(path_bsd_gout)):
    print(f"#########  SPECTROGRAM GENERATION LOOP {i}/{len(path_bsd_gout) - 1}  #########")	
    psr.noise_spectr(img, bsdpath, timelength = 0.1, lfft = 512,
                     images = True, directory = clean_train_dir,
                     title1 = 'z_dns_noise_cleanspectrtrain_', directory2=noisy_train_dir,
                     title2='z_dns_noise_noisyspectrtrain_', n=i*img)


# test
for i, bsdpath in tqdm(enumerate(path_bsd_gout)):
    print(f"#########  SPECTROGRAM GENERATION LOOP {i}/{len(path_bsd_gout) - 1}  #########")
    psr.noise_spectr(img, bsdpath, timelength = 0.1, lfft = 512,
                     images = True, directory = clean_test_dir,
                     title1 = 'z_dns_noise_cleanspectrtest_', directory2=noisy_test_dir,
                     title2='z_dns_noise_noisyspectrtest_', n=i*img)


#### signal + noise #####################################################################################################
# to run in the console when showing the images (run the command separately) #!!!
# %matplotlib inline

# spectrograms
img = 40
case = 'Mpcscale'

# train
for i, bsdpath in tqdm(enumerate(path_bsd_gout)):
    print(f"#########  SPECTROGRAM GENERATION LOOP {i}/{len(path_bsd_gout) - 1}  #########")
    psr.sgn_spectr(img, lfft = 512, images = True,
                   directory=noisy_train_dir, directory2=clean_train_dir, path_bsd_gout = bsdpath, n=i*img + 1056,
                   title='dns_noisy_spectrtrain_' + case + '_', title2='dns_clean_spectrtrain_' + case + '_')


# test
for i, bsdpath in tqdm(enumerate(path_bsd_gout)):
    print(f"#########  SPECTROGRAM GENERATION LOOP {i}/{len(path_bsd_gout) - 1}  #########")
    psr.sgn_spectr(img, lfft = 512, images = True,
                   directory=noisy_test_dir, directory2=clean_test_dir, path_bsd_gout = bsdpath, n=i*img + 1061,
                   title='dns_noisy_spectrtest_' + case + '_', title2='dns_clean_spectrtest_' + case + '_')






##########################################################################################################
#########################################  classifier  ###################################################

import PySpectr as psr                          # spectrograms
import PyUtils as pu                            # interferometer data
from tqdm import tqdm          
import pathlib


# path for interferometer data
imgdir_path = pathlib.Path("data_path")
path_bsd_gout = sorted([str(path) for path in imgdir_path.glob('*.mat')])

imgdir_path_test = pathlib.Path("data_path")    # different BSD for train and test images
path_bsd_gout_test = sorted([str(path) for path in imgdir_path_test.glob('*.mat')])


#### noise spectrograms ################################################################################################
# directory for noise spectrograms
directory_train = "path"
directory_test = "path"

# to run in the console when showing the images (run the command separately) #!!!
# %matplotlib inline

img = 20
# spectrograms
for i, bsdpath in tqdm(enumerate(path_bsd_gout)):
    print(f"#########  SPECTROGRAM GENERATION LOOP {i}/{len(path_bsd_gout) - 1}  #########")
    psr.noise_spectr(img, bsdpath, timelength = 0.1, lfft = 512,
                     images = True, directory = directory_train,
                     title1 = 'z_cls_noise_', n=i*img)


for i, bsdpath in tqdm(enumerate(path_bsd_gout_test)):
    print(f"#########  SPECTROGRAM GENERATION LOOP {i}/{len(path_bsd_gout_test) - 1}  #########")
    psr.noise_spectr(img, bsdpath, timelength = 0.1, lfft = 512,
                     images = True, directory = directory_test,
                     title1 = 'z_cls_noise_', n=i*img)


#### signal + noise #####################################################################################################
# directory for signals spectrograms
directory_train = "path"
directory_test = "path"

# to run in the console when showing the images (run the command separately) #!!!
# %matplotlib inline

# spectrograms
img = 25

# train
for i, bsdpath in tqdm(enumerate(path_bsd_gout)):
    print(f"#########  SPECTROGRAM GENERATION LOOP {i}/{len(path_bsd_gout) - 1}  #########")
    psr.sgn_spectr(img, lfft = 512, norm = False, norm_value=None, images = True,
                   directory=directory_train, path_bsd_gout = bsdpath, n=i*img, title='clf_spectr_')


# test
for i, bsdpath in tqdm(enumerate(path_bsd_gout_test)):
    print(f"#########  SPECTROGRAM GENERATION LOOP {i}/{len(path_bsd_gout_test) - 1}  #########")
    psr.sgn_spectr(img, lfft = 512, norm = False, norm_value=None, images = True,
                   directory=directory_test, path_bsd_gout = bsdpath, n=i*img, title='clf_spectr_')



# end