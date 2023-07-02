# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 02:23:45 2023

@author: Edoardo Giancarli
"""

import PySpectr as psr                          # spectrograms
import matplotlib.pyplot as plt                 # plotting
import numpy as np

#### max distributions ################################################################################################
# path for interferometer data
path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/gout_58633_58638_295_300.mat"

# directory to save the distributions
directory = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Max_Spectr_Distr"

#### noise spectrograms max distr
noise_max, noise_std = psr.noise_spectr_maxdistr(50, path_bsd_gout, timelength = 0.1, lfft = 1024,
                                                 key= 'gout_58633_58638_295_300', mat_v73=True)

fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)    
ax = fig.add_subplot(111)
ax.plot(np.arange(len(noise_max)), noise_max, c='OrangeRed')
plt.title('Noise spectrograms max values distribution in [295, 300]Hz')
plt.ylabel('max values [1/Hz]')
ax.grid(True)
ax.label_outer()            
ax.tick_params(which='both', direction='in',width=2)
ax.tick_params(which='major', direction='in',length=7)
ax.tick_params(which='minor', direction='in',length=4)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.show()

# save noise max distribution
psr.save_to_csv(noise_max, directory, 'NoiseSpectr_MaxDistribution')


#### signals spectrograms max distr
sgn_max, sgn_std = psr.sgn_spectr_maxdistr(30, path_bsd_gout, lfft = 1024,
                                           key= 'gout_58633_58638_295_300', mat_v73=True)

plt.figure(None)
plt.plot(np.arange(len(sgn_max)), sgn_max)
plt.show()

# save noise max distribution
psr.save_to_csv(sgn_max, directory, 'SgnSpectr_MaxDistribution_1')


#### total max distribution
max_distr = noise_max + sgn_max    # max: 2.603130588172712e-06

plt.figure(None)
plt.plot(np.arange(len(max_distr)), max_distr)
plt.show()


#### join multiple max distribution
path_dir = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Max_Spectr_Distr"

n = load_from_csv(path_dir + 'NoiseSpectr_MaxDistribution')

s1 = load_from_csv(path_dir + 'SgnSpectr_MaxDistribution_1')
s2 = load_from_csv(path_dir + 'SgnSpectr_MaxDistribution_2')
t = n + s1 + s2

plt.figure(None)
plt.plot(np.arange(len(max_distr)), max_distr)
plt.show()


#### noise spectrograms ################################################################################################
# directory for noise spectrograms
directory = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_cnn/Spectrograms_noise/"

# to run in the console when showing the images (run the command separately) #!!!
%matplotlib inline 

# spectrograms
psr.noise_spectr(50, path_bsd_gout, timelength = 0.1, lfft = 256, max_value = 2.603130588172712e-06, images = False,
                 directory = directory, key= 'gout_58633_58638_295_300', mat_v73=True)


#### signal + noise #####################################################################################################
# directory for signals spectrograms
directory = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Prova_cnn/Spectrograms_im/"

# to run in the console when showing the images (run the command separately) #!!!
%matplotlib inline

# spectrograms
psr.sgn_spectr(30, path_bsd_gout, lfft = 128, max_value = 2.603130588172712e-06, images = False,
               directory = directory, key= 'gout_58633_58638_295_300', mat_v73=True)


# end