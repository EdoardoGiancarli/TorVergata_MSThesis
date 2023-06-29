# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:48:09 2023

@author: Edoardo Giancarli
"""

import numpy as np                              # operations
from scipy import signal                        # spectrograms
from scipy.fft import fftshift
import matplotlib.pyplot as plt                 # plot
import DriverMSThesis_v1 as dms

from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=24, usetex=True)


#### path for LIGO-L data and simulated signal
path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/gout_58633_58638_295_300.mat"
path_bsd_gsinj = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/bsd_softInj_LL_longtransient_powlawC01_GATED_SUB60HZ_O3_295_300_.mat"

#### dict with data
bsd_data = dms.get_data(path_bsd_gout, path_bsd_gsinj, amp=1e1, key= 'gout_58633_58638_295_300', mat_v73=True)

#### set spectrogram parameters
m = 10
lfft = 2**m                                # number of FFT points
window_length = lfft - 1                   # chunk length
data = bsd_data['y']                       # data: signal + noise             
dx = bsd_data['dx']                        # sampling time

plt.plot(np.arange(len(data)), data)
plt.show()



#### spectrogram

F, T, spectrogram = signal.spectrogram(data, fs=1/dx, window='hann', 
                                       nperseg=window_length, noverlap=lfft//2,
                                       nfft=lfft, return_onesided=False, scaling='density')

# T, F = np.meshgrid(T, F)

## plot spectrogram
plt.figure(num = 1, figsize = (12, 12), tight_layout = True)
# a = plt.pcolormesh(T/86400, F, np.log10(spectrogram), cmap='viridis')
a = plt.pcolormesh(T/86400, fftshift(F), np.log10(fftshift(spectrogram, axes=0)), cmap='viridis', shading='gouraud')
plt.colorbar(a)
plt.title('Spectrogram')
plt.ylabel('frequency [Hz]')
plt.xlabel('time [days]')
plt.show()


## images
directory = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Spectrograms_im/"
im_title = 'my_plot'

# %matplotlib inline    to run in the console when generating the images

for i in range(2):
    plt.figure(num = i, tight_layout = True) # 842 x 842
    plt.pcolormesh(T/86400, fftshift(F), np.log10(fftshift(spectrogram, axes=0)), cmap='viridis', shading='gouraud')
    plt.axis('off')
    plt.title('')
    plt.savefig(directory + im_title + str(i) + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()

