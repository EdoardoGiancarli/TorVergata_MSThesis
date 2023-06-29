# -*- coding: utf-8 -*-
"""
Created on Wed May  3 17:02:28 2023

@author: Edoardo Giancarli
"""

#### Driver MS Thesis version 1 ###############################################

#### libraries

import numpy as np                                # operations
import pandas as pd                               # dataframe  
from scipy import signal                          # spectrograms
from scipy.fft import fftshift, ifftshift

from scipy.io import loadmat                      # matlab datafile
import hdf5storage as hd                          # matlab datafile -v7.3  

import matplotlib.pyplot as plt                   # plotting
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=12, usetex=True)


#### content







#### codes

# class tempname1(self):



def mat_to_dict(path, key = 'goutL', mat_v73 = False):  # convert the data from MATLAB: it seems that is better to leave goutL as a dict
    
    """
    Conversion from MATLAB data file to dict.
    Parameters:
    -----------------------------------------------------------------------------------------------------------------
    path : (str) path of your MATLAB data file ---
    key : keyword with info from L, H or V interferometer (insert gout + interf. or gsinj + interf., default = goutL) ---
    mat_v73 : (bool) if the matlab datafile version is -v7.3 insert the 'True' value (default = 'False') ---
    
    -----------------------------------------------------------------------------------------------------------------
    return:
        
    data_dict: (dict) dict from MATLAB data file ---
    perczero: (float) perc of total zero data in y (data from the Obs run)
    -----------------------------------------------------------------------------------------------------------------     
    """
    
    if mat_v73:       
        
        mat = hd.loadmat(path)                                        # load mat-file -v7.3 

        mdata = mat[key]                                              # variable in mat file
        mdtype = mdata.dtype                                          # dtypes of structures are "unsized objects"
        data_dict = {n: mdata[n][0] for n in mdtype.names}            # express mdata as a dict
        
        dx = data_dict['dx'][0, 0]                                    # sampling time of the input data
        y = data_dict['y']
        y = y.reshape(len(y))
        data_dict['y'] = y[int(86400/dx):]                            # first half is zero due to matlab to python conversion
        y_zero = np.where(y == 0)[0]
        perczero = len(y_zero)/len(y)                                 # perc of total zero data in y (data from the Obs run)

        cont = data_dict['cont']
        cont_dtype = cont.dtype
        cont_dict = {u: cont[str(u)] for u in cont_dtype.names}       # cont in data_dict is a structured array, I converted it in a dict
        data_dict['cont'] = cont_dict
        
        return data_dict, perczero
    
    else:
        
        mat = loadmat(path)                                               # load mat-file
        
        if key == 'sour':
    
            # SciPy reads in structures as structured NumPy arrays of dtype object
            # The size of the array is the size of the structure array, not the number-
            #  -elements in any particular field. The shape defaults to 2-dimensional.
            # For convenience make a dictionary of the data using the names from dtypes
            # Since the structure has only one element, but is 2-D, index it at [0, 0]                               
            
            mdata = mat['sour']                                           # variable in mat file
            mdtype = mdata.dtype                                          # dtypes of structures are "unsized objects"
            data_dict = {n: mdata[n][0, 0] for n in mdtype.names}         # express mdata as a dict
            
            return data_dict
        
        else:
            
            mdata = mat[key]                                              # variable in mat file
            mdtype = mdata.dtype                                          # dtypes of structures are "unsized objects"
            data_dict = {n: mdata[n][0, 0] for n in mdtype.names}         # express mdata as a dict
            
            dx = data_dict['dx'][0, 0]                                    # sampling time of the input data
            y = data_dict['y']
            y = y.reshape(len(y))
            data_dict['y'] = y[int(86400/dx):]                            # first half is zero due to matlab to python conversion
            y_zero = np.where(y == 0)[0]
            perczero = len(y_zero)/len(y)                                 # perc of total zero data in y (data from the Obs run)
            
            cont = data_dict['cont']
            cont_dtype = cont.dtype
            cont_dict = {u: cont[str(u)] for u in cont_dtype.names}       # cont in data_dict is a structured array, I converted it in a dict
            data_dict['cont'] = cont_dict                                 # now we have a fully accessible dict
        
            return data_dict, perczero




#############################################################################################################################




def get_data(path_bsd_gout, path_bsd_gsinj, amp, key = 'goutL', mat_v73 = False):       # take noise + signal
    
    """
    It takes some info from bsd_gout (dx, n, y, t0, inifr, bandw) and from bsd_gsinj (y) to obtain noise + signal for the filtering.
    Parameters:
    -----------------------------------------------------------------------------------------------------------------
    path_bsd_gout : (str) bsd_gout containing the interferometer's noise ---
    path_bsd_gsinj : (str) _gsinj containing the injected signal ---
    amp : (float) factor for injected signal amplitude wrt noise ---
    key : (str) keyword with info from L, H or V interferometer (insert gout + interf. or gsinj + interf., default = goutL) ---
    mat_v73 : (bool) if the matlab datafile version is -v7.3 insert the 'True' value (default = 'False') ---
    
    -----------------------------------------------------------------------------------------------------------------
    return:
        
    bsd_out: (dict) bsd with the info to use for filter data chunk and for the database
    -----------------------------------------------------------------------------------------------------------------     
    """    
    
    bsd_gout, perczero_gout = mat_to_dict(path_bsd_gout, key = key, mat_v73 = mat_v73)     # gout and perczero of y_gout 
    bsd_gsinj, perczero_gsinj = mat_to_dict(path_bsd_gsinj, key = 'gsinjL')                # gsinj and perczero of y_gsinj
    source = mat_to_dict(path_bsd_gsinj, key = 'sour')                                     # source
    
    
    dx = bsd_gout['dx'][0, 0]                  # sampling time of the input data
    y_gout = bsd_gout['y']                     # data from the gout bsd
    
    y_gsinj = bsd_gsinj['y']                   # data from the gsinj bsd
    
    try:
        t0_gout = bsd_gout['cont']['t0'][0, 0]      # starting time of the gout signal [days]
    except:
        t0_gout = bsd_gout['cont']['t0'][0, 0, 0]
    
    tcoe = source['tcoe'][0, 0]                     # starting time of the signal [days]
    t0_gout = np.int64(t0_gout)
    tcoe = np.int64(tcoe)
    
    t_ind = int(((tcoe - t0_gout + 1)*86400)/dx)    # index of gout data with respect to injected signal
    
    y_gout = y_gout[t_ind:t_ind + len(y_gsinj)]     # we take a chunk of y_gout that contain y_gsinj
    n_new = len(y_gout)                             # number of samples to consider for filter data chunk
    
    y_tot = amp*y_gsinj + y_gout                    # signal + noise
    
    y_goutzeros = np.where(y_gout == 0)[0]
    for z in y_goutzeros:
        y_tot[z] = 0 + 0j
    
    try:
        inifr = bsd_gout['cont']['inifr'][0, 0][0, 0]       # initial bin freq
        bandw = bsd_gout['cont']['bandw'][0, 0][0, 0]       # bandwidth of the bin
    except:
        inifr = bsd_gout['cont']['inifr'][0, 0, 0]          # initial bin freq
        bandw = bsd_gout['cont']['bandw'][0, 0, 0]          # bandwidth of the bin
        
    
    bsd_out = {'dx': dx,
               'n': n_new,
               'y': y_tot,
               'y_gout' : y_gout,
               'y_gsinj': y_gsinj,
               'perczero_gout': perczero_gout,
               'perczero_gsinj': perczero_gsinj,
               'inifr': inifr,
               'bandw': bandw,
               'amp' : amp,
               'source': source}
    
    return bsd_out




#############################################################################################################################




def Spectrograms_injsign(n_im, path_bsd_gout, path_bsd_gsinj, amp, lfft,
                 images = False, key= 'gout_58633_58638_295_300', mat_v73=True):
    
    # directory for spectrograms
    directory = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Spectrograms_im/"

    for j in range(n_im):
        
        # dict with data
        bsd_data = get_data(path_bsd_gout, path_bsd_gsinj + str(j + 1) + '.mat',
                            amp=amp, key = key, mat_v73 = mat_v73)
        
        data = bsd_data['y']       # data: signal + noise             
        dx = bsd_data['dx']        # sampling time
        
        # spectrogram
        F, T, spectrogram = signal.spectrogram(data, fs=1/dx, window = 'cosine',
                                               nperseg = lfft - 1 , noverlap=lfft//2,
                                               nfft=lfft, return_onesided=True, scaling='density')
        
        if images:
            
            im_title = 'Spectrogram_' + str(j + 1)        # images titles
            
            plt.figure(num = j, tight_layout = True)
            plt.pcolormesh(T/86400, fftshift(F), np.log10(fftshift(spectrogram, axes=0)), cmap='viridis', shading='gouraud')
            plt.axis('off')
            plt.title('')
            plt.savefig(directory + im_title + '.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        
        else:
            ## plot spectrogram
            plt.figure(num = j, figsize = (12, 12), tight_layout = True)             
            # a = plt.pcolormesh(T/86400, F, np.log10(spectrogram), cmap='viridis', shading='gouraud')
            a = plt.pcolormesh(T/86400, fftshift(F), np.log10(fftshift(spectrogram, axes=0)), cmap='viridis', shading='gouraud')
            plt.colorbar(a)
            plt.title('Spectrogram' + str(j + 1))
            plt.ylabel('frequency [Hz]')
            plt.xlabel('time [days]')
            plt.show()
            



#############################################################################################################################




def Spectrograms_noise(n_im, path_bsd_gout, lfft, images = False,
                       key= 'gout_58633_58638_295_300', mat_v73=True):
    
    # directory for spectrograms
    directory = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Spectrograms_im/"
    
    # dict with data
    bsd_data = mat_to_dict(path_bsd_gout, key = key, mat_v73 =  mat_v73)

    data = bsd_data['y']            # data: noise             
    dx = bsd_data['dx']             # sampling time

    for j in range(n_im):
        
        
        
        # spectrogram
        F, T, spectrogram = signal.spectrogram(data, fs=1/dx, window = 'cosine',
                                               nperseg = lfft - 1 , noverlap=lfft//2,
                                               nfft=lfft, return_onesided=True, scaling='density')
        
        if images:
            
            im_title = 'Spectrogram_' + str(j + 1)        # images titles
            
            plt.figure(num = j, tight_layout = True)
            plt.pcolormesh(T/86400, fftshift(F), np.log10(fftshift(spectrogram, axes=0)), cmap='viridis', shading='gouraud')
            plt.axis('off')
            plt.title('')
            plt.savefig(directory + im_title + '.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        
        else:
            ## plot spectrogram
            plt.figure(num = j, figsize = (12, 12), tight_layout = True)             
            # a = plt.pcolormesh(T/86400, F, np.log10(spectrogram), cmap='viridis', shading='gouraud')
            a = plt.pcolormesh(T/86400, fftshift(F), np.log10(fftshift(spectrogram, axes=0)), cmap='viridis', shading='gouraud')
            plt.colorbar(a)
            plt.title('Spectrogram' + str(j + 1))
            plt.ylabel('frequency [Hz]')
            plt.xlabel('time [days]')
            plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        