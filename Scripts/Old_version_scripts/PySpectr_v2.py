# -*- coding: utf-8 -*-
"""
Created on Fri May  5 23:14:07 2023

@author: Edoardo Giancarli
"""

#### DriverMSThesis - Spectrograms simulation version 2 ###############################################

#### libraries

import numpy as np                                # operations
from scipy import signal                          # spectrograms
from scipy.fft import fftshift, ifftshift

from scipy.io import loadmat                      # matlab datafile
import hdf5storage as hd                          # matlab datafile -v7.3  

import matplotlib.pyplot as plt                   # plotting
# import matplotlib.patches as mpatches
# from matplotlib.ticker import AutoMinorLocator
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=26, usetex=True)


#### content

# mat_to_dict: Conversion from MATLAB data file to python dict
#
# get_data: It takes some info from bsd_gout and from bsd_gsinj to obtain noise + amp*signal
#
# spectrograms_injsign: It makes the spectrogram of the injected signal within the interferometer noise.
#
# spectrograms_noise: It makes the spectrogram of the interferometer noise.

#### codes


def mat_to_dict(path, key = 'goutL', data_file = 'noise', mat_v73 = False):  # convert the data from MATLAB: it seems that is better to leave goutL as a dict
    
    """
    Conversion from MATLAB data file to dict.
    Parameters:
    -------------------------------------------------------------------------------------------------------
    path : (str) path of the MATLAB data file --
    key : keyword with info from L, H or V interferometer (default = goutL) --
    data_file : (str) keyword to identified the kind of data contained in the MATLAB
                      data file ('noise' (default) or 'signal') --
    mat_v73 : (bool) if the MATLAB datafile version is -v7.3 insert the 'True' value (default = 'False') --
    -------------------------------------------------------------------------------------------------------
    return:
    data_dict: (dict) dict from MATLAB data file --
    perczero: (float) perc of total zero data in the data
    -------------------------------------------------------------------------------------------------------
    """
    
    if mat_v73:       
        
        mat = hd.loadmat(path)                                        # load mat-file -v7.3 

        mdata = mat[key]                                              # variable in mat file
        mdtype = mdata.dtype                                          # dtypes of structures are "unsized objects"
        data_dict = {n: mdata[n][0] for n in mdtype.names}            # express mdata as a dict
        
        y = data_dict['y']
        y = y.reshape(len(y))
        
        if data_file == 'noise':
            data_dict['y'] = y
        elif data_file == 'signal':
            data_dict['y'] = y[y != 0]                                # take only the signal
        
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
            
            y = data_dict['y']
            y = y.reshape(len(y))

            if data_file == 'noise':
                data_dict['y'] = y
            elif data_file == 'signal':
                data_dict['y'] = y[y != 0]                                # take only the signal
            
            y_zero = np.where(y == 0)[0]
            perczero = len(y_zero)/len(y)                                 # perc of total zero data in y (data from the Obs run)
            
            cont = data_dict['cont']
            cont_dtype = cont.dtype
            cont_dict = {u: cont[str(u)] for u in cont_dtype.names}       # cont in data_dict is a structured array, I converted it in a dict
            data_dict['cont'] = cont_dict                                 # now we have a fully accessible dict
        
            return data_dict, perczero




#############################################################################################################################




def get_data(path_bsd_gout, path_bsd_gsinj, amp, key = 'goutL', mat_v73 = False):
    
    """
    It takes some info from bsd_gout and from bsd_gsinj to obtain noise + signal.
    Parameters:
    -------------------------------------------------------------------------------------------------------
    path_bsd_gout : (str) bsd_gout containing the interferometer's noise --
    path_bsd_gsinj : (str) _gsinj containing the injected signal --
    amp : (float) factor for injected signal amplitude wrt noise --
    key : (str) keyword with info from L, H or V interferometer (default = goutL) --
    mat_v73 : (bool) if the matlab datafile version is -v7.3 insert the 'True' value (default = 'False') --
    -------------------------------------------------------------------------------------------------------
    return:
    bsd_out: (dict) bsd with the info from the injected signal and the interferometer noise
    -------------------------------------------------------------------------------------------------------     
    """    
    
    bsd_gout, perczero_gout = mat_to_dict(path_bsd_gout, key = key,
                                          data_file = 'noise', mat_v73 = mat_v73)                     # gout and perczero of y_gout
    bsd_gsinj, perczero_gsinj = mat_to_dict(path_bsd_gsinj, key = 'gsinjL', data_file = 'signal')     # gsinj and perczero of y_gsinj
    source = mat_to_dict(path_bsd_gsinj, key = 'sour')                                                # source
    
    
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
    
    t_ind = int(((tcoe - t0_gout)*86400)/dx)        # index of gout data with respect to injected signal
    
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




def spectrograms_injsign(n_im, path_bsd_gout, path_bsd_gsinj, amp, lfft,
                 images = False, key= 'gout_58633_58638_295_300', mat_v73=True):
    
    """
    It makes the spectrogram of the injected signal within the interferometer noise.
    Parameters:
    -------------------------------------------------------------------------------------------------------
    n_im : (int) number of images to generate --
    path_bsd_gout : (str) bsd_gout containing the interferometer's noise --
    path_bsd_gsinj : (str) _gsinj containing the injected signal --
    amp : (float) factor for injected signal amplitude wrt noise --
    lfft : (int) fft lenght for the spectrogram --
    images : (bool) if True Spectrograms_injsign generates and saves the spectrograms
                    images in directory (default = False) --
    key : (str) keyword with info from L, H or V interferometer (default = goutL) --
    mat_v73 : (bool) if the matlab datafile version is -v7.3 insert the 'True' value (default = 'False') --
    -------------------------------------------------------------------------------------------------------
    return: None
    -------------------------------------------------------------------------------------------------------     
    """
    
    # directory for spectrograms
    directory = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Spectrograms_im/"

    for j in range(n_im):
        
        # dict with data
        bsd_data = get_data(path_bsd_gout, path_bsd_gsinj + str(j + 1) + '.mat',
                            amp=amp, key = key, mat_v73 = mat_v73)
        
        data = ifftshift(bsd_data['y'])       # data: signal + noise (zero values shifted to the center for better fft evaluation)         
        dx = bsd_data['dx']                   # sampling time
        
        # spectrogram
        F, T, spectrogram = signal.spectrogram(data, fs=1/dx, window = 'cosine',
                                               nperseg = lfft - 1 , noverlap=lfft//2,
                                               nfft=lfft, scaling='density')
        
        im_title = 'Spectrogram_' + str(j + 1)        # images titles

        if images:
            
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
            plt.title(im_title)
            plt.ylabel('frequency [Hz]')
            plt.xlabel('time [days]')
            plt.show()
            



#############################################################################################################################




def spectrograms_noise(n_im, path_bsd_gout, lfft, images = False,
                       key= 'gout_58633_58638_295_300', mat_v73=True):
    
    """
    It makes the spectrogram of the interferometer noise.
    Parameters:
    -------------------------------------------------------------------------------------------------------
    n_im : (int) number of images to generate --
    path_bsd_gout : (str) bsd_gout containing the interferometer's noise --
    lfft : (int) fft lenght for the spectrogram --
    images : (bool) if True Spectrograms_injsign generates and saves the spectrograms
                    images in directory (default = False) --
    key : (str) keyword with info from L, H or V interferometer (default = goutL) --
    mat_v73 : (bool) if the matlab datafile version is -v7.3 insert the 'True' value (default = 'False') --
    -------------------------------------------------------------------------------------------------------
    return: None
    -------------------------------------------------------------------------------------------------------    
    """
    
    # directory for spectrograms
    directory = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/Spectrograms_noise/"
    
    # dict with data
    bsd_gout, perczero_gout = mat_to_dict(path_bsd_gout, key = key,
                                          data_file = 'noise', mat_v73 = mat_v73)      # gout and perczero of y_gout 

    data = bsd_gout['y']                   # data: noise         
    dx = bsd_gout['dx'][0, 0]              # sampling time of the input data
    timelenght_chunk = int(86400/(4*dx))   # each spectrogram covers 1/4 day
    
        
    if n_im > (len(data)//timelenght_chunk):
        
        print('Oops, too many spectrograms..slow down: max N images is {a}'.format(a = len(data)//timelenght_chunk))
    
    else:

        for j in range(n_im):
            
            data_chunk = ifftshift(data[j*timelenght_chunk : (j + 1)*timelenght_chunk])     # shift zero values to the center for fft evaluation
            
            if len(np.where(data_chunk == 0)[0])/timelenght_chunk < 0.3:                    # if there are too many zeros it skips the chunk
            
                # spectrogram
                F, T, spectrogram = signal.spectrogram(data_chunk, fs=1/dx, window = 'cosine', nperseg = lfft - 1,
                                                       noverlap=lfft//2, nfft=lfft, scaling='density')
                
                # print(len(data_chunk), len(F), len(T), spectrogram.shape)
                
                im_title = 'Noise spectrogram_' + str(j + 1)        # images titles
    
                if images:
                                    
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
                    plt.title(im_title)
                    plt.ylabel('frequency [Hz]')
                    plt.xlabel('time [days]')
                    plt.show()
            
            else:
                pass




#############################################################################################################################
