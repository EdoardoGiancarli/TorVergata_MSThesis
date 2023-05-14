# -*- coding: utf-8 -*-
"""
Created on Sat May 13 17:01:05 2023

@author: Edoardo Giancarli
"""

#### ModuleMSThesis - Spectrograms simulation version 3 ###############################################

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

# mat_to_dict (function): conversion from MATLAB data file to python dict
#
# extract_bsdgoutinfo (function): it extracts the principal information from the interferometer data
#
# load_simsignalinfo (function): it loads the simulated signals data
#
# spectr (function): it makes the spectrograms
#
# spectrograms_noise (function): it makes the spectrograms of the interferometer noise

#### codes


def mat_to_dict(path, key = 'goutL', mat_v73 = False):
    
    """
    Conversion from MATLAB data file to dict.
    -------------------------------------------------------------------------------------------------------
    Parameters:
    path (str): path of the MATLAB data file --
    key (str): keyword with info from L, H or V interferometer (default = goutL) --
    mat_v73 (bool): if the MATLAB datafile version is -v7.3 insert the 'True' value (default = 'False') --
    -------------------------------------------------------------------------------------------------------
    return:
    data_dict (dict): dict from MATLAB data file --
    perczero (float): perc of total zero data in the data
    -------------------------------------------------------------------------------------------------------
    """
    
    # SciPy reads in structures as structured NumPy arrays of dtype object
    # The size of the array is the size of the structure array, not the number-
    #  -elements in any particular field. The shape defaults to 2-dimensional.
    # For convenience make a dictionary of the data using the names from dtypes
    # Since the structure has only one element, but is 2-D, index it at [0, 0]
    
    if mat_v73:                                                    # load mat-file -v7.3 
        mat = hd.loadmat(path)                                        

        mdata = mat[key]                                           # variable in mat file
        mdtype = mdata.dtype                                       # dtypes of structures are "unsized objects"
        data_dict = {n: mdata[n][0] for n in mdtype.names}         # express mdata as a dict
        
        y = data_dict['y']
        y = y.reshape(len(y))                                      # adjust data format
        data_dict['y'] = y
        
        cont = data_dict['cont']
        cont_dtype = cont.dtype
        cont_dict = {u: cont[str(u)] for u in cont_dtype.names}    # cont conversion in dict
        data_dict['cont'] = cont_dict
        
    else:
        mat = loadmat(path)                                        # load mat-file

        mdata = mat[key]                                           # variable in mat file
        mdtype = mdata.dtype                                       # dtypes of structures are "unsized objects"
        data_dict = {n: mdata[n][0, 0] for n in mdtype.names}      # express mdata as a dict
        
        y = data_dict['y']
        y = y.reshape(len(y))                                      # adjust data format
        data_dict['y'] = y
        
        cont = data_dict['cont']
        cont_dtype = cont.dtype
        cont_dict = {u: cont[str(u)] for u in cont_dtype.names}    # cont conversion in dict
        data_dict['cont'] = cont_dict
    
    y_zero = np.where(y == 0)[0]
    data_dict['perczero'] = len(y_zero)/len(y)                     # perc of total zero data in y (data from the Obs run)
    
    return data_dict
    
    
##########################################################################################################################
    
    
def extract_bsdgoutinfo(bsd_gout, key= 'gout_58633_58638_295_300', mat_v73=True):
    
    """
    It extract the principal information from the interferometer data.
    -----------------------------------------------------------------------------------------------------------------
    Parameters:
    bsd_gout (str, dict): bsd_gout containing the interferometer's noise (if the path to bsd_gout is inserted, then
                          the MATLAB data file will be converted in dict with the mat_to_dict function) --
    key (str): keyword with info from L, H or V interferometer (default = gout_58633_58638_295_300) --
    mat_v73 (bool): if the matlab datafile version is -v7.3 insert the 'True' value (default = 'True') --    
    -----------------------------------------------------------------------------------------------------------------
    return:   
    gout (dict): dict with the info on the interferometer data
    -----------------------------------------------------------------------------------------------------------------
    """
    
    if isinstance(bsd_gout, str):
        bsd_gout = mat_to_dict(bsd_gout, key = key,
                               data_file = 'noise', mat_v73 = mat_v73)      # gout and perczero of y_gout
        
    dt = bsd_gout['dx'][0, 0]         # sampling time of the input data
    y_gout = bsd_gout['y']            # data from the gout bsd
    n = len(y_gout)                   # N of samples inside y_gout
    v_eq = bsd_gout['cont']['v_eq']   # interferometer velocity values (eq coordinate)
    p_eq = bsd_gout['cont']['p_eq']   # interferometer position values (eq coordinate)
    perczero = bsd_gout['perczero']   # perc of total zero data
    
    try:
        inifr = bsd_gout['cont']['inifr'][0, 0][0, 0]         # initial band frequency
        bandw = bsd_gout['cont']['bandw'][0, 0][0, 0]         # frequency bandwidth
        t0_gout = np.int64(bsd_gout['cont']['t0'][0, 0])      # starting time of the gout signal [mjd]
        ant = bsd_gout['cont']['ant'][0, 0]                   # info on the interferometer data: (antenna, cal and notes)
        cal = bsd_gout['cont']['cal'][0, 0]
        notes = bsd_gout['cont']['notes'][0, 0]
    except:
        inifr = bsd_gout['cont']['inifr'][0, 0, 0]            # for -v 7.3 MATLAB data file
        bandw = bsd_gout['cont']['bandw'][0, 0, 0]
        t0_gout = np.int64(bsd_gout['cont']['t0'][0, 0, 0])
        ant = bsd_gout['cont']['ant'][0, 0, 0]
        cal = bsd_gout['cont']['cal'][0, 0, 0]
        notes = bsd_gout['cont']['notes'][0, 0, 0]
        
    gout = {'dt': dt, 'y_gout': y_gout, 'n': n,
            'inifr': inifr, 'bandw': bandw,
            't0_gout': t0_gout, 'v_eq': v_eq, 'p_eq': p_eq,
            'info': ant + '_' + cal + notes, 'perczero': perczero}
    
    return gout  
    
    
##########################################################################################################################
  
    
def load_simsignalinfo():
    
    
    return 1
    
    
##########################################################################################################################
   
    
def spectr(data, dt, lfft, title = None, images = False, directory = None):
    
    # spectrogram
    F, T, spectrogram = signal.spectrogram(data, fs=1/dt, window = 'cosine', nperseg = lfft - 1,
                                           noverlap=lfft//2, nfft=lfft, scaling='density')
    
    # print(len(data_chunk), len(F), len(T), spectrogram.shape)
    
    if images: # spectrogram image
        if directory is None:
            raise ValueError('Specify the directory to save the spectrograms images')
        else:
            pass
        
        plt.figure(num = None, tight_layout = True)
        plt.pcolormesh(T/86400, fftshift(F), np.log10(fftshift(spectrogram, axes=0)), cmap='viridis', shading='gouraud')
        plt.axis('off')
        plt.title('')
        plt.savefig(directory + title + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    
    else: # plot spectrogram
        plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
        # a = plt.pcolormesh(T/86400, F, np.log10(spectrogram), cmap='viridis', shading='gouraud')
        a = plt.pcolormesh(T/86400, fftshift(F), np.log10(fftshift(spectrogram, axes=0)), cmap='viridis', shading='gouraud')
        plt.colorbar(a)
        plt.title(title)
        plt.ylabel('frequency [Hz]')
        plt.xlabel('time [days]')
        plt.show()


##########################################################################################################################


def spectrograms_noise(n_im, path_bsd_gout, lfft, images = False, directory = None,
                       key= 'gout_58633_58638_295_300', mat_v73=True):
    
    """
    It makes the spectrogram of the interferometer noise.
    -------------------------------------------------------------------------------------------------------
    Parameters:
    n_im (int): number of images to generate --
    path_bsd_gout (str): bsd_gout containing the interferometer's noise --
    lfft (int): fft lenght for the spectrogram --
    images (bool): if True Spectrograms_injsign generates and saves the spectrograms
                    images in directory (default = False) --
    directory (str): path for the directory to save the spectrograms images --
    key (str): keyword with info from L, H or V interferometer (default = goutL) --
    mat_v73 (bool): if the matlab datafile version is -v7.3 insert the 'True' value (default = 'False') --
    -------------------------------------------------------------------------------------------------------
    return: None
    -------------------------------------------------------------------------------------------------------    
    """
    
    # dict with data
    bsd_gout = mat_to_dict(path_bsd_gout, key = key, mat_v73 = mat_v73)      # gout and perczero of y_gout 

    data = bsd_gout['y']                   # data: noise         
    dt = bsd_gout['dx'][0, 0]              # sampling time of the input data
    c = 4
    timelenght_chunk = int(86400/(c*dt))   # each spectrogram covers 1/c day
       
    if n_im > (len(data)//timelenght_chunk):
        print(f"""Oops, too many spectrograms..slow down: max N images is {len(data)//timelenght_chunk} with
              each spectrograms that covers 1/{c} day""")
    
    else:
        for j in range(n_im):
            
            # shift zero values to the center for fft evaluation
            data_chunk = ifftshift(data[j*timelenght_chunk : (j + 1)*timelenght_chunk])     
            # if there are too many zeros it skips the chunk
            if len(np.where(data_chunk == 0)[0])/timelenght_chunk < 0.3:
                spectr(data_chunk, dt, lfft, title = 'Noise spectrogram_' + str(j + 1), images = images,
                       directory = directory)
            
            else:
                pass


# end
