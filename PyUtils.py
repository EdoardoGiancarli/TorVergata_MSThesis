# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 01:58:50 2023

@author: Edoardo Giancarli
"""

#### ModuleMSThesis - Useful function for modules version 4 ###############################################

####   libraries   #####

import numpy as np                                # operations
from tqdm import tqdm                             # loop progress bar
from scipy import signal                          # spectrograms
from scipy.fft import fftshift
import os
import cv2                                        # images visualization
from astropy.time import Time
import random
import shutil

from scipy.io import loadmat                      # matlab datafile
import hdf5storage as hd                          # matlab datafile -v7.3  

import matplotlib.pyplot as plt                   # plotting
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=26, usetex=True)

import pandas as pd                               # dataframe
import PyLTSim as psm                             # simulations

####    content    #####

# mat_to_dict (function): conversion from MATLAB data file to python dict
#
# extract_bsdgoutinfo (function): it extracts the principal information from the interferometer data
#
# spectr_comp (function): it computes the spectrogram of the input data
#
# spectr_plot (function): it plots the input spectrogram data
#
# cv2disp (function): display images with cv2


####    codes    #####

def mat_to_dict(path, key = 'goutL', mat_v73 = False):
    
    """
    Conversion from MATLAB data file to dict.
    -------------------------------------------------------------------------------------------------------
    Parameters:
        path (str): path of the MATLAB data file --
        key (str): keyword with info from L, H or V interferometer (default = goutL) --
        mat_v73 (bool): if the MATLAB datafile version is -v7.3 insert the 'True' value (default = 'False') --
    
    Return:
        data_dict (dict): dict from MATLAB data file --
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
    
    
def extract_bsdgoutinfo(bsd_gout, key= 'bsd_L_C00_sub', mat_v73=True):
    
    """
    It extract the principal information from the interferometer data.
    -----------------------------------------------------------------------------------------------------------------
    Parameters:
        bsd_gout (str, dict): bsd_gout containing the interferometer's noise (if the path to bsd_gout is inserted, then
                              the MATLAB data file will be converted in dict with the mat_to_dict function) --
        key (str): keyword with info from L, H or V interferometer (default = gout_58633_58638_295_300) --
        mat_v73 (bool): if the matlab datafile version is -v7.3 insert the 'True' value (default = 'True') --    
    
    Return:   
        gout (dict): dict with the info on the interferometer data
    """
    
    if isinstance(bsd_gout, str):
        bsd_gout = mat_to_dict(bsd_gout, key = key, mat_v73 = mat_v73)      # gout and perczero of y_gout
        
    dt = bsd_gout['dx'][0, 0]         # sampling time of the input data
    y_gout = bsd_gout['y']            # data from the gout bsd
    n = len(y_gout)                   # N of samples inside y_gout
    # v_eq = bsd_gout['cont']['v_eq']   # interferometer velocity values (eq coordinate)
    # p_eq = bsd_gout['cont']['p_eq']   # interferometer position values (eq coordinate)
    perczero = bsd_gout['perczero']   # perc of total zero data
    
    try:
        inifr = bsd_gout['cont']['inifr'][0, 0][0, 0]         # initial band frequency
        bandw = bsd_gout['cont']['bandw'][0, 0][0, 0]         # frequency bandwidth
        t0_gout = np.int64(bsd_gout['cont']['t0'][0, 0])      # starting time of the gout signal [mjd]
    except:
        try:
            inifr = bsd_gout['cont']['inifr'][0, 0, 0]            # for -v 7.3 MATLAB data file
            bandw = bsd_gout['cont']['bandw'][0, 0, 0]
            t0_gout = np.int64(bsd_gout['cont']['t0'][0, 0, 0])
        except:
            inifr = bsd_gout['cont']['inifr'][0, 0]                 # initial band frequency
            bandw = bsd_gout['cont']['bandw'][0, 0][0, 0]           # frequency bandwidth
            t0_gout = np.int64(bsd_gout['cont']['t0'][0, 0])        # starting time of the gout signal [mjd]
        
    gout = {'dt': dt, 'y_gout': y_gout, 'n': n,
            'inifr': inifr, 'bandw': bandw,
            't0_gout': t0_gout, 'perczero': perczero}
    
    return gout  
    
    
##########################################################################################################################


def spectr_comp(data, dt, lfft, setting_plot = False, normalisation = False, norm_value = None):
    """
    Spectrogram computation.
    -------------------------------------------------------------------------------------------------------
    Parameters:
        data (array): input data --
        dt (int, float): sampling time --
        lfft (int): fft lenght for the spectrogram --
        setting_plot (bool): if True the spectrogram values are shifted properly for a correct
                              visualisation and the frequency and time array are generated (defaut = False) --
        max_value (int, float): value for spectrogram normalisation (defaut = None) --
    
    Return:
        f (array): frequency array for the spectrogram (if setting_plot is True) --
        t (array): time array for the spectrogram (if setting_plot is True) --
        Sxx (array): spectrogram values --    
    """
    
    # spectrogram
    _, t, Sxx = signal.spectrogram(data, fs=1/dt, window = 'cosine', nperseg = lfft - 1,
                                   noverlap=lfft//8, nfft=lfft, return_onesided=False,
                                   scaling='density', mode='psd')
    
    # set the spectrogram matrix for visualisation
    if setting_plot:
        
        if normalisation:
            if norm_value is None:
                Sxx /= np.max(Sxx)
            else:
                Sxx /= norm_value
        
        # shift frequency zero component
        Sxx_shft = fftshift(Sxx, axes=0)
        
        # rotate matrix with spectrum and define frequency and time arrays
        Sxx_trs = np.vstack((Sxx_shft[int(lfft/2):, :], Sxx_shft[0:int(lfft/2), :]))
        
        freq_trs = np.arange(0, lfft)/(dt*lfft)
        
        return freq_trs, t, Sxx_trs
    
    else:
        return Sxx
    
    
##########################################################################################################################


def spectr_plot(t, f, Sxx, title = None, images = False, directory = None):
    """
    Spectrogram plot.
    -------------------------------------------------------------------------------------------------------
    Parameters:
        f (array): frequency array --
        t (array): time array --
        Sxx (array): spectrogram values --
        timelength (float): time lenght for the spectrogram [days] --
        title (str): plot title (default = None) --
        images (bool): if True the spectrograms are generated as images and saved
                        in directory (default = False) --
        directory (str): path for the directory to save the spectrograms images --
        
    Return: none    
    """
    
    if images: # spectrogram image
            
        if directory is None:
            raise ValueError('Specify the directory to save the spectrograms images.')
        
        figsize = (128/80, 128/80)
        dpi=150
                        
        plt.figure(num = None, figsize=figsize, dpi=dpi, tight_layout = True)
        plt.pcolormesh(t, f, np.log10(Sxx + 1e-15), cmap='viridis',
                       vmin=-13, vmax=-2, shading='gouraud')
        plt.axis('off')
        plt.title('')
        plt.savefig(directory + title + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    
    else: # plot spectrogram
        
        plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
        a = plt.pcolormesh(t/86400, f, np.log10(Sxx + 1e-15), cmap='viridis', shading='gouraud')
        plt.colorbar(a)
        
        if title is not None:
            plt.title(title)
        
        plt.ylabel('frequency [Hz]')
        plt.xlabel('time [days]')
        plt.show()


##########################################################################################################################


# display the images
def cv2disp(img_name, img, x_pos = 10, y_pos = 10):
    
    cv2.imshow(img_name, img/(np.max(img) + 1e-10))
    cv2.moveWindow(img_name, x_pos, y_pos)



# end