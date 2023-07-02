# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 00:34:00 2023

@author: Edoardo Giancarli
"""

#### ModuleMSThesis - Spectrograms simulation version 8 ###############################################

####   libraries   #####

import numpy as np                                # operations
from tqdm import tqdm                             # loop progress bar
from scipy import signal                          # spectrograms
from scipy.fft import fftshift

from scipy.io import loadmat                      # matlab datafile
import hdf5storage as hd                          # matlab datafile -v7.3  

import matplotlib.pyplot as plt                   # plotting
# import matplotlib.patches as mpatches
# from matplotlib.ticker import AutoMinorLocator
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=26, usetex=True)

import PySim as psm                               # signal simulation/signal injection
import pandas as pd                               # dataframe

####    content    #####

# mat_to_dict (function): conversion from MATLAB data file to python dict
#
# extract_bsdgoutinfo (function): it extracts the principal information from the interferometer data
#
# spectr_comp (function): it computes the spectrogram of the input data
#
# spectr_plot (function): it plots the input spectrogram data
#
# noise_spectr_maxdistr (function): it computes the noise spectrograms max values distribution
#
# noise_spectr (function): it computes the noise spectrograms images
#
# sng_spectr_maxdistr (function): it computes the signals spectrograms max values distribution
# 
# sng_spectr (function): it computes the signals spectrograms images

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
        perczero (float): perc of total zero data in the data
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
    
    Return:   
        gout (dict): dict with the info on the interferometer data
    """
    
    if isinstance(bsd_gout, str):
        bsd_gout = mat_to_dict(bsd_gout, key = key, mat_v73 = mat_v73)      # gout and perczero of y_gout
        
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


def spectr_comp(data, dt, lfft, setting_plot = False, max_value = None):
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
        
        if max_value is None:
            raise ValueError("Insert max value for normalisation.")
        else:
            Sxx /= max_value
        
        # shift frequency zero component
        Sxx_shft = fftshift(Sxx, axes=0)
        
        # rotate matrix with spectrum and define frequency and time arrays
        Sxx_trs = np.vstack((Sxx_shft[int(Sxx.shape[0]/2):, :],
                             Sxx_shft[0:int(Sxx.shape[0]/2), :]))
        
        freq_trs = np.arange(0, lfft)/(dt*lfft)
        
        return freq_trs, t, Sxx_trs
    
    else:
        return Sxx
    
    
##########################################################################################################################


def spectr_plot(t, f, Sxx, dt = None, lfft = None, title = None, images = False, directory = None):
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
                        
        plt.figure(num = None, figsize=(128/70, 128/70), dpi=100, tight_layout = True)
        plt.pcolormesh(t, f, np.log10(Sxx + 1e-15), cmap='viridis', shading='gouraud')
        plt.axis('off')
        plt.title('')
        plt.savefig(directory + title + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    
    else: # plot spectrogram
        
        plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
        a = plt.pcolormesh(t/86400, f, np.log10(Sxx + 1e-15),
                           cmap='viridis', shading='gouraud')
        plt.colorbar(a)
        
        if title is not None:
            plt.title(title)
        
        plt.ylabel('frequency [Hz]')
        plt.xlabel('time [days]')
        plt.show()


##########################################################################################################################


def noise_spectr_maxdistr(n_imgs, path_bsd_gout, timelength = 0.1, lfft = 1024,
                          key= 'gout_58633_58638_295_300', mat_v73=True):
    """
    Noise spectrograms max values distribution.
    -------------------------------------------------------------------------------------------------------
    Parameters:
        n_imgs (int): number of images to generate --
        path_bsd_gout (str): bsd_gout containing the interferometer's noise --
        timelength (float): time lenght for the spectrogram [days] --
        lfft (int): fft lenght for the spectrogram --
        key (str): keyword with info from L, H or V interferometer (default = goutL) --
        mat_v73 (bool): if the matlab datafile version is -v7.3 insert the 'True' value (default = 'False') --
    
    Return: 
        max_values_noise (list): noise spectrograms max distribution
        std_values_noise (list): noise spectrograms std distribution
    """
    
    # initialize max values list
    max_values_noise, std_values_noise = [], []
    
    # dict with data
    bsd_gout = mat_to_dict(path_bsd_gout, key = key, mat_v73 = mat_v73)

    data = bsd_gout['y']                         # noise data    
    dt = bsd_gout['dx'][0, 0]                    # sampling time of the input data
    timelenght_chunk = int(86400*timelength/dt)  # each spectrogram covers timelength day
    
    if n_imgs > (len(data)//timelenght_chunk):
        raise ValueError(f"Oops, too many spectrograms: max N images is {len(data)//timelenght_chunk} with\n",
                         f"each spectrograms that covers {timelength} days")
    
    else:
        for j in tqdm(range(n_imgs)):
            
            # random initialisation
            i = np.random.randint(0, int(len(data)/timelenght_chunk))
            
            # select data chunk
            data_chunk = data[i*timelenght_chunk : (i + 1)*timelenght_chunk]
            
            # if there are too many zeros it skips the chunk
            if len(np.where(data_chunk == 0)[0])/timelenght_chunk < 0.1:
                
                spectrogram = spectr_comp(data_chunk, dt, lfft)
                
                max_values_noise.append(np.max(spectrogram))
                std_values_noise.append(np.std(spectrogram))
    
    return max_values_noise, std_values_noise



def noise_spectr(n_imgs, path_bsd_gout, timelength = 0.1, lfft = 1024,
                 max_value = None, images = False, directory = None,
                 key= 'gout_58633_58638_295_300', mat_v73=True):
    """
    Spectrograms of the interferometer noise.
    -------------------------------------------------------------------------------------------------------
    Parameters:
        n_imgs (int): number of images to generate --
        path_bsd_gout (str): bsd_gout containing the interferometer's noise --
        timelength (float): time lenght for the spectrogram [days] --
        lfft (int): fft lenght for the spectrogram --
        max_value (int, float): value for spectrogram normalisation (defaut = None) --
        images (bool): if True spectrograms_noise generates and saves the spectrograms
                        images in directory (default = False) --
        directory (str): path for the directory to save the spectrograms images --
        key (str): keyword with info from L, H or V interferometer (default = goutL) --
        mat_v73 (bool): if the matlab datafile version is -v7.3 insert the 'True' value (default = 'False') --
    
    Return: none    
    """
    
    # initialize N of computed spectrograms
    s = 0
    
    # dict with data
    bsd_gout = mat_to_dict(path_bsd_gout, key = key, mat_v73 = mat_v73)
    
    data = bsd_gout['y']                         # noise data    
    dt = bsd_gout['dx'][0, 0]                    # sampling time of the input data
    timelenght_chunk = int(86400*timelength/dt)  # each spectrogram covers timelength day
    
    if n_imgs > (len(data)//timelenght_chunk):
        raise ValueError(f"Oops, too many spectrograms: max N images is {len(data)//timelenght_chunk} with\n",
                         f"each spectrograms that covers {timelength} days")
    
    else:
        for j in tqdm(range(n_imgs)):
            
            # random initialisation
            i = np.random.randint(0, int(len(data)/timelenght_chunk))
            
            # select data chunk
            data_chunk = data[i*timelenght_chunk : (i + 1)*timelenght_chunk]
            
            # if there are too many zeros it skips the chunk
            if len(np.where(data_chunk == 0)[0])/timelenght_chunk < 0.1:
                freq, time, spectrogram = spectr_comp(data_chunk, dt, lfft, setting_plot = True,
                                                      max_value = max_value)
                
                spectr_plot(time, freq, spectrogram, title='noise_spectr_' + str(s),
                            images=images, directory=directory)
                
                # update number of computed spectrogram
                s += 1
    
    print(' ############################################\n',
          "Percentage of computed spectrograms wrt input n_im {:.2f}%\n".format((s/n_imgs)*100),
          '############################################')
         

##########################################################################################################################


def sgn_spectr_maxdistr(n_imgs, path_bsd_gout, lfft = 1024,
                        key= 'gout_58633_58638_295_300', mat_v73=True):
    """
    Signals spectrograms max values distribution.
    -------------------------------------------------------------------------------------------------------
    Parameters:
        n_imgs (int): number of images to generate --
        path_bsd_gout (str): bsd_gout containing the interferometer's noise --
        lfft (int): fft lenght for the spectrogram --
        key (str): keyword with info from L, H or V interferometer (default = goutL) --
        mat_v73 (bool): if the matlab datafile version is -v7.3 insert the 'True' value (default = 'False') --
    
    Return: 
        max_values_noise (list): signals spectrograms max distribution --
        std_values_noise (list): signals spectrograms std distribution
    """
    
    # initialize max values list
    max_values_noise, std_values_noise = [], []
    
    for j in tqdm(range(n_imgs)):
                        
        # initialize random parameters for the long transient signals
        fgw0 = 301.01 + np.random.uniform(0, 20)
        tcoe = 58633 + np.random.uniform(0, 4)
        tau = 1 + np.random.uniform(0, 5)
        eta = np.random.uniform(-1, 1)
        psi = np.random.uniform(0, 90)
        amp = np.random.randint(1, 100)*1e-1
        right_asc = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 90)
        
        # group signal parameters inside a dict
        params = psm.parameters(days = 3, dt = 0.5, fgw0 = fgw0, tcoe = tcoe, n = 5, k = None, tau = tau, Phi0 = 0,
                                right_asc = right_asc, dec = dec, eta = eta, psi = psi, Einstein_delay = False,
                                Doppler_effect = True, interferometer = 'LIGO-L', signal_inj = True,
                                bsd_gout = path_bsd_gout, key=key, mat_v73=mat_v73)
        
        # generate and inject the signals
        gwinj = psm.GW_injection(params, amp = amp)
        
        try:
            y = gwinj.injection()    # signal + noise data
            
            # if there are too many zeros it skips the chunk
            if len(np.where(y == 0)[0])/len(y) < 0.1:
                
                spectrogram = spectr_comp(y, params['dt'], lfft)
                
                max_values_noise.append(np.max(spectrogram))
                std_values_noise.append(np.std(spectrogram))
                                
        except ValueError:
            pass
        except IndexError:
            pass
    
    return max_values_noise, std_values_noise



def sgn_spectr(n_imgs, path_bsd_gout, lfft = 1024, max_value = None, images = False,
               directory = None, key= 'gout_58633_58638_295_300', mat_v73=True):
    """
    Spectrograms of the interferometer noise.
    -------------------------------------------------------------------------------------------------------
    Parameters:
        n_imgs (int): number of images to generate --
        path_bsd_gout (str): bsd_gout containing the interferometer's noise --
        lfft (int): fft lenght for the spectrogram --
        max_value (int, float): value for spectrogram normalisation (defaut = None) --
        images (bool): if True spectrograms_noise generates and saves the spectrograms
                        images in directory (default = False) --
        directory (str): path for the directory to save the spectrograms images --
        key (str): keyword with info from L, H or V interferometer (default = goutL) --
        mat_v73 (bool): if the matlab datafile version is -v7.3 insert the 'True' value (default = 'False') --
    
    Return: none    
    """
    
    # initialize N of computed spectrograms
    s = 0
    
    for j in tqdm(range(n_imgs)):
        
        print('############################################\n')
                        
        # initialize random parameters for the long transient signals
        fgw0 = 301.01 + np.random.uniform(0, 20)
        tcoe = 58633 + np.random.uniform(0, 4)
        tau = 1 + np.random.uniform(0, 5)
        eta = np.random.uniform(-1, 1)
        psi = np.random.uniform(0, 90)
        amp = np.random.randint(1, 100)*1e-1
        right_asc = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 90)
        
        # group signal parameters inside a dict
        params = psm.parameters(days = 3, dt = 0.5, fgw0 = fgw0, tcoe = tcoe, n = 5, k = None, tau = tau, Phi0 = 0,
                                right_asc = right_asc, dec = dec, eta = eta, psi = psi, Einstein_delay = False,
                                Doppler_effect = True, interferometer = 'LIGO-L', signal_inj = True,
                                bsd_gout = path_bsd_gout, key=key, mat_v73=mat_v73)
        
        # generate and inject the signals
        gwinj = psm.GW_injection(params, amp = amp)
        
        try:
            y = gwinj.injection()    # signal + noise data
            
            # if there are too many zeros it skips the chunk
            if len(np.where(y == 0)[0])/len(y) < 0.1:
                print("# Spectrogram computed!\n")
                
                freq, time, spectrogram = spectr_comp(y, params['dt'], lfft, setting_plot = True,
                                                      max_value = max_value)
                
                spectr_plot(time, freq, spectrogram, title='sgn_spectr_' + str(s),
                            images=images, directory=directory)
                
                # update number of computed spectrogram
                s += 1
                                
        except ValueError:
            pass
        except IndexError:
            pass

    # print % of computed spectrograms
    print(' ############################################\n',
          "Percentage of computed spectrograms wrt input n_im {:.2f}%\n".format((s/n_imgs)*100),
          '############################################')


##########################################################################################################################


def save_to_csv():
    
    return 1


def load_from_csv():
    
    return 1



# end