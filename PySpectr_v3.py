# -*- coding: utf-8 -*-
"""
Created on Sat May 13 17:01:05 2023

@author: Edoardo Giancarli
"""

#### ModuleMSThesis - Spectrograms simulation version 3 ###############################################

####   libraries   #####

import numpy as np                                # operations
from tqdm import tqdm                             # loop progress bar
from scipy import signal                          # spectrograms
from scipy.fft import fftshift, ifftshift

from scipy.io import loadmat                      # matlab datafile
import hdf5storage as hd                          # matlab datafile -v7.3  

import matplotlib.pyplot as plt                   # plotting
# import matplotlib.patches as mpatches
# from matplotlib.ticker import AutoMinorLocator
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=26, usetex=True)

import PySim_v2 as psm                            # signal simulation/signal injection
import pandas as pd                               # dataframe

####    content    #####

# mat_to_dict (function): conversion from MATLAB data file to python dict
#
# extract_bsdgoutinfo (function): it extracts the principal information from the interferometer data
#
# load_simsignalinfo (function): it loads the simulated signals data
#
# spectr (function): it makes the spectrograms
#
# spectrograms_noise (function): it makes the spectrograms of the interferometer noise

####    codes    #####


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
  
    
def rotate(arr):
    
    r, c = arr.shape[0], arr.shape[1]
    
    A = arr[:r//2, :c//2]
    B = arr[:r//2, c//2:c]
    C = arr[r//2:r, :c//2]
    D = arr[r//2:r, c//2:c]
    
    rot_arr = np.block([[D, C], [B, A]])
    
    return rot_arr
    
    
##########################################################################################################################
   
    
def spectr(data, dt, lfft, title = None, images = False, directory = None):
    
    # spectrogram
    F, T, spectrogram = signal.spectrogram(data, fs=1/dt, window = 'cosine', nperseg = lfft - 1,
                                           noverlap=lfft//2, nfft=lfft, return_onesided=False,
                                           scaling='density', mode='psd')
    
    # sp_log = np.log10(fftshift(spectrogram, axes=0))
    # sp_norm = (sp_log - np.mean(sp_log))/np.std(sp_log)
    
    # print(len(data_chunk), len(F), len(T), spectrogram.shape)
    # vmin = np.min(np.log10(sp_norm)), vmax = np.max(np.log10(sp_norm))
    
    if images: # spectrogram image
    
        if directory is None:
            raise ValueError('Specify the directory to save the spectrograms images')
        else:
            pass
                        
        plt.figure(num = None, figsize=(128/70, 128/70), dpi=100, tight_layout = True)
        plt.pcolormesh(T/86400, fftshift(F), np.log10(fftshift(spectrogram, axes=0)), cmap='viridis',
                       vmin = -13, vmax = -3, shading='gouraud')
        plt.axis('off')
        plt.title('')
        plt.savefig(directory + title + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    
    else: # plot spectrogram
        
        plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
        # a = plt.pcolormesh(T/86400, F, np.log10(spectrogram), cmap='viridis', shading='gouraud')
        a = plt.pcolormesh(T/86400, fftshift(F), np.log10(fftshift(spectrogram, axes=0)), cmap='viridis',
                           vmin = -13, vmax = -3, shading='gouraud')
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
    images (bool): if True spectrograms_noise generates and saves the spectrograms
                    images in directory (default = False) --
    directory (str): path for the directory to save the spectrograms images --
    key (str): keyword with info from L, H or V interferometer (default = goutL) --
    mat_v73 (bool): if the matlab datafile version is -v7.3 insert the 'True' value (default = 'False') --
    -------------------------------------------------------------------------------------------------------
    return: None
    -------------------------------------------------------------------------------------------------------    
    """
    
    # initialize N of computed spectrograms
    s = 0
    
    # dict with data
    bsd_gout = mat_to_dict(path_bsd_gout, key = key, mat_v73 = mat_v73)      # gout and perczero of y_gout 

    data = bsd_gout['y']                   # data: noise         
    dt = bsd_gout['dx'][0, 0]              # sampling time of the input data
    c = 4
    timelenght_chunk = int(86400/(c*dt))   # each spectrogram covers 1/4 day
       
    if n_im > (len(data)//timelenght_chunk):
        print(f"Oops, too many spectrograms..slow down: max N images is {len(data)//timelenght_chunk} with\n",
              f"each spectrograms that covers 1/{c} day")
    
    else:
        for j in tqdm(range(n_im)):
            
            # shift zero values to the center for fft evaluation
            data_chunk = ifftshift(data[j*timelenght_chunk : (j + 1)*timelenght_chunk])
            
            # if there are too many zeros it skips the chunk
            if len(np.where(data_chunk == 0)[0])/timelenght_chunk < 0.1:
                spectr(data_chunk, dt, lfft, title = 'noise_spectrogram_' + str(s), images = images,
                       directory = directory)
                
                # update number of computed spectrogram
                s += 1
            
            else:
                pass
    
    print(' ############################################\n',
          "Percentage of computed spectrograms wrt input n_im {:.2f}%\n".format((s/n_im)*100),
          '############################################')


##########################################################################################################################


def spectrograms_injsignal(n_im, lfft, images = False, directory = None, store_results = False, save_to_csv = False):
    
    """
    It makes the spectrograms of the randomly generated n_im simulated GW signals injected into the
    interferometer noise (interf data defined inside the function).
    -------------------------------------------------------------------------------------------------------
    Parameters:
    n_im (int): number of images to generate --
    lfft (int): fft lenght for the spectrogram --
    images (bool): if True Spectrograms_injsignal generates and saves the spectrograms
                    images in directory (default = False) --
    directory (str): path for the directory to save the spectrograms images --
    store_results (bool): if True, it stores the signals parameters in lists --
    save_to_csv (bool): if True, it saves the DataFrame with the data (default = False) --
    -------------------------------------------------------------------------------------------------------
    return:
    par (dict): dictionary with the parameter of the signal whose spectrogram has been computed
    -------------------------------------------------------------------------------------------------------    
    """
    
    path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/gout_58633_58638_295_300.mat"
    
    # initialize lists for the long transient signals parameters
    if store_results:
        fgw0_list, tcoe_list, tau_list = [], [], []
        eta_list, psi_list, amp_list = [], [], []
        right_asc_list, dec_list = [], []
    else:
        pass
    
    # initialize N of computed spectrograms
    s = 0
    
    for j in tqdm(range(n_im)):
        
        print('############################################\n')
                
        # initialize random parameters for the long transient signals
        fgw0 = 301.01 + np.random.uniform(0, 20)
        tcoe = 58633 + np.random.uniform(0, 4)
        tau = 1 + np.random.uniform(0, 5)
        eta = np.random.uniform(-1, 1)
        psi = np.random.uniform(0, 90)
        amp = np.random.randint(1, 500)*1e-3
        right_asc = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 90)
        
        # group signal parameters inside a dict
        params = psm.parameters(days = 3, dt = 0.5, fgw0 = fgw0, tcoe = tcoe, n = 5, k = None, tau = tau, Phi0 = 0,
                                right_asc = right_asc, dec = dec, eta = eta, psi = psi, Einstein_delay = False,
                                Doppler_effect = True, interferometer = 'LIGO-L', h0factor=1e22, signal_inj = True,
                                bsd_gout = path_bsd_gout, key='gout_58633_58638_295_300', mat_v73=True)
        
        # generate and inject the signals
        gwinj = psm.GW_injection(params, amp = amp)
        
        try:
            y = gwinj.injection()
            
            # loop for the spectrograms (if there are too many zeros it skips the chunk)
            if len(np.where(y == 0)[0])/len(y) < 0.1:
                
                print("# Spectrogram computed!\n")
                
                # data: signal + noise (zero values shifted to the center for better fft evaluation)
                spectr(ifftshift(y), params['dt'], lfft, title = 'signal_spectrogramzzz_' + str(s),
                       images = images, directory = directory)
                
                # lists with parameters for accepted signals
                if store_results:
                    fgw0_list.append(fgw0)
                    tcoe_list.append(tcoe)
                    tau_list.append(tau)
                    eta_list.append(eta)
                    psi_list.append(psi)
                    amp_list.append(amp)
                    right_asc_list.append(right_asc)
                    dec_list.append(dec)
                else:
                    pass
                
                # update number of computed spectrogram
                s += 1
        
            else:
                pass
        
        except ValueError:
            pass
        except IndexError:
            pass
    
    # print % of computed spectrograms
    print(' ############################################\n',
          "Percentage of computed spectrograms wrt input n_im {:.2f}%\n".format((s/n_im)*100),
          '############################################')
    
    # store parameters for accepted signals
    if store_results:
        par = {'fgw0': fgw0_list,
               'tcoe': tcoe_list,
               'tau': tau_list,
               'eta': eta_list,
               'psi': psi_list,
               'amp': amp_list,
               'right_asc': right_asc_list,
               'dec': dec_list}
            
        if save_to_csv:
            if directory is None:
                raise ValueError('Specify the directory for the DataFrame')
            else:
                pd.DataFrame.from_dict(par).to_csv(directory + 'GWsignalParameters.csv')
        
        return par
    
    else:
        return "No parameters stored..."
        
    


# end