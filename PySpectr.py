# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 00:34:00 2023

@author: Edoardo Giancarli
"""

#### ModuleMSThesis - Spectrograms simulation version 11 ###############################################

####   libraries   #####

import numpy as np                                # operations
from tqdm import tqdm                             # loop progress bar
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=26, usetex=True)
from astropy.time import Time

import PyLTSim as psm                             # signal simulation/signal injection
import PyUtils as pu                              # interferometer data

####    content    #####

# noise_spectr (function): it computes the noise spectrograms images
# 
# sng_spectr (function): it computes the signals spectrograms images


def _random_tcoe(tcoe, start_point=0, interval=60):
    
    t = tcoe + np.random.uniform(start_point, interval)
    t += np.random.normal(0, 10)
    t += np.random.normal(0, 5)
    t += np.random.normal(0, 1)
    t += np.random.normal(0, 0.3)
    
    if t < (tcoe + start_point):
        t += np.random.uniform(np.abs(t - tcoe - start_point), interval)
    
    elif (t - tcoe) > interval:
        t -= np.random.uniform(0, interval - start_point)
    
    return t
    
####    codes    #####


def noise_spectr(n_imgs, path_bsd_gout, timelength = 0.1, lfft = 1024,
                 normalisation = False, norm_value = None, images = False, directory = None,
                 title1 = None, directory2 = None, title2=None,
                 key= 'bsd_L_C00_sub', mat_v73=True, n=0):
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
    bsd_gout = pu.mat_to_dict(path_bsd_gout, key = key, mat_v73 = mat_v73)
    
    data = bsd_gout['y']                         # noise data    
    dt = bsd_gout['dx'][0, 0]                    # sampling time of the input data
    timelenght_chunk = int(86400*timelength/dt)  # each spectrogram covers timelength day
    
    if n_imgs > (len(data)//timelenght_chunk):
        raise ValueError(f"Oops, too many spectrograms: max N images is {len(data)//timelenght_chunk} with",
                         f"each spectrograms that covers {timelength} days")
    
    else:
        for j in tqdm(range(n_imgs)):
            
            # random initialisation
            i = np.random.randint(0, len(data) - timelenght_chunk)
            
            # select data chunk
            data_chunk = data[i : i + timelenght_chunk]
            
            # if there are too many zeros it skips the chunk
            if len(np.where(data_chunk == 0)[0])/timelenght_chunk < 0.1:
                freq, time, spectrogram = pu.spectr_comp(data_chunk, dt, lfft, setting_plot = True,
                                                         normalisation = normalisation, norm_value = norm_value)
                
                pu.spectr_plot(time, freq, 5*(spectrogram + np.random.normal(0, 0.5)*1e-12),
                               title=title1 + str(s + n), images=images, directory=directory)
                
                if directory2 is not None:
                    pu.spectr_plot(time, freq, 10*spectrogram, title=title2 + str(s + n),
                                   images=images, directory=directory2)
                
                # update number of computed spectrogram
                s += 1
    
    print(' ############################################\n',
          "Percentage of computed spectrograms wrt input n_im {:.2f}%\n".format((s/n_imgs)*100),
          '############################################')
         

##########################################################################################################################

def sgn_spectr(n_imgs, lfft = 1024, norm = False, norm_value=None, images = True,
               directory=None, directory2=None, path_bsd_gout = None, key='bsd_L_C00_sub',
               mat_v73=True, n=0, title=None, title2=None):    
    """
    Spectrograms of the injected signals.
    -------------------------------------------------------------------------------------------------------
    Parameters:
        n_imgs (int): number of images to generate --
        lfft (int): fft lenght for the spectrogram --
        norm (bool): spectrogram normalisation (defaut = False) --
        norm_value (int, float): value for spectrogram normalisation (defaut = None) --
        images (bool): if True spectrograms_noise generates and saves the spectrograms
                        images in directory (default = False) --
        directory (str): path for the directory to save the spectrograms images --
        path_bsd_gout (str): bsd_gout containing the interferometer's noise --
        key (str): keyword with info from L, H or V interferometer (default = goutL) --
        mat_v73 (bool): if the matlab datafile version is -v7.3 insert the 'True' value (default = 'False') --
    
    Return: none    
    """
    
    # group signal parameters inside a dict
    params = psm.parameters(days = 1, dt = 0, fgw0 = 0, Einstein_delay = True,
                            signal_inj = True, bsd_gout = path_bsd_gout, key=key, mat_v73=mat_v73)
    
    # initialize N of computed spectrograms
    s = 0
    
    # define lower frequency
    f0 = params['bsd_gout']['inifr'] + params['bsd_gout']['bandw']
    
    # define start day [mjd]
    tcoe = params['bsd_gout']['t0_gout']
        
    for j in tqdm(range(n_imgs)):
        
        # initialize random parameters for the long transient signals
        params['fgw0'] = f0 + np.random.randint(0, 10)*params['bsd_gout']['bandw']
        params['tau'] = 0.7 + np.random.uniform(0, 2)
        params['eta'] = np.random.uniform(-1, 1)
        params['psi'] = np.random.uniform(0, 90)*np.pi/180
        params['right_asc'] = np.random.uniform(0, 360)*np.pi/180
        params['dec'] = np.random.uniform(-90, 90)*np.pi/180
        params['tcoe'] = Time(_random_tcoe(tcoe), format='mjd', scale='utc').iso
        params['NS_parameters'] = [np.random.uniform(0.5, 1.5)*1e38,
                                   np.random.uniform(1, 1e3)*1e-6,
                                   np.random.uniform(6, 40)*1e3]     #!!! (1, 30)  (30, 420)  (4.2, 31)*1e2  (6, 40)*1e3
        
        # signal simulation and injection into noise
        gwinj = psm.GW_injection(params)
        
        try:
            gwinj.spectr(1, lfft, image=images, directory=directory, normalisation=norm, norm_value=norm_value,
                         _img_gen=True, _namefig=title + str(s + n), _offset=True)
            
            if directory2 is not None:
                gwinj.spectr(1e3, lfft, image=images, directory=directory2, normalisation=norm, norm_value=norm_value,
                             _img_gen=True, _namefig=title2 + str(s + n), _offset=False)
                        
            # update number of computed spectrogram
            s += 1
            
        except (ValueError, IndexError):
            pass
    
    # print % of computed spectrograms
    print(' ############################################\n',
          "Percentage of computed spectrograms wrt input n_im {:.2f}%\n".format((s/n_imgs)*100),
          '############################################')



# end