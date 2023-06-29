# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 19:31:21 2023

@author: Edoardo Giancarli
"""

#### check spectrograms ########################################################
import numpy as np
import PySim_v2 as psm
import PySpectr_v3 as psr
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy import signal                       


def spectr(data, dt, lfft, title = None, images = False, directory = None, shift = True):
    
    # # spectrogram
    # if shift:
    #     data = ifftshift(data)
    # else:
    #     pass
    
    F, T, spectrogram = signal.spectrogram(data, fs=1/dt, window = 'cosine', nperseg = lfft - 1,
                                           noverlap=lfft//8, nfft=lfft, scaling='density', mode='psd')
    
    if shift:
        F = fftshift(F)
        spectrogram = fftshift(spectrogram, axes=0)
    else:
        pass
    
    if images: # spectrogram image
    
        if directory is None:
            raise ValueError('Specify the directory to save the spectrograms images')
        else:
            pass
                        
        plt.figure(num = None, figsize=(128/70, 128/70), dpi=100, tight_layout = True)        
        plt.pcolormesh(T/86400, F, np.log10(spectrogram), cmap='viridis',
                       vmin = -13, vmax = -1, shading='gouraud')
        plt.axis('off')
        plt.title('')
        plt.savefig(directory + title + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    
    else: # plot spectrogram
        
        plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
        a = plt.pcolormesh(T/86400, F, np.log10(spectrogram), cmap='viridis',
                           vmin = -13, vmax = -1, shading='gouraud')
        plt.colorbar(a)
        plt.title(title)
        plt.ylabel('frequency [Hz]')
        plt.xlabel('time [days]')
        plt.show()


################################################################################
path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/gout_58633_58638_295_300.mat"

params = psm.parameters(days = 1.85, dt = 0.5, fgw0 = 350, tcoe = '2019-05-31 14:00:51.946', n = 5, k = None, tau = 1.2, Phi0 = 0, right_asc = '00 42 30',
                        dec = '+41 12 00', eta = -1, psi = 0, Einstein_delay = True, Doppler_effect = True, interferometer = 'LIGO-L',
                        h0factor=1e22, signal_inj = True, bsd_gout = path_bsd_gout, key='gout_58633_58638_295_300', mat_v73=True) 



gwinj = psm.GW_injection(params, amp = 10)
obs = gwinj.frequency_band(plot = True, observable = 'frequency', values = 'abs')
y = gwinj.injection(plot=True, values='abs')

spectr(y.real, params['dt'], 1024, title = 'Spectrogram of $Re(h(t))$', images = False, directory = None)
spectr(y.imag, params['dt'], 1024, title = 'Spectrogram of $Im(h(t))$', images = False, directory = None)
spectr(y, params['dt'], 1024, title = 'Spectrogram of $h(t)$', shift = True)



F, T, spectrogram = signal.spectrogram(y, fs=1/params['dt'], window = 'cosine', nperseg = 1023,
                                       noverlap=512, nfft=1024, scaling='density', mode='psd')


spectr_pos = spectrogram[0:int(spectrogram.shape[0]/2), :]
F = F[:len(F)//2]

plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
a = plt.pcolormesh(T/86400, F + 295, np.log10(spectr_pos), cmap='viridis',
                   vmin = -13, vmax = -3, shading='gouraud')
plt.colorbar(a)
plt.ylabel('frequency [Hz]')
plt.xlabel('time [days]')
plt.show()


#### prova spectrograms
import numpy as np
import PySim_v2 as psm
import PySpectr_v4 as psr
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.signal import spectrogram        


dt = 0.1
t = np.arange(0, 4096, dt)

h = 2*np.sqrt(t)*np.exp(2j*np.pi*3*t)
y = 2*np.sqrt(t)*np.exp(2j*np.pi*(3 + 0.5*np.cos(2*np.pi*2*t))*t)

f1, t1, spectr1 = signal.spectrogram(h, fs=1/dt, window = 'cosine', nperseg = 1023,
                                  noverlap=512, nfft=1024, scaling='density', mode='psd')

f2, t2, spectr2 = signal.spectrogram(y, fs=1/dt, window = 'hann', nperseg = 1023,
                                  noverlap=512, nfft=1024, scaling='density', mode='psd')

# shift zero frequency component to the center with fftshift from scipy
f1_shft, f2_shft = fftshift(f1), fftshift(f2)
spectr1_shft, spectr2_shft = fftshift(spectr1, axes=0), fftshift(spectr2, axes=0)


plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
a = plt.pcolormesh(t1, f1_shft, np.log10(spectr1_shft), cmap='viridis', shading='gouraud')
plt.colorbar(a)
plt.title('Spectrogram of h (with shift in freq)')
plt.ylabel('frequency [Hz]')
plt.xlabel('time [s]')
plt.show()

plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
b = plt.pcolormesh(t2, f2_shft, np.log10(spectr2_shft), cmap='viridis', shading='gouraud')
plt.colorbar(b)
plt.title('Spectrogram of y (with shift in freq)')
plt.ylabel('frequency [Hz]')
plt.xlabel('time [s]')
plt.show()

#######################

def f(f0, tau, a, t):
    f = f0*(1 + t/tau)**(-a)
    
    fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)    
    ax = fig.add_subplot(111)
    ax.plot(t, f, c='b')
    plt.title('Frequency $f(t)$ of the simulated signal')
    plt.ylabel('frequency [Hz]')
    plt.xlabel('time [s]')
    ax.grid(True)
    plt.show()
    
    return f

def phi(phi0, f0, tau, a, t):
    return phi0 + (2*np.pi*f0*tau/(1 - a))*((1 + t/tau)**(1 - a) - 1)

def signal(f0, tau, a, t, phi0 = 0):
    
    freq = f(f0, tau, a, t)
    phase = np.exp(1j*phi(phi0, f0, tau, a, t))
    s = 10*freq*phase
    
    fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)    
    ax = fig.add_subplot(111)
    ax.plot(t, np.abs(s), c='OrangeRed')
    plt.title('Amplitude $s(t)$ of the simulated signal (abs values)')
    plt.ylabel('frequency [Hz]')
    plt.xlabel('time [s]')
    ax.grid(True)
    plt.show()
    
    return s, freq

dt = 1./10
lfft = 1024
t = np.arange(0, 86400, dt)
data = signal(10, 86400//4, 2, t)
s = data[0]

# freq = data[1]
# d_f = freq[0] - freq[-1]
# dt = 1./d_f

fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)    
ax = fig.add_subplot(111)
ax.plot(t/86400, np.abs(fft(s.real)), c='OrangeRed')

# plt.title('Spectrum of the simulated signal $F[s(t)]$ (abs values)')
# plt.xlabel('frequency [Hz]')
# plt.ylabel('$|F[s(t)]|$')

plt.title('Spectrum of the real part of the simulated signal $F[Re[s(t)]]$')
plt.xlabel('frequency [Hz]')
plt.ylabel('$|F[Re[s(t)]]|$')

# plt.title('Amplitude $s(t)$ of the simulated signal (abs values)')
# plt.ylabel('frequency [Hz]')
# plt.xlabel('time [days]')
ax.grid(True)
plt.show()

F, T, spectrogram = spectrogram(s, fs=1/dt, window = 'cosine', nperseg = lfft - 1,
                                noverlap=lfft//8, nfft=lfft, scaling='density', mode='psd')

f1_shft, spectr1_shft = fftshift(F), fftshift(spectrogram, axes=0)

plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
a = plt.pcolormesh(T/86400, f1_shft, np.log10(spectr1_shft), cmap='viridis', shading='gouraud')
plt.colorbar(a)
plt.title('Spectrogram of $s(t)$')
plt.ylabel('frequency [Hz]')
plt.xlabel('time [days]')
plt.show()

spectr_trs = np.vstack((spectr1_shft[int(spectrogram.shape[0]/2):, :], spectr1_shft[0:int(spectrogram.shape[0]/2), :]))
freq_trs = np.arange(0, len(f1_shft))/(dt*len(f1_shft))

plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
b = plt.pcolormesh(T/86400, freq_trs, np.log10(spectr_trs), cmap='viridis', shading='gouraud')
plt.colorbar(b)
plt.title('Spectrogram of $s(t)$')
plt.ylabel('frequency [Hz]')
plt.xlabel('time [days]')
plt.show()


#### check with a real GW signal

path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/gout_58633_58638_295_300.mat"

params = psm.parameters(days = 1.85, dt = 0.5, fgw0 = 350, tcoe = '2019-05-31 14:00:51.946', n = 5, k = None, tau = 1.2, Phi0 = 0, right_asc = '00 42 30',
                        dec = '+41 12 00', eta = -1, psi = 0, Einstein_delay = True, Doppler_effect = True, interferometer = 'LIGO-L',
                        h0factor=1e22, signal_inj = True, bsd_gout = path_bsd_gout, key='gout_58633_58638_295_300', mat_v73=True) 



gwinj = psm.GW_injection(params, amp = 10)
obs = gwinj.frequency_band(plot = True, observable = 'frequency', values = 'abs')
y = gwinj.injection(plot=True, values='abs')
lfft = 1024

F, T, spectrogram = sfs.spectrogram(y, fs=1/params['dt'], window = 'cosine', nperseg = lfft - 1,
                                    noverlap=lfft//8, nfft=lfft, scaling='density', mode='psd')

f1_shft, spectr1_shft = fftshift(F), fftshift(spectrogram, axes=0)

plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
a = plt.pcolormesh(T/86400, f1_shft, np.log10(spectr1_shft), cmap='viridis', shading='gouraud')
plt.colorbar(a)
plt.title('Spectrogram of $s(t)$')
plt.ylabel('frequency [Hz]')
plt.xlabel('time [days]')
plt.show()

spectr_trs = np.vstack((spectr1_shft[int(spectrogram.shape[0]/2):, :], spectr1_shft[0:int(spectrogram.shape[0]/2), :]))
freq_trs = np.arange(0, len(f1_shft))/(params['dt']*len(f1_shft))

plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
b = plt.pcolormesh((T + obs[-1][0])/86400, freq_trs, np.log10(spectr_trs), cmap='viridis', shading='gouraud')
plt.colorbar(b)
plt.title('Spectrogram of $s(t)$')
plt.ylabel('frequency [Hz]')
plt.xlabel('time [days]')
plt.show()




def spectr(data, dt, lfft, image = False, directory = None):
    
    F, T, spectrogram = sfs.spectrogram(data, fs=1/dt, window = 'cosine', nperseg = lfft - 1,
                                        noverlap=lfft//8, nfft=lfft, scaling='density', mode='psd')

    spectr_shft = fftshift(spectrogram, axes=0)

    spectr_trs = np.vstack((spectr_shft[int(spectrogram.shape[0]/2):, :], spectr_shft[0:int(spectrogram.shape[0]/2), :]))
    freq_trs = np.arange(0, lfft)/(dt*lfft)
    title = 'Spectrogram of $h(t)$'

    if image: # spectrogram image
    
        if directory is None:
            raise ValueError('Specify the directory to save the spectrograms image')
        else:
            pass
                        
        plt.figure(num = None, figsize=(128/70, 128/70), dpi=100, tight_layout = True)        
        plt.pcolormesh(T, freq_trs, np.log10(spectr_trs), cmap='viridis',
                       vmin = -13, vmax = -1, shading='gouraud')
        plt.axis('off')
        plt.title('')
        plt.savefig(directory + title + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    
    else: # plot spectrogram
        
        plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
        col_bar = plt.pcolormesh((T + obs[-1][0])/86400, freq_trs, np.log10(spectr_trs), cmap='viridis',
                                 vmin = -13, vmax = -1, shading='gouraud')
        plt.colorbar(col_bar)
        plt.title(title)
        plt.ylabel('frequency [Hz]')
        plt.xlabel('time [days]')
        plt.show()


def spectr(data, dt, lfft, title = None, images = False, directory = None):
    
    # spectrogram
    F, T, spectrogram = signal.spectrogram(data, fs=1/dt, window = 'cosine', nperseg = lfft - 1,
                                           noverlap=lfft//8, nfft=lfft, scaling='density', mode='psd')
    
    spectr_shft = fftshift(spectrogram, axes=0)

    spectr_trs = np.vstack((spectr_shft[int(spectrogram.shape[0]/2):, :], spectr_shft[0:int(spectrogram.shape[0]/2), :]))
    freq_trs = np.arange(0, lfft)/(dt*lfft)
    
    if images: # spectrogram image
    
        if directory is None:
            raise ValueError('Specify the directory to save the spectrograms images')
        else:
            pass
                        
        plt.figure(num = None, figsize=(128/70, 128/70), dpi=100, tight_layout = True)
        plt.pcolormesh(T, fftshift(F), np.log10(spectr_trs), cmap='viridis',
                       vmin = -13, vmax = -1, shading='gouraud')
        plt.axis('off')
        plt.title('')
        plt.savefig(directory + title + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    
    else: # plot spectrogram
        
        plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
        a = plt.pcolormesh(T/86400, freq_trs, np.log10(spectr_trs), cmap='viridis',
                           vmin = -13, vmax = -3, shading='gouraud')
        plt.colorbar(a)
        plt.title(title)
        plt.ylabel('frequency [Hz]')
        plt.xlabel('time [days]')
        plt.show()



















