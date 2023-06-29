# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:19:34 2023

@author: Edoardo Giancarli
"""

import numpy as np
import pandas as pd
import PySim_v2 as psm
# import PySpectr_v3 as psr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#### 0 check: parameter function ---- yes [08/05/23]

# gw parameters
params = psm.parameters(days = 1, dt = 1/20, fgw0 = 125, tcoe = '2019-05-31 14:40:51.946', n = 5, k = None, tau = 1, Phi0 = 0, right_asc = '00 42 30',
                        dec = '+41 12 00', eta = -1, psi = 0, Einstein_delay = True, Doppler_effect = True, interferometer = 'LIGO-L')

# k-tau test: yes yes
# coord test: yes yes
# bsd_gout test: yes yes


#### 1st check: GW signal without Einstein effect and Doppler effect ---- yes [08/05/23]

# class check
gwsim = psm.GW_signal(params) # yes

t = gwsim.time_vec() # yes
iota = gwsim.iota(u = 'ddeg')  # yes
tau = gwsim.compute_tau(1.2167e-16) # yes
f = gwsim.frequency(plot = True) # yes
phi = gwsim.Phi(plot = True) # yes
h0 = gwsim.h0() # yes
H0 = gwsim.H0() # yes
Hplus = gwsim.H_plus() # yes
Hcross = gwsim.H_cross() # yes
Aplus = gwsim.A_plus(plot = False) # yes
Across = gwsim.A_cross(plot = False) # yes
h_t = gwsim.h_t(plot = True, values = 'abs') # yes


#### 2nd check: GW signal without Einstein effect and with Doppler effect ---- yes [08/05/23] 

# class check
gwsim = psm.GW_signal(params) # yes

t = gwsim.time_vec() # yes
iota = gwsim.iota(u = 'ddeg')  # yes
tau = gwsim.compute_tau(1.2167e-16) # yes
date = gwsim.UTCdate() # yes
f = gwsim.frequency(plot = True) # yes
phi = gwsim.Phi(plot = True) # yes
h0 = gwsim.h0() # yes
H0 = gwsim.H0() # yes
Hplus = gwsim.H_plus() # yes
Hcross = gwsim.H_cross() # yes
lst = gwsim.compute_lst() # yes
Aplus = gwsim.A_plus(plot = True) # yes
Across = gwsim.A_cross(plot = True) # yes
h_t = gwsim.h_t(plot = True, values = 'real') # yes
df = gwsim.to_pdDataframe(objects = 'data', save_to_csv = False, directory = None) # yes
df2 = gwsim.to_pdDataframe(objects = 'parameters', save_to_csv = False, directory = None) # ValueError: All arrays must be of the same length


#### 3rd check: GW signal with Einstein effect and with Doppler effect + data options ---- None []

# class check
gwsim = psm.GW_signal(params)

t = gwsim.time_vec() 
iota = gwsim.iota(u = 'ddeg')  
tau = gwsim.compute_tau(1.2167e-16) 
date_mjd = gwsim.MJDdate()
f = gwsim.frequency(plot = True)
eins = gwsim.EinsteinDelay()
phi = gwsim.Phi(plot = True) 
h0 = gwsim.h0() 
H0 = gwsim.H0()
Hplus = gwsim.H_plus() 
Hcross = gwsim.H_cross() 
lst = gwsim.compute_lst()
Aplus = gwsim.A_plus(plot = True) 
Across = gwsim.A_cross(plot = True) 
h_t = gwsim.h_t(plot = True, values = 'abs')
h_t = gwsim.h_t(plot = True, values = 'real') 
df = gwsim.to_pdDataframe(objects = 'data', save_to_csv = False, directory = None)
f, t, spectrogram = gwsim.spectr()



##############

n = 250

fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)    
ax = fig.add_subplot(111)
ax.plot(gwsim.time_vec()[:n]/86400, h_t.real[:n], label = '$Re(h(t))$')
ax.plot(gwsim.time_vec()[:n]/86400, h_t.imag[:n], label ='$Im(h(t))$')
plt.title('$Re(h(t))$ and $Im(h(t))$ displacement')
plt.ylabel('strain')
plt.xlabel('time [days]' + ' - start: ' + gwsim.signal_date + 'UTC')
plt.legend(loc='best')
ax.grid(True)
ax.label_outer()            
ax.tick_params(which='both', direction='in',width=2)
ax.tick_params(which='major', direction='in',length=7)
ax.tick_params(which='minor', direction='in',length=4)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.show()


h_real_EfDf = gwsim.h_t().real
h_real_EtDf = gwsim.h_t().real
h_real_EfDt = gwsim.h_t().real
h_real_EtDt = gwsim.h_t().real

fig, axs = plt.subplots(4, 1, figsize=(20, 14), sharex=True, tight_layout = True)

# plot the data on each subplot and save the lines for the legend + grid + ticks options
axs[0].plot(gwsim.time_vec()/86400, h_real_EtDf - h_real_EfDf, c = 'cyan')             
axs[1].plot(gwsim.time_vec()/86400, h_real_EfDt - h_real_EfDf, c = 'b')                
axs[2].plot(gwsim.time_vec()/86400, h_real_EtDt - h_real_EfDf, c = 'm')                
axs[3].plot(gwsim.time_vec()/86400, h_real_EtDt - h_real_EfDt, c = 'LawnGreen')                

for ind in range(4):
    axs[ind].plot([0, len(h_real_EfDf)*gwsim.dt/86400], [0, 0], c = 'r')
    axs[ind].grid(True)
    axs[ind].label_outer()                                              # ticks options                     
    axs[ind].tick_params(which='both', direction='in',width=2)
    axs[ind].tick_params(which='major', direction='in',length=7)
    axs[ind].tick_params(which='minor', direction='in',length=4)
    axs[ind].xaxis.set_ticks_position('both')
    axs[ind].yaxis.set_ticks_position('both')

axs[0].set_title('Einstein delay and Doppler effect action on $Re(h(t))$')
axs[1].set_title('')
axs[2].set_title('')
axs[3].set_title('')

axs[0].set_ylabel('$Re(h(t)) res.$')                                           # add y labels to the subplots
axs[1].set_ylabel('$Re(h(t)) res.$')
axs[2].set_ylabel('$Re(h(t)) res.$')
axs[3].set_ylabel('$Re(h(t)) res.$')
plt.xlabel('time [days]' + ' - start: ' + gwsim.signal_date + 'UTC')

# add a legend to the bottom subplot and adjust its position
patch1 = mpatches.Patch(color='cyan', label='wE - wtoD')
patch2 = mpatches.Patch(color='b', label='wtoE - wD')
patch3 = mpatches.Patch(color='m', label='wE - wD')
patch4 = mpatches.Patch(color='LawnGreen', label='wE - wD wrt wtoE - wD')

for ind, patch in zip([0, 1, 2, 3], [patch1, patch2, patch3, patch4]):
    axs[ind].legend(handles=[patch], loc='best')

# adjust the layout of the subplots to avoid overlapping
fig.tight_layout()
plt.show()




#### 1st check GW_injection ----- [11/05/23 - 12/05/23] oke - oke

path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/gout_58633_58638_295_300.mat"

params = psm.parameters(days = 1.85, dt = 0.5, fgw0 = 350, tcoe = '2019-05-31 14:00:51.946', n = 5, k = None, tau = 1.2, Phi0 = 0, right_asc = '00 42 30',
                        dec = '+41 12 00', eta = -1, psi = 0, Einstein_delay = True, Doppler_effect = True, interferometer = 'LIGO-L',
                        h0factor=1e22, signal_inj = True, bsd_gout = path_bsd_gout, key='gout_58633_58638_295_300', mat_v73=True) 

gwinj = psm.GW_injection(params, amp = 5)
f = gwinj.frequency
h = gwinj.h_t
obs_freq, obs_phi, obs_h, obs_t = gwinj.frequency_band(plot = True, observable='frequency')
y_chunk = gwinj.data_chunk()
y = gwinj.injection(plot = True, values = 'abs')
df = gwinj.to_pdDataframe(objects = 'data', save_to_csv = False, directory = None)
f, t, spectrogram = gwinj.spectr()



#### check simulated signal ########################################################################
df_path = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/check_sim_signals/GWsignalParameters.csv"
df1_path = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/check_sim_signals/GWsignalParameters1.csv"


df = pd.read_csv(df_path, index_col=0)
df1 = pd.read_csv(df1_path, index_col=0)


fgw0 = [df['fgw0'][6], df1['fgw0'][1], df1['fgw0'][3]]
tcoe = [df['tcoe'][6], df1['tcoe'][1], df1['tcoe'][3]]
tau = [df['tau'][6], df1['tau'][1], df1['tau'][3]]
right_asc = [df['right_asc'][6], df1['right_asc'][1], df1['right_asc'][3]]
dec = [df['dec'][6], df1['dec'][1], df1['dec'][3]]
eta = [df['eta'][6], df1['eta'][1], df1['eta'][3]]
psi = [df['psi'][6], df1['psi'][1], df1['psi'][3]]


for i in range(3):

    params = psm.parameters(days = 3, dt = 0.2, fgw0 = fgw0[i], tcoe = tcoe[i], n = 5, k = None, tau = tau[i], Phi0 = 0,
                            right_asc = right_asc[i], dec = dec[i], eta = eta[i], psi = psi[i], Einstein_delay = False,
                            Doppler_effect = True, interferometer = 'LIGO-L', h0factor=1e22, signal_inj = False,
                            bsd_gout = path_bsd_gout, key='gout_58633_58638_295_300', mat_v73=True)

    gwsim = psm.GW_signal(params) 
    gwsim.frequency(plot = True)




#### check spectrograms ########################################################
import numpy as np
import PySim_v2 as psm
import PySpectr_v3 as psr
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift

path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/gout_58633_58638_295_300.mat"

params = psm.parameters(days = 1.85, dt = 0.5, fgw0 = 350, tcoe = '2019-05-31 14:00:51.946', n = 5, k = None, tau = 1.2, Phi0 = 0, right_asc = '00 42 30',
                        dec = '+41 12 00', eta = -1, psi = 0, Einstein_delay = True, Doppler_effect = True, interferometer = 'LIGO-L',
                        h0factor=1e22, signal_inj = True, bsd_gout = path_bsd_gout, key='gout_58633_58638_295_300', mat_v73=True) 



gwinj = psm.GW_injection(params, amp = 10)
obs = gwinj.frequency_band(plot = True, observable = 'frequency', values = 'abs')
y = gwinj.injection(plot=True, values='abs')

psr.spectr(y.real, params['dt'], 1024, title = 'Spectrogram of $Re(h(t))$', images = False, directory = None)
psr.spectr(y.imag, params['dt'], 1024, title = 'Spectrogram of $Im(h(t))$', images = False, directory = None)
psr.spectr(y, params['dt'], int(1024*params['dt']), title = 'Spectrogram of $h(t)$', images = False, directory = None)



## check on h(t) fft ########################################################
y_fft = fft(y)
y_fft_shift = fftshift(y_fft)
freq = np.arange(len(y))/(len(y)*params['dt']) # come definite da numpy (e forse anche da scipy)

plt.figure(None)
plt.plot(freq, np.abs(y_fft))
plt.show()

fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)    
ax = fig.add_subplot(111)
ax.plot(freq, np.abs(y_fft), c='Lawngreen')
# ax.plot(freq, np.abs(y_fft_shift), c='Lawngreen')
# ax.plot(freq, y_fft.real, c='Lawngreen')
# ax.plot(freq, y_fft.imag, c='Lawngreen')
plt.title('Observed F(h(t)) (abs values)')
plt.ylabel('$F(h(t))$')
plt.xlabel('frequency [Hz]')
ax.grid(True)
ax.label_outer()            
ax.tick_params(which='both', direction='in',width=2)
ax.tick_params(which='major', direction='in',length=7)
ax.tick_params(which='minor', direction='in',length=4)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.show()


Sxx = np.zeros((int(1024*params['dt']), int(len(y)/int(1024*params['dt']))), dtype = complex)
               
for i in range(Sxx.shape[1]):
    Sxx[:, i] = y[i*Sxx.shape[0]:(i + 1)*Sxx.shape[0]]




fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
rng = np.random.default_rng()
noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise

from scipy import signal

f, t, Sxx = signal.spectrogram(x, fs)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


y_fft = fftshift(fft(y))
yy_fft = (y_fft + np.flip(y_fft))/2
yy = ifft(yy_fft)


f, t, Sxx = signal.spectrogram(yy.real, 1./params['dt'])
plt.figure(None)
plt.pcolormesh(t/86400, f, np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

plt.figure(None)
plt.plot(np.arange(len(y)), yy_fft)
plt.show()





## spectrograms with gwpy ########################################################
from gwpy.timeseries import TimeSeries
import PySim_v2 as psm
import matplotlib.pyplot as plt


path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/gout_58633_58638_295_300.mat"

params = psm.parameters(days = 1, dt = 0.5, fgw0 = 310, tcoe = '2019-05-31 14:00:51.946', n = 5, k = None, tau = 2, Phi0 = 0, right_asc = '00 42 30',
                        dec = '+41 12 00', eta = -1, psi = 0, Einstein_delay = True, Doppler_effect = True, interferometer = 'LIGO-L',
                        h0factor=1e22, signal_inj = True, bsd_gout = path_bsd_gout, key='gout_58633_58638_295_300', mat_v73=True) 

gwinj = psm.GW_injection(params, amp = 10)
obs = gwinj.frequency_band(plot = True, observable = 'frequency', values = 'abs')
y = gwinj.injection(plot=False, values='abs')

y_ts = TimeSeries(y)
spectr = y_ts.spectrogram(5, fftlength=4, overlap=None) ** (1/2.)

