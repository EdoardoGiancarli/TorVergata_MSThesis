# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 02:46:08 2023

@author: Edoardo Giancarli
"""

import PySim as psm

#### signal simulation #############################################################
# gw parameters
params = psm.parameters(days = 1, dt = 1/20, fgw0 = 125, tcoe = '2019-05-31 14:40:51.946', n = 5, k = None, tau = 1, Phi0 = 0, right_asc = '00 42 30',
                        dec = '+41 12 00', eta = -1, psi = 0, Einstein_delay = True, Doppler_effect = True, interferometer = 'LIGO-L')

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



#### signal simulation and injection #############################################################
# path for interferometer data
path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/Master Thesis/Tesi_Codes/gout_58633_58638_295_300.mat"

# gw parameters
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



# end