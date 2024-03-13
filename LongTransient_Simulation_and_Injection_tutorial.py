# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:55:50 2024

@author: Edoardo Giancarli
"""

#### Tutorial for long transient gravitational signals (ltGW) simulation and injection in real data
#
# References:
#   Giancarli E., "Efficiency of Deep Learning in long transient Gravitational Waves classification",
#                 Master of Science in Astrophysics and Space Science Thesis, Tor Vergata University of Rome,
#                 Supervisor: Prof. Viviana Fafone, Co-Supervisor: Dr. Sabrina D'Antonio,
#                 Github: https://github.com/EdoardoGiancarli/TorVergata_MSThesis
#

#### Libraries
import PyLTSim as psm                        # module for long transient simulation and injection


#### Steps
#
# 1. Define the ltGW parameters
#
# 2a. Simulation of a ltGW
#
# 2b. Simulation and injection of a ltGW




#### 1. Define the ltGW parameters
params = psm.parameters(days=1, dt=0.01, fgw0=1400, tcoe=0, n=5, k=None, tau=1, Phi0=0, right_asc='05 34 30.9',
                        dec='+22 00 53', eta=-1, psi=0, Einstein_delay=False, Doppler_effect=False,
                        interferometer='LIGO-L', NS_parameters=[1e38, 1e-6, 1], h0factor=5e19,
                        signal_inj=False, bsd_gout=None, key='bsd_L_C00_sub', mat_v73=True)

#! Here the method "parameters" is used to define the parameters that characterize the ltGW (see the function "parameters"
#  in the PyLTSim module). Specifically, the following ltGW properties can be initialized:

#         days (int, float): duration of the GW signal in [days]
#         dt (float): sampling time in [s] of the signal
#         fgw0 (int, float): initial freq in [Hz] of the GW signal
#         tcoe (int, float, str): coalescing time/initial time in [days] or [date in UTC scale, iso format] of the simulated
#                                 long transient GW signal; if tcoe in [days] it represents the starting time wtr the actual
#                                 time date at which the signal is simulated (astropy.time.Time.now() date); tcoe can be <, =
#                                 or > 0 (if tcoe < 0, the signal is generated tcoe days in the past, default = 0); also if tcoe
#                                 is >= 57277 (2015-09-12, the begin of O1 run) will be considered as a date in MJD format and 
#                                 converted to iso-UTC date
#         n (int, float): breaking index (default = 5)
#         k (int, float): spin-down proportionality const (default = None)
#         tau (int, float): frequency characteristic time (spin-down timescale) in [days] (default = 1)
#         Phi0 (int, float): initial phase const in [decimal degrees] (default = 0)
#         right_asc (str, float): right ascension of the source in [hour angle] (str) or [decimal degree] (float)
#                                 (default = '05 34 30.9' from Crab pulsar, coord by SIMBAD archive)
#         dec (str, float): declination of the source in [deg] (str) or [decimal degree] (float) (default = '+22 00 53'
#                           from Crab pulsar, coord by SIMBAD archive)
#         eta (int, float): polarization degree, values between [-1, 1] (default = -1, corresponding to iota = 0)
#         psi (int, float): angle between source major axis wrt source celestial parallel (counterclockwise)
#                           in [decimal degrees] (default = 0)
#         Einstein_delay (bool): Einstein delay application (default = False)
#         Doppler_effect (bool): Doppler effect application (Doppler due to Earth rotation) (default = False)
#         interferometer (str): choose the interferometer (LIGO-L, LIGO-H, Virgo or KAGRA, defult = LIGO-L)
#         NS_parameters (list): this list contains the fiducial values of the ellipticity (default = 1e-6), moment of
#                               inertia (default = 1e38 [kg m**2]) and distance (default = 1 [kpc]) of a 1.4M_sun
#                               neutron star (these fiducial values set the constant of the h0(t) strain amplitude)
#         h0factor (float): multiplying factor for h0 to reduce the strain amplitude h(t) values (for computational costs,
#                           default = 1/2 * 1e+20; 1/2 because h(t) has complex values and 1e+20 because of the BSD data)
#
#
#
#         signal_inj (bool): if you want to inject the simulated signal into some interf noise, the data about the interf are
#                            also loaded (check the interferometer input, default = False)
#         bsd_gout (str, dict): bsd_gout containing the interferometer's data (if the path to bsd_gout is inserted, then
#                               the MATLAB data file will be converted in dict, default = None)
#         key : (str) keyword with info from L, H or V interferometer (default = 'bsd_L_C00_sub')
#         mat_v73 (bool): if the matlab datafile version is -v7.3 insert the 'True' value (default = 'True')




#### 2. Once the ltGW parameters are defined and stored in params (which is a dictionary), the next steps can be:
#       (2a.) to simply simulate a ltGW;
#       (2b.) to simulate a ltGW and inject it into real data from the chosen interferometer.


#### 2a. Simulation of a ltGW
# A ltGW can be simulated by using the GW_signal class (see the class "GW_signal" in the PyLTSim module)
# after having defined its parameters.

# variable with ltGW parameters (in this way we do not load the ltGW info each time, they are stored in gwsim)
gwsim = psm.GW_signal(params)


#! To see which info can be called see the Attributes of the "GW_signal" class in the PyLTSim module
#! To see which methods can be called see the Methods of the "GW_signal" class in the PyLTSim module

# from gwsim we can call different things, like all the ltGW parameters or its strain (complex) amplitude:

# f(t), Phi(t) and the other properties of the ltGW can be studied singularly (and shown in the respective plot)
frequency = gwsim.frequency(plot = True)
phi = gwsim.Phi(plot = True) 
h0 = gwsim.h0() 
H0 = gwsim.H0()
Hplus = gwsim.H_plus() 
Hcross = gwsim.H_cross() 
Aplus = gwsim.A_plus(plot = True) 
Across = gwsim.A_cross(plot = True)

# the ltGW strain (complex) amplitude can be studied choosing its real, imag or abs values
h_t = gwsim.h_t(plot = True, values = 'abs')

# the spectrogram of the ltGW can be called with the "spectr" method inside the GW_signal class
f, t, spectrogram = gwsim.spectr(freq_band=[1300, 1400], lfft = 1024, normalisation = True,
                                 sim_noise = True, amp_noise = 1e-3)

# the ltGW data (or the parameters) can be saved in a dataframe with the "to_pdDataframe" method
df = gwsim.to_pdDataframe(objects = 'data', save_to_csv = False, directory = None)



#### 2b. Simulation and injection of a ltGW
# A ltGW can be simulated and injected by using the GW_injection class (see the class "GW_injection" in the PyLTSim module)
# after having defined its parameters (!!! in this case, the info about the interferometer data have to be inserted in the
# "parameters" method input; see the last four input of the method described above).

# variable with ltGW parameters (in this way we do not load the ltGW info each time, they are stored in gwsim)
gwinj = psm.GW_injection(params)


#! To see which info can be called see the Attributes of the "GW_injection" class in the PyLTSim module
#! To see which methods can be called see the Methods of the "GW_injection" class in the PyLTSim module

# from gwinj we can call again different things, like the ltGW parameters or its strain (complex) amplitude in
# the data segment that has been specified in the "parameters" method:

# f(t), Phi(t) and the other properties of the ltGW can be studied singularly as before through the "gws" Attribute
frequency = gwinj.gws.frequency(plot = True)
phi = gwinj.gws.Phi(plot = True) 
h0 = gwinj.gws.h0() 
H0 = gwinj.gws.H0()
Hplus = gwinj.gws.H_plus() 
Hcross = gwinj.gws.H_cross() 
Aplus = gwinj.gws.A_plus(plot = True) 
Across = gwinj.gws.A_cross(plot = True)

# again, the total ltGW strain (complex) amplitude can be studied choosing its real, imag or abs values
h_t = gwsim.gws.h_t(plot = True, values = 'abs')

#! these quantities above represent the entire simulated ltGW (step 2a.)


# In the injection procedure, the strain is cutted in the right time and frequency intervals depending on the inserted data
obs_freq, obs_phi, obs_h, obs_t = gwinj.frequency_band(plot = True, observable='strain', values='abs')

#! these quantities above represent the segments of the ltGW frequency, phase, strain and time indicated by the inserted data
#! (all these four arrays can be plotted by changing the "observable" input)

# The total data (s(t) = h(t) + n(t)) can be directly retrieved with the "injection" method
s_t = gwinj.injection(amp = 1, plot = True, values = 'abs')


# the spectrogram of the injected ltGW can be called with the "spectr" method inside the GW_injection class
f, t, spectrogram = gwinj.spectr(amp = 1, lfft = 512, plot=False, image = True, directory = "path",
                                 normalisation = False, norm_value = None, _img_gen=False, _namefig='title', _offset=False)

# the injected ltGW data (or the parameters) can be saved in a dataframe with the "to_pdDataframe" method
df = gwinj.to_pdDataframe(objects = 'data', save_to_csv = False, directory = None)



# end
