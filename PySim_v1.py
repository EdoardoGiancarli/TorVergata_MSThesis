# -*- coding: utf-8 -*-
"""
Created on Sat May  6 01:55:26 2023

@author: Edoardo Giancarli
"""

#### DriverMSThesis - Signal simulation version 1 ###############################################

####   libraries   #####

import numpy as np                                # operations
import pandas as pd                               # dataframe  

import PySpectr_v2 as psr                         # spectrograms

import matplotlib.pyplot as plt                   # plotting
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=26, usetex=True)

# coordinates, units and sidereal time computation
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic, EarthLocation
import astropy.units as u
from astropy.time import Time


####    content    #####

# extract_bsdgoutinfo (function)
#
# parameters (function)
#
# GW_signal (class)
#
# GW_injection (class)


####    codes    #####

## initialization for the signal to be simulated

def extract_bsdgoutinfo(bsd_gout, key= 'gout_58633_58638_295_300', mat_v73=True):
    
    """
    It extract the principal information from the interferometer data.
    Parameters:
    -----------------------------------------------------------------------------------------------------------------
    bsd_gout (str, dict): bsd_gout containing the interferometer's noise (if the path to bsd_gout is inserted, then
                          the MATLAB data file will be converted with ) --
    key (str): keyword with info from L, H or V interferometer (default = gout_58633_58638_295_300) --
    mat_v73 (bool): if the matlab datafile version is -v7.3 insert the 'True' value (default = 'True') --    
    -----------------------------------------------------------------------------------------------------------------
    return:   
    gout (dict): dict with the parameters to simulate the signal
    -----------------------------------------------------------------------------------------------------------------
    """
    
    if isinstance(bsd_gout, str):
        bsd_gout, perczero_gout = psr.mat_to_dict(bsd_gout, key = key,
                                                  data_file = 'noise', mat_v73 = mat_v73)      # gout and perczero of y_gout
        
    dx = bsd_gout['dx'][0, 0]         # sampling time of the input data
    y_gout = bsd_gout['y']            # data from the gout bsd
    n = len(y_gout)                   # N of samples inside y_gout
    v_eq = bsd_gout['cont']['v_eq']
    p_eq = bsd_gout['cont']['p_eq']
    
    try:
        inifr = bsd_gout['cont']['inifr'][0, 0][0, 0]         # initial bin freq
        bandw = bsd_gout['cont']['bandw'][0, 0][0, 0]         # bandwidth of the bin
        t0_gout = np.int64(bsd_gout['cont']['t0'][0, 0])      # starting time of the gout signal [days]
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
        
    gout = {'dx': dx, 'y_gout': y_gout, 'n': n,
            'inifr': inifr, 'bandw': bandw,
            't0_gout': t0_gout, 'v_eq': v_eq, 'p_eq': p_eq,
            'info': ant + '_' + cal + notes}
    
    return gout   # !!! forse anche tfstr (in cont)



## define parameters for the signal to be simulated

def parameters(days, dx, fgw0, tcoe = 0, n = 5, k = 1.2167e-16, tau = None, Phi0 = 0, right_asc = '05 34 30.9',
               dec = '+22 00 53', eta = -1, psi = 0, Einstein_delay = False, Doppler_effect = False,
               interferometer = 'LIGO-L', NS_parameters = [1e38, 1e-6, 1], h0factor = 1e23,
               signal_inj = False, bsd_gout = None, key='gout_58633_58638_295_300', mat_v73=True):
    
    """
    It takes the arguments and gruop them in a dictionary, defining the parameters to be utilized.
    Parameters:
    -----------------------------------------------------------------------------------------------------------------
    days (int, float): duration of the GW signal in [days] --
    dx (float): sampling time in [s] of the signal --
    fgw0 (int, float): initial freq in [Hz] of the GW signal --
    tcoe (int, float, str): coalescing time/initial time in [days] or [date in UTC scale, iso format] of the simulated
                            long transient GW signal; if tcoe in [days] it represents the starting time wtr the actual
                            time date at which the signal is simulated (astropy.time.Time.now() date); tcoe can be <, =
                            or > 0 (if tcoe < 0, the signal is generated tcoe days in the past, default = 0) --
    n (int, float): breaking index (default = 5) --
    k (int, float): spin-down proportionality const (if given, default = 1.2167e-16, which gives tau = 1 (\pm 0.005)
                    in [s] for an fgw0 = 125 [Hz] and n = 5) --
    tau (int, float): frequency characteristic time (spin-down timescale) in [days] (if given, default = None) --
    Phi0 (int, float): initial phase const in [decimal degrees] (default = 0 since the interf work in dark fringe) --
    right_asc (str): right ascension of the source in [hour angle] (default = '05 34 30.9' from Crab pulsar,
                     coord by SIMBAD archive) --
    dec (str): declination of the source in [deg] (default = '+22 00 53' from Crab pulsar, coord by SIMBAD archive) --
    eta (int, float): polarization degree, values between [-1, 1] (default = -1, corresponding to iota = 0) --
    psi (int, float): angle between source major axis wrt source celestial parallel (counterclockwise)
                      in [decimal degrees] (default = 0) --
    Einstein_delay (bool): Einstein delay application (default = False) --
    Doppler_effect (bool): Doppler effect application (default = False) --
    interferometer (str): choose the interferometer (LIGO-L, LIGO-H, Virgo or KAGRA, defult = LIGO-L) --
    NS_parameters (list): this list contains the fiducial values of the ellipticity (default = 1e-6), moment of
                          inertia (default = 1e38 [kg m**2]) and distance (default = 1 [kpc]) of a 1.4M_sun
                          neutron star (these fiducial values set the constant of the h0(t) strain amplitude) --
    h0factor (float): multiplying factor for h0 to reduce the strain amplitude h(t) values (for computational costs,
                      default = 1e23) --
    signal_inj (bool): if you want to inject the simulated signal into some interf noise, the data about the interf are
                       also loaded (check the interferometer input, default = False) --
    bsd_gout (str, dict): bsd_gout containing the interferometer's data (if the path to bsd_gout is inserted, then
                          the MATLAB data file will be converted with extract_bsdgoutinfo (PySim), default = None) --
    key : (str) keyword with info from L, H or V interferometer (default = 'gout_58633_58638_295_300') --
    mat_v73 (bool): if the matlab datafile version is -v7.3 insert the 'True' value (default = 'True') --    
    -----------------------------------------------------------------------------------------------------------------
    return:
    par (dict): dict with the parameters to simulate the signal
        
        - additional argument: source ecliptical coordinates [long, lat] in [rad] (GeocentricTrueEcliptic frame),
                               interferometer's longitude, latitude, X-arm azimut in [rad] and elevation in [m],
                               bsd_gout (dict) (if signal_inj = True)
    -----------------------------------------------------------------------------------------------------------------     
    """
    
    ## choice between k or tau (since tau = tau(k))
    if tau is None and k is None:
        raise ValueError('Specify k or tau')
    elif tau is not None and k is not None:
        raise ValueError('Specify k or tau')
    
    ## check on the tcoe format
    if isinstance(tcoe, str):
        try:
            Time(tcoe, format='iso', scale='utc')
        except ValueError:
            raise ValueError("The date format is wrong: if tcoe in [date iso UTC] it has to be represented with 'yyyy-mm-dd hh:mm:ss'")
    
    ## interferometer's latitude[rad], longitude[rad], azimut[rad] (!!! X-arm azimut) and elevation [m]
    # url: https://lscsoft.docs.ligo.org/lalsuite/lal/group___detector_constants.html
    # Ref.: LIGO Scientic Collaboration, "LIGO Algorithm Library - LALSuite, free software" (2021)
    # if not in radiant: to multiply by *np.pi/180
    if interferometer == 'LIGO-L':               
        lat, long, azmt, interf_heigth = 0.53342313506, -1.58430937078, 4.40317772346, -6.574
    elif interferometer == 'LIGO-H':
        lat, long, azmt, interf_heigth = 0.81079526383, -2.08405676917, 5.65487724844, 142.554
    elif interferometer == 'Virgo':
        lat, long, azmt, interf_heigth = 0.76151183984, 0.18333805213, 0.33916285222, 51.884
    elif interferometer == 'KAGRA':
        lat, long, azmt, interf_heigth = 0.6355068497, 2.396441015, 1.054113, 414.181
    else:
        raise ValueError('Specify interferometer')
    
    ## create SkyCoord object with (astropy)
    coord = SkyCoord(right_asc + ' ' + dec, frame = 'icrs', unit=(u.hourangle, u.deg))

    # coordinates wrt ecliptic
    ecl_coord = coord.transform_to(GeocentricTrueEcliptic())
        
    ## dictionary with source and signal parameters
    par = {'tcoe': tcoe,
           'days': days,
           'dx_signal': dx,
           'fgw0': fgw0,
           'n': n,
           'k': k,                                                              # spin-down const [s**(1/n)]
           'tau': tau,
           'Phi0': Phi0*np.pi/180,                                              # Phi0 [rad]
           'right_asc': coord.ra.radian,                                        # right ascension [rad]
           'dec': coord.dec.radian,                                             # declination [rad]
           'eta': eta,
           'psi': psi*np.pi/180,                                                # psi [rad]
           'Einstein_delay': Einstein_delay,
           'Doppler_effect': Doppler_effect,
           'ecliptical_coord': [ecl_coord.lon.radian, ecl_coord.lat.radian],    # ecliptic longitude and latitude [rad]
           'interf_lat': lat,
           'interf_long': long,
           'interf_azmt': azmt,
           'interf_heigth': interf_heigth,
           'NS_parameters': NS_parameters,
           'h0factor': h0factor}
    
    ## dict containing the interferometer's data and info for the injection
    if signal_inj:
        if bsd_gout is None:
            raise ValueError('Insert bsd_gout')
        else:
            bsd_gout = extract_bsdgoutinfo(bsd_gout, key=key, mat_v73=mat_v73)
            par['dx_signal'] = None
            par['bsd_gout'] = bsd_gout
    else:
        pass
    
    return par


Einstein_effect

#############################################################################################################################




#### class: long transient GW simulation

class GW_signal:
    """
    This class simulate a long transient GW signal from an isolated neutron star
    using the supplied parameters and options.
    -----------------------------------------------------------------------------
    Attributes:
        tcoes (int, float): coalescing time/initial time in [s] of the signal (from tcoe in input dict)
        signal_date: date (UTC) of the GW signal generation
        days (int, float): duration of the GW signal in [days]
        dx (float): sampling time in [s] of the signal
        fgw0 (float): initial freq in [Hz] of the GW signal
        n (int): breaking index
        tau (int, float): frequency characteristic time (spin-down timescale) in [days]
        Phi0 (int, float): initial phase const in [rad]
        right_asc (float): right ascension of the source in [rad]
        dec (float): declination of the source in [rad]
        eta (int, float): polarization degree, values between [-1, 1]
        psi (float): angle between source major axis wrt source celestial parallel (counterclockwise) in [rad]
        lat (float): interferometer latitude (see parameters function) in [rad]
        long (float): interferometer longitude (see parameters function) in [rad]
        azmt (float): interferometer azimut (see parameters function) in [rad]
        heigth (float): interferometer elevation (see parameters function) in [m]
        Einstein_delay (bool): Einstein effect application
        Doppler_effect (bool): Doppler effect application
        NS_parameters (list): fiducial values of eps, I [kg m**2] and d [kpc] for a 1.4 M_sun neutron star (in h0, #[5]#)
        h0factor (float): multiplying factor for h0 to reduce the strain amplitude h(t) values (for computational cost)
        iota (float): inclination angle of the NS rot axis wrt interferometer in [ddeg]

    Methods:
        time_vec: gives the time vector in [s]
        UTCdate: computes the iso-UTC format date of the GW signal starting time if tcoe is an int/float (if called as a class
                 module, UTCdate return the actual time in iso-UTC format each time is called)
        MJDdate: computes the MJD format date of the GW signal starting time.
        iota: gives the inclination angle (iota) in [rad] or [decimal degrees] of the system wrt interferometer, linked to
              the GW polarization degree (eta) #[2, 3, 4]#
        compute_tau: gives the frequency characteristic time in [s] #[1]#
        frequency: gives the GW signal frequency f(t) in [Hz] #[1]#
        EinsteinDelay: compute the Einstein delay and its influence on Phi(t) #[]# !!!
        Phi: gives the phase shift Phi(t) in [rad] of the GW signal (from f(t) integration) #[1]#
        h0: gives the GW strain amplitude h0(t) for isolated neutron stars #[1]#
        H0: gives the GW strain amplitude after eta introduction H0(t) #[2, 3, 4]#
        H_plus: gives the GW plus polarization strain H_plus(t) #[2, 3, 4]#
        H_cross: gives the GW cross polarization strain H_cross(t) #[2, 3, 4]#
        compute_lst: compute the Local Sidereal Time of the interferometer in [rad] #[6]# 
        A_plus: gives the interferometer response A_plus(t) to GW cross polarization strain because of Earth rotation #[2, 3, 4]#
        A_cross: gives the interferometer response A_cross(t) to GW cross polarization strain because of Earth rotation #[2, 3, 4]#
        h_t: gives the GW signal total strain h(t) #[2, 3, 4]#
        to_pdDataframe: gives a pandas DataFrame with the simulated data
        
    Ref:
        [1] N. Sarin, et al. "X-ray guided gravitational-wave search for binary neutron
            star merger remnants." (2018)
        [2] P. Astone, et al. "A method for detection of known sources of continuous gravitational
            wave signals in non-stationary data." (2010)
        [3] P. Astone, et al. "Method for all-sky searches of continuous gravitational wave signals
            using the frequency-Hough transform." (2014)
        [4] O. J. Piccinni, et al. "A new data analysis framework for the search of continuous
            gravitational wave signals." (2018)
        [5] S. Bonazzola and E. Gourgoulhon. "Gravitational waves from pulsars: Emission by the magnetic
            field induced distortion." (1996).
        [6] The Astropy project
    """
    
    def __init__(self, parameters):
        
        if isinstance(parameters['tcoe'], str):
            self.tcoes = 0                                     # coalescing time/initial time [s] is zero since the 
            self.signal_date = parameters['tcoe']              # date is already specified
        else:
            self.tcoes = parameters['tcoe']*86400              # coalescing time/initial time [s]
            self.signal_date = self.UTCdate()                  # date (iso-UTC) of the GW signal generation (coalescing time)
        
        self.days = parameters['days']                         # duration of the GW signal [days]
        self.dx = parameters['dx_signal']                      # sampling time in [s] of the signal
        
        self.fgw0 = parameters['fgw0']                         # initial freq of the GW signal [Hz]
        self.n = parameters['n']                               # breaking index
        self.Phi0 = parameters['Phi0']                         # initial phase const
        
        self.right_asc = parameters['right_asc']               # right ascension of the source [rad]
        self.dec = parameters['dec']                           # declination of the source [rad]
        self.eta = parameters['eta']                           # polarization degree [-1, 1]
        self.psi = parameters['psi']                           # angle between source major axis wrt source celestial parallel (counterclockwise) [rad]
        
        self.lat = parameters['interf_lat']                    # interferometer's latitude (see parameters function) [rad]
        self.long = parameters['interf_long']                  # interferometer's longitude (see parameters function) [rad]
        self.azmt = parameters['interf_azmt']                  # interferometer's X-arm azimut (see parameters function) [rad]
        self.heigth = parameters['interf_heigth']              # interferometer's elevation (see parameters function) [m]
        
        self.Einstein_delay = parameters['Einstein_delay']     # Einstein delay application
        self.Doppler_effect = parameters['Doppler_effect']     # Doppler effect application
        
        self.NS_parameters = parameters['NS_parameters']       # fiducial values of eps, I and d
        self.h0factor = parameters['h0factor']                 # multiplying factor for h0
        
        self.t = self.time_vec() - self.tcoes                  # time array [s]
        self.iota = self.iota(u = 'ddeg')                      # inclination angle of the NS rot axis wrt interferometer [decimal deg]
        self.mjdsignal_date = self.MJDdate()                   # date (MJD) of the GW signal generation (coalescing time) 
        
        if parameters['tau'] is None:                          # frequency characteristic time (spin-down timescale)
            self.tau = self.compute_tau(parameters['k'])
        else:
            self.tau = parameters['tau']*86400 # in [s]
            
        if self.Doppler_effect:
            self.lst = self.compute_lst()                      # Local Sidereal Time of the interferometer
        else:
            pass
                
    
    ######## methods
    
    def time_vec(self):
        """
        Time vector (t [s], dx is the sampling time)
        """
                
        return np.arange(self.tcoes, self.tcoes + self.days*86400, self.dx)
    
    
    def UTCdate(self):
        """
        It computes the iso-UTC format date of the GW signal starting time if input tcoe is an int/float.
        """
        
        # simulation UTC date
        simdate_utc = Time.now()

        # GPS time
        signaldate_gps = Time(simdate_utc.gps + self.tcoes, format='gps')

        # date (UTC) of the signal detection
        date_utc = signaldate_gps.utc.iso
    
        return date_utc
    
    
    def MJDdate(self):
        """
        It computes the MJD format date of the GW signal starting time.
        """
        
        # date (MJD) of the signal detection
        date_utc = Time(self.signal_date, format='iso', scale='utc')
        date_mjd = date_utc.mjd
        
        return date_mjd
    
    
    def iota(self, u = 'rad'):
        """
        Inclination angle (iota [rad or ddeg]) of the system wrt interferometer, linked to the GW polarization degree (eta).
        --------------------------------
        Par:
            u (str): 'rad' for [rad] or 'ddeg' for [decimal deg] (default = 'rad')
        """
        
        if self.eta == 0:
            i = np.pi/2.
        elif self.eta > 0:
            i = np.arccos(-1./self.eta + np.sqrt(1./self.eta**2 - 1))
        else:
            i = np.arccos(1./np.abs(self.eta) - np.sqrt(1./self.eta**2 - 1))
        
        if u == 'rad':
            return i
        elif u == 'ddeg':
            return i*180/np.pi
        
    
    def compute_tau(self, k):
        """
        Frequency characteristic time in [s].
        --------------------------------
        Par:
            k (int, float): spin-down constant (see Ref. [1])
        """
        
        if k is None:
            raise ValueError('spin-down constant is not specified')
        else:
            pass
        
        return ((np.pi*self.fgw0)**(1 - self.n))/(k*(self.n - 1))
    
    
    def frequency(self, plot = False):
        """
        GW signal frequency (f(t) [Hz]).
        --------------------------------
        Par:
            plot (bool): it makes the plot of the frequency f(t) (default = False)
        """
        
        # time and frequency
        f = self.fgw0*(1. + self.t/self.tau)**(1./(1 - self.n))
        
        # plot
        if plot:
            fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)    
            ax = fig.add_subplot(111)
            ax.plot(self.t/86400, f, c = 'OrangeRed')
            plt.title('Long transient GW signal frequency trend $f(t)$')
            plt.ylabel('frequency [Hz]')
            plt.xlabel('time [days]' + ' - start: ' + self.signal_date + 'UTC')
            ax.grid(True)
            ax.label_outer()            
            ax.tick_params(which='both', direction='in',width=2)
            ax.tick_params(which='major', direction='in',length=7)
            ax.tick_params(which='minor', direction='in',length=4)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            plt.show()
        else:
            pass
        
        return f
    
    
    def EinsteinDelay():
        """
        Compute the Einstein delay and its influence on Phi(t).
        """
        
        return 1
    
    
    def Phi(self, plot = False):   # !!! inserire Einstein delay (here or in a method inside the class)
        """
        Phase of the GW signal (Phi(t), from frequency integration).
        --------------------------------
        Par:
            plot (bool): it makes the plot of the phase(t) (default = False)
        """
        
        # time and phase
        m = (1. - self.n)/(2 - self.n)
        p = self.Phi0 + 2*np.pi*self.tau*self.fgw0*m*((1. + self.t/self.tau)**(1./m) - 1)  # Phi in [rad]
        # p_rad = p*np.pi/180
        
        # plot
        if plot:
            fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)    
            ax = fig.add_subplot(111)
            ax.plot(self.t/86400, p, c = 'OrangeRed')
            plt.title('Long transient GW signal phase trend $\Phi(t)$')
            plt.ylabel('phase [rad]')
            plt.xlabel('time [days]' + ' - start: ' + self.signal_date + 'UTC')
            ax.grid(True)
            ax.label_outer()            
            ax.tick_params(which='both', direction='in',width=2)
            ax.tick_params(which='major', direction='in',length=7)
            ax.tick_params(which='minor', direction='in',length=4)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            plt.show()
        else:
            pass
        
        return p  # !!! check dim of Phi
    
    
    def h0(self):
        """
        GW strain amplitude (h0(t)) which describe the system.
        """
        
        # const scaled with fiducial value of I [kg m**2], eps, d [kpc] and multiplied by h0factor
        const = self.h0factor*4.21e-30*(self.NS_parameters[0]/1e38)*(self.NS_parameters[1]/1e-6)*(1/self.NS_parameters[2])
        
        freq = self.frequency()
        
        return const*(freq**2)

    
    def H0(self):
        """
        GW strain amplitude after eta introduction.
        """
    
        if self.eta == 0:       # no need to compute iota
            c = 0
        elif self.eta > 0:
            c = -1./self.eta + np.sqrt(1./self.eta**2 - 1)
        else:
            c = 1./np.abs(self.eta) - np.sqrt(1./self.eta**2 - 1)
        
        amp0 = self.h0()
        
        # i = self.iota()
        # return amp0*np.sqrt((1 + 6*(np.cos(i))**2 + np.cos(i)**4)/4.)
        
        return amp0*np.sqrt((1 + 6*c**2 + c**4)/4.)
    
    
    def H_plus(self):
        """
        GW plus polarization strain (H_plus(t)).
        """
        
        return (np.cos(2*self.psi) - 1j*self.eta*np.sin(2*self.psi))/np.sqrt(1 + self.eta**2)
        
    
    def H_cross(self):
        """
        GW cross polarization strain (H_cross(t)).
        """
        
        return (np.sin(2*self.psi) + 1j*self.eta*np.cos(2*self.psi))/np.sqrt(1 + self.eta**2)
    
    
    def compute_lst(self):
        """
        Compute the sidereal time.
        """
        
        # define the interferometer location
        location = EarthLocation.from_geodetic(self.long*u.radian, self.lat*u.radian, self.heigth*u.m)

        # simulation UTC date
        actual_time_utc = Time(self.signal_date, format='iso', scale='utc')

        # GPS time array
        actual_time_gps = actual_time_utc.gps
        time_gps = self.time_vec() + actual_time_gps

        # Local Sidereal Time (LST) of the interferometer
        time_utc = Time(time_gps, format = 'gps', scale = 'utc', location = location)
        lst = time_utc.sidereal_time('apparent').radian
    
        return lst
    
    
    def A_plus(self, plot = False):    
        """
        Detector response to GW plus polarization strain (A_plus(t)).
        --------------------------------
        Par:
            plot (bool): it makes the plot of  A_plus(t) (default = False)
        """
        
        # Doppler effect application (GW response to Earth rotation, Omega*t = lst - right_asc + long)
        if self.Doppler_effect:
            Omega_t = self.lst - self.right_asc + self.long
        else:
            Omega_t = 0
        
        # coeff
        a0 = -(3/16)*(1 + np.cos(2*self.dec))*(1 + np.cos(2*self.lat))*np.cos(2*self.azmt)
        a1c = -np.sin(2*self.dec)*np.sin(2*self.lat)*np.cos(2*self.azmt)/4.
        a1s = -np.sin(2*self.dec)*np.cos(2*self.lat)*np.sin(2*self.azmt)/2.
        a2c = -(3 - np.cos(2*self.dec))*(3 - np.cos(2*self.lat))*np.cos(2*self.azmt)/16.
        a2s = -(3 - np.cos(2*self.dec))*np.sin(self.lat)*np.sin(2*self.azmt)/4.
        
        # A_plus(t)
        A = a0 + a1c*np.cos(Omega_t) + a1s*np.sin(Omega_t) + a2c*np.cos(2*Omega_t) + a2s*np.sin(2*Omega_t)
        
        # plot
        if plot:
            fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)    
            ax = fig.add_subplot(111)
            ax.plot(Omega_t, A, c = 'OrangeRed')
            plt.title('Interferometer response to GW plus polarization $A_{+}(t)$')
            plt.ylabel('$A_{+}(t)$')
            plt.xlabel('$\Theta$ - right\_asc + interf\_long')
            ax.grid(True)
            ax.label_outer()            
            ax.tick_params(which='both', direction='in',width=2)
            ax.tick_params(which='major', direction='in',length=7)
            ax.tick_params(which='minor', direction='in',length=4)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            plt.show()
        else:
            pass
        
        return A
    
    
    def A_cross(self, plot = False):
        """
        Detector response to GW cross polarization strain (A_cross(t)).
        --------------------------------
        Par:
            plot (bool): it makes the plot of A_cross(t) (default = False)
        """
        
        # Doppler effect application (GW response to Earth rotation, Omega*t = lst - right_asc + long)
        if self.Doppler_effect:
            Omega_t = self.lst - self.right_asc + self.long
        else:
            Omega_t = 0
        
        # coeff
        b1c = -np.cos(self.dec)*np.cos(self.lat)*np.sin(2*self.azmt)
        b1s = np.cos(self.dec)*np.sin(2*self.lat)*np.cos(2*self.azmt)/2.
        b2c = -np.sin(self.dec)*np.sin(self.lat)*np.sin(2*self.azmt)
        b2s = (3 - np.cos(2*self.lat))*np.sin(self.dec)*np.cos(2*self.azmt)/4.
        
        # A_cross(t)
        B = b1c*np.cos(Omega_t) + b1s*np.sin(Omega_t) + b2c*np.cos(2*Omega_t) + b2s*np.sin(2*Omega_t)
        
        # plot
        if plot:
            fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)    
            ax = fig.add_subplot(111)
            ax.plot(Omega_t, B, c = 'OrangeRed')
            plt.title('Interferometer response to GW cross polarization $A_{x}(t)$')
            plt.ylabel('$A_{x}(t)$')
            plt.xlabel('$\Theta$ - right\_asc + interf\_long')
            ax.grid(True)
            ax.label_outer()            
            ax.tick_params(which='both', direction='in',width=2)
            ax.tick_params(which='major', direction='in',length=7)
            ax.tick_params(which='minor', direction='in',length=4)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            plt.show()
        else:
            pass
        
        return B
    
    
    ## long transient GW signal total strain
    def h_t(self, plot = False, values = 'real'):
        """
        GW signal total strain (h(t)).
        --------------------------------
        Par:
            plot (bool): it makes the plot of the total long transient GW strain h(t) (default = False)
            values (str): it specifics the values of h(t) to plot ('real', 'imag' or 'abs', default = 'real')
        """
    
        # compute h(t)
        AMP0 = self.H0()
        AMP_plus = self.H_plus()
        AMP_cross = self.H_cross()
        Pattfunc_plus = self.A_plus()
        Pattfunc_cross = self.A_cross()
        Phase = self.Phi()
        
        h = AMP0*(AMP_plus*Pattfunc_plus + AMP_cross*Pattfunc_cross)*np.exp(1j*Phase)
        
        # plot        
        if plot:
            title = 'Long transient GW signal total strain $h(t)$'
            
            if values == 'real':
                h_values = h.real
                title += ' (real values)'
            elif values == 'imag':
                h_values = h.imag
                title += ' (imag values)'
            elif values == 'abs':
                h_values = np.abs(h)
                title += ' (abs values)'
            else:
                raise ValueError("Must specify values: values = 'real', 'imag' or 'abs' (default = 'real')")
            
            fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)    
            ax = fig.add_subplot(111)
            ax.plot(self.t/86400, h_values, c = 'OrangeRed')
            plt.title(title)
            plt.ylabel('strain')
            plt.xlabel('time [days]' + ' - start: ' + self.signal_date + 'UTC')
            ax.grid(True)
            ax.label_outer()            
            ax.tick_params(which='both', direction='in',width=2)
            ax.tick_params(which='major', direction='in',length=7)
            ax.tick_params(which='minor', direction='in',length=4)
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            plt.show()
        else:
            pass
    
        return h
    
    
    def to_pdDataframe(self, objects = 'data', save_to_csv = False, directory = None):   # !!!
        """
        It generates a pandas DataFrame with the simulated data.
        --------------------------------
        Par:
            objects (str): if 'data' the out DataFrame contains the simulated data, if 'parameters'
                           the simulation parameters will be moved in the out DataFrame (default = 'data')
            save_to_csv (bool): if True, it saves the DataFrame with the data (default = False)
            directory (str): path for the directory where the DataFrame will be saved (default = None)
        """
        
        if objects is None:
            raise ValueError("Specify objects: 'data' or 'parameters'")
        
        elif objects == 'data':  # makes the data dataframe
            df = pd.DataFrame([self.frequency(), self.Phi(), self.A_plus(),
                               self.A_cross(), self.h_t().real, self.h_t().imag]).transpose()
            
            df.columns = ['frequency', 'phase', 'A_plus', 'A_cross', 'h_real', 'h_imag']
        
        elif objects == 'parameters':  # makes the parameters dataframe
            df = pd.DataFrame.from_dict(vars(self))
        
        else:
            raise ValueError("Specify objects: 'data' or 'parameters'")
            
        # save the dataframe
        if save_to_csv:
            if directory is None:
                raise ValueError('Specify the directory for the DataFrame')
            elif objects == 'data':
                df.to_csv(directory + 'LT_GWSimulatedSignal_' + self.signal_date + '.csv')
            elif objects == 'parameters':
                df.to_csv(directory + 'LT_GWSimulatedParameters_' + self.signal_date + '.csv')
        
        return df





#############################################################################################################################





#### class: simulated long transient GW injection into interferometer noise

## class variables from bsd_gout (to call them: GW_signal)

# gout = parameters['bsd_gout']                # dict containing the interferometer's data and info
# dx = gout['dx']                              # sampling time [s] (ricordarsi di mutare quello in parameters)
# freq_band
