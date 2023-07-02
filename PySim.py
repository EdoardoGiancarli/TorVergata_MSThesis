# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:56:48 2023

@author: Edoardo Giancarli
"""

#### ModuleMSThesis - Signal simulation/injection version 2 ###############################################

####   libraries   #####

import numpy as np                              # operations
import pandas as pd                             # dataframe

import PySpectr as psr                          # interferometer data

import matplotlib.pyplot as plt                 # plotting
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=26, usetex=True)

# coordinates, units and sidereal time computation
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic, EarthLocation #, ITRS
import astropy.units as u
from astropy.time import Time
import astropy.constants as const               # physical constants

#from skyfield.api import load                   # ephemeris

# import Spectrogram_from_SciPy as sfs            # spectrogram
from scipy import signal                          # spectrogram
from scipy.fft import fftshift

####    content    #####

# parameters (function): it defines the parameters to be utilized for simulating the long transient GW signal
#
# GW_signal (class): long transient GW signal simulation
#
# GW_injection (class): long transient GW signal injection in interferometer noise

####    codes    #####


def parameters(days, dt, fgw0, tcoe = 0, n = 5, k = 1.2167e-16, tau = None, Phi0 = 0, right_asc = '05 34 30.9',
               dec = '+22 00 53', eta = -1, psi = 0, Einstein_delay = False, Doppler_effect = False,
               interferometer = 'LIGO-L', NS_parameters = [1e38, 1e-6, 1], h0factor = 1e20,
               signal_inj = False, bsd_gout = None, key='gout_58633_58638_295_300', mat_v73=True):
    
    """
    It takes the arguments and gruop them in a dictionary, defining the parameters to be utilized for simulating the
    long transient GW signal.
    -----------------------------------------------------------------------------------------------------------------
    Parameters:
        days (int, float): duration of the GW signal in [days] --
        dt (float): sampling time in [s] of the signal --
        fgw0 (int, float): initial freq in [Hz] of the GW signal --
        tcoe (int, float, str): coalescing time/initial time in [days] or [date in UTC scale, iso format] of the simulated
                                long transient GW signal; if tcoe in [days] it represents the starting time wtr the actual
                                time date at which the signal is simulated (astropy.time.Time.now() date); tcoe can be <, =
                                or > 0 (if tcoe < 0, the signal is generated tcoe days in the past, default = 0); also if tcoe
                                is >= 57277 (2015-09-12, the begin of O1 run) will be considered as a date in MJD format and 
                                converted to iso-UTC date --
        n (int, float): breaking index (default = 5) --
        k (int, float): spin-down proportionality const (if given, default = 1.2167e-16, which gives tau = 1 (\pm 0.005)
                        in [s] for an fgw0 = 125 [Hz] and n = 5) --
        tau (int, float): frequency characteristic time (spin-down timescale) in [days] (if given, default = None) --
        Phi0 (int, float): initial phase const in [decimal degrees] (default = 0 since the interf work in dark fringe) --
        right_asc (str, float): right ascension of the source in [hour angle] (str) or [decimal degree] (float)
                                (default = '05 34 30.9' from Crab pulsar, coord by SIMBAD archive) --
        dec (str, float): declination of the source in [deg] (str) or [decimal degree] (float) (default = '+22 00 53'
                          from Crab pulsar, coord by SIMBAD archive) --
        eta (int, float): polarization degree, values between [-1, 1] (default = -1, corresponding to iota = 0) --
        psi (int, float): angle between source major axis wrt source celestial parallel (counterclockwise)
                          in [decimal degrees] (default = 0) --
        Einstein_delay (bool): Einstein delay application (default = False) --
        Doppler_effect (bool): Doppler effect application (Doppler due to Earth rotation) (default = False) --
        interferometer (str): choose the interferometer (LIGO-L, LIGO-H, Virgo or KAGRA, defult = LIGO-L) --
        NS_parameters (list): this list contains the fiducial values of the ellipticity (default = 1e-6), moment of
                              inertia (default = 1e38 [kg m**2]) and distance (default = 1 [kpc]) of a 1.4M_sun
                              neutron star (these fiducial values set the constant of the h0(t) strain amplitude) --
        h0factor (float): multiplying factor for h0 to reduce the strain amplitude h(t) values (for computational costs,
                          default = 1e23) --
        signal_inj (bool): if you want to inject the simulated signal into some interf noise, the data about the interf are
                           also loaded (check the interferometer input, default = False) --
        bsd_gout (str, dict): bsd_gout containing the interferometer's data (if the path to bsd_gout is inserted, then
                              the MATLAB data file will be converted in dict, default = None) --
        key : (str) keyword with info from L, H or V interferometer (default = 'gout_58633_58638_295_300') --
        mat_v73 (bool): if the matlab datafile version is -v7.3 insert the 'True' value (default = 'True') --    
    
    Return:
        par (dict): dict with the parameters to simulate the signal
    
    Info:        
        additional argument in par: source ecliptical coordinates [long, lat] in [rad] (GeocentricTrueEcliptic frame),
                                    interferometer's longitude, latitude, azimut in [rad] and elevation in [m],
                                    bsd_gout (dict) (if signal_inj = True)
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
    
    ## check on tcoe (if tcoe >= 57277 its value will be considered in MJD format)
    elif tcoe >= 57277:
        tcoe = Time(tcoe, format='mjd', scale='utc').iso
        print("Friendly Warning: since tcoe is >= 57277 it has been considered as a date in MJD format and converted to iso-UTC date")
            
    ## check on eta
    if np.abs(eta) > 1:
        raise ValueError('Parameter eta must be a value in [-1, 1]')
    else:
        pass
    
    ## !!! interferometer's latitude[rad], longitude[rad], elevation[m] #[1]# and azimut[rad] #[2]#
    # url: https://lscsoft.docs.ligo.org/lalsuite/lal/group___detector_constants.html
    # url: https://web.infn.it/VirgoRoma/index.php/en/research-eng/data-analysis-eng/snag-eng-page
    # Ref.: [1] LIGO Scientic Collaboration, "LIGO Algorithm Library - LALSuite, free software" (2021)
    #       [2] Snag MATLAB toolbox 
    # if not in radiant: to multiply by *np.pi/180
    if interferometer == 'LIGO-L':               
        lat, long, azmt, interf_heigth = 0.53342313506, -1.58430937078, 72.2836*np.pi/180, -6.574
    elif interferometer == 'LIGO-H':
        lat, long, azmt, interf_heigth = 0.81079526383, -2.08405676917, 144.0006*np.pi/180, 142.554
    elif interferometer == 'Virgo':
        lat, long, azmt, interf_heigth = 0.76151183984, 0.18333805213, 199.4326*np.pi/180, 51.884
    elif interferometer == 'KAGRA':
        lat, long, azmt, interf_heigth = 0.6355068497, 2.396441015, 1.054113, 414.181 # for KAGRA the azimut is from [1]
    else:
        raise ValueError('Specify interferometer')
    
    ## create SkyCoord object with (astropy)
    if isinstance(right_asc, str) and isinstance(dec, str):
        coord = SkyCoord(right_asc + ' ' + dec, frame = 'icrs', unit=(u.hourangle, u.deg))
    elif isinstance(right_asc, float) and isinstance(dec, float):
        coord = SkyCoord(right_asc, dec, frame='icrs', unit='deg')
    else:
        raise ValueError("Right ascension and declination must be in [hourangle] and [deg] (str) or [decimal degree]\n",
                         "and [decimal degree] (float)")

    # coordinates wrt ecliptic
    ecl_coord = coord.transform_to(GeocentricTrueEcliptic())
        
    ## dictionary with source and signal parameters
    par = {'tcoe': tcoe,
           'days': days,
           'dt': dt,
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
            bsd_gout = psr.extract_bsdgoutinfo(bsd_gout, key=key, mat_v73=mat_v73)
            par['bsd_gout'] = bsd_gout
            par['dt'] = bsd_gout['dt']       # sampling time of the interferometer's data
            print("Friendly warning: since the signal is meant to be injected in the detector's noise,\n",
                  "the sampling time dt is replaced with the sampling time of the interferometer's data")
    else:
        pass
    
    return par


#############################################################################################################################


#### class: long transient GW simulation

class GW_signal:
    """
    This class simulate a long transient GW signal from an isolated neutron star
    using the supplied parameters and options.
    -----------------------------------------------------------------------------
    Parameters:
        parameters (dict): dictionary from parameters function
        
    Attributes:
        tcoes (int, float): coalescing time/initial time in [s] of the signal (from tcoe in input dict)
        signal_date: date (UTC) of the GW signal generation
        days (int, float): duration of the GW signal in [days]
        dt (float): sampling time in [s] of the signal
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
                 module, UTCdate return the actual time in iso-UTC format each time is called) #[6]#
        MJDdate: computes the MJD format date of the GW signal starting time. #[6]#
        iota: gives the inclination angle (iota) in [rad] or [decimal degrees] of the system wrt interferometer, linked to
              the GW polarization degree (eta) #[2, 3, 4]#
        compute_tau: gives the frequency characteristic time in [s] #[1]#
        plot: makes a plot
        frequency: gives the GW signal frequency f(t) in [Hz] #[1]#
        EinsteinDelay: compute the Einstein delay and its influence on Phi(t) #[6]# !!!
        Phi: gives the phase shift Phi(t) in [rad] of the GW signal (from f(t) integration) #[1]#
        h0: gives the GW strain amplitude h0(t) for isolated neutron stars #[1]#
        H0: gives the GW strain amplitude after eta introduction H0(t) #[2, 3, 4]#
        H_plus: gives the GW plus polarization strain H_plus(t) #[2, 3, 4]#
        H_cross: gives the GW cross polarization strain H_cross(t) #[2, 3, 4]#
        compute_lst: compute the Local Sidereal Time of the interferometer in [rad] #[6]# 
        A_plus: gives the interferometer response A_plus(t) to GW cross polarization strain because of Earth rotation #[2, 3, 4]#
        A_cross: gives the interferometer response A_cross(t) to GW cross polarization strain because of Earth rotation #[2, 3, 4]#
        h_t: gives the GW signal total strain h(t) #[2, 3, 4]#
        spectr: generate the spectrogram of h(t) (to be careful with the signal frequency band - sampling time relation) #[7]#
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
        [6] The Astropy project (url: https://www.astropy.org/)
        [7] SciPy (scipy.signal module, url: https://docs.scipy.org/doc/scipy/reference/signal.html)
        [8] Skyfield (url: https://rhodesmill.org/skyfield/) #!!!
    """
    
    def __init__(self, parameters):
        
        if isinstance(parameters['tcoe'], str):
            self.tcoes = 0                                     # coalescing time/initial time [s] is zero since the 
            self.signal_date = parameters['tcoe']              # date is already specified
        else:
            self.tcoes = parameters['tcoe']*86400              # coalescing time/initial time [s]
            self.signal_date = self.UTCdate()                  # date (iso-UTC) of the GW signal generation (coalescing time)
        
        self.days = parameters['days']                         # duration of the GW signal [days]
        self.dt = parameters['dt']                             # sampling time in [s] of the signal
        
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
        
        self.NS_parameters = parameters['NS_parameters']       # fiducial values of I, eps and d
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
        Time vector (t [s], dt is the sampling time)
        """
                
        return np.arange(self.tcoes, self.tcoes + self.days*86400, self.dt)
    
    
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
        
        # date (iso-UTC) of the signal detection
        date_utc = Time(self.signal_date, format='iso', scale='utc')
        
        return date_utc.mjd
    
    
    def iota(self, u = 'rad'):
        """
        Inclination angle (iota [rad or decimal deg]) of the system wrt interferometer,
        linked to the GW polarization degree (eta, see Ref [2, 3, 4]).
        --------------------------------
        Par:
            u (str): 'rad' for [rad] or 'ddeg' for [decimal deg] (default = 'rad')
        """
        
        # iota from eta
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
            raise ValueError('Spin-down constant is not specified.')
        else:
            pass
        
        return ((np.pi*self.fgw0)**(1 - self.n))/(k*(self.n - 1))
    
    
    def plot(self, x, y, title, xlabel, ylabel):
        """
        Plot module.
        --------------------------------
        Par:
            x (ndarray): x values
            y (ndarray): y values
            title (str): plot title
            xlabel (str): x label
            ylabel (str): y label
        """
        
        fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)    
        ax = fig.add_subplot(111)
        ax.plot(x, y, c = 'OrangeRed')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        ax.grid(True)
        ax.label_outer()            
        ax.tick_params(which='both', direction='in',width=2)
        ax.tick_params(which='major', direction='in',length=7)
        ax.tick_params(which='minor', direction='in',length=4)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        plt.show()
    
    
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
            self.plot(self.t/86400, f,
                      'Long transient GW signal frequency trend $f(t)$',
                      'time [days]' + ' - start: ' + self.signal_date + 'UTC',
                      'frequency [Hz]')
            
        else:
            pass
        
        return f
    
    
    def EinsteinDelay(self):
        """
        Compute the Einstein delay and its influence on Phi(t).
        """
        
        # # define the interferometer location and obs time
        # obs_location = EarthLocation.from_geodetic(self.long*u.radian, self.lat*u.radian, self.heigth*u.m)
        # obs_time = Time(self.signal_date, format='iso', scale='utc')
        
        # # load the ephemeris data (valid from 1900 to 2050)
        # eph = load('de421.bsp')
        
        # # define time
        # obs_time_gps = obs_time.gps                                   # simulation start time gps    
        # time_gps = self.time_vec() + obs_time_gps                     # simulation time array gps
        # gps_time = Time(time_gps, format = 'gps', scale = 'utc')      # simulation time array gps (Time object)
        
        # # define Earth and Sun position
        # earth = eph['earth'].at(time_gps).position.au
        # sun = eph['sun'].at(time_gps).position.au
        
        # # source position and distance
        # r = self.NS_parameters[2].to(u.m)  # distance in meters
        # source_location = SkyCoord(ra = self.right_asc, dec = self.dec,
        #                            distance = r, frame='icrs')
        # source_position = source_location.transform_to(ITRS(obstime = obs_time, location = obs_location)).cartesian.xyz.to(u.au)
        # pos_source = SkyCoord(source_position[0], source_position[1], source_position[2], unit='au')
        
        # # Earth - Sun distance
        # pos_earth = SkyCoord(earth[0], earth[1], earth[2], unit='au')
        # pos_sun = SkyCoord(sun[0], sun[1], sun[2], unit='au')
        # d = pos_earth.separation_3d(pos_sun).to(u.m).value
        
        # # Sun - source position
        # h = pos_sun.separation_3d(pos_source).to(u.m).value
        
        # # Einstein delay
        # einstein_delay = -(2*const.GM_sun.value/(const.c.value)**3)*np.log(1 - (r**2 + d**2 - h**2)/2*r*d)
        
        ##########
        
        # julian date
        obs_time = Time(self.signal_date, format='iso', scale='utc')
        obs_time_gps = obs_time.gps                                   # simulation start time gps    
        time_gps = self.time_vec() + obs_time_gps                     # simulation time array gps
        gps_time = Time(time_gps, format = 'gps', scale = 'utc')      # simulation time array gps (Time object)
        
        JD2000 = gps_time.jd - 2451271.                               # time array in JD wrt Jan 1 2000 (51271 in MJD)
        
        # Earth aphelion and perielion
        aph = 1.521e11
        per = 1.47095e11
        
        # Schwarzschild radius for the Sun
        Rs_Sun = 2*const.GM_sun.value/(const.c.value)**2
        
        # compute orbital position angle in radians and Earth-Sun distance
        mean_anomaly = 357.53 + 0.98560028*JD2000                               # average angular position of Earth at JD
        g = np.mod(mean_anomaly, 360)*np.pi/180                                  
        d = ((aph - per)/2)*(np.sin(g) + 0.0084439*np.sin(2*g)) + (aph + per)/2
        
        # Einstein effect
        f_einst = self.fgw0*(1 - Rs_Sun*(1./d - 1./const.au.value))
        
        return 2*np.pi*np.cumsum(f_einst)*self.dt
    
    
    def Phi(self, plot = False):
        """
        Phase of the GW signal (Phi(t) [rad], from frequency integration).
        --------------------------------
        Par:
            plot (bool): it makes the plot of the phase(t) (default = False)
        """
        
        # time and phase
        m = (1. - self.n)/(2 - self.n)
        p = self.Phi0 + 2*np.pi*self.tau*self.fgw0*m*((1. + self.t/self.tau)**(1./m) - 1)  # Phi in [rad]
        title = 'Long transient GW signal phase trend $\Phi(t)$'
        # p_rad = p*np.pi/180
        
        # Einstein delay
        if self.Einstein_delay:
            p -= self.EinsteinDelay()
            title += ' with Einstein delay'
        else:
            pass
        
        # plot
        if plot:
            self.plot(self.t/86400, p,
                      title,
                      'time [days]' + ' - start: ' + self.signal_date + 'UTC',
                      'phase [rad]')
            
        else:
            pass
        
        return p
    
    
    def h0(self):
        """
        GW strain amplitude (h0(t)) which describe the system.
        """
        
        # const scaled with fiducial value of I [kg m**2], eps, d [kpc] and multiplied by h0factor
        a = self.h0factor*4.21e-30*(self.NS_parameters[0]/1e38)*(self.NS_parameters[1]/1e-6)*(1/self.NS_parameters[2])
        # frequency values
        freq = self.frequency()
        
        return a*(freq**2)

    
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
    
    
    def compute_lst(self): # mean and apparent differ 1e-5 from each other on average
        """
        Compute the Local Sidereal Time.
        """
        
        # define the interferometer location
        location = EarthLocation.from_geodetic(self.long*u.radian, self.lat*u.radian, self.heigth*u.m)

        # simulation UTC date
        actual_time_utc = Time(self.signal_date, format='iso', scale='utc')

        # GPS time array
        actual_time_gps = actual_time_utc.gps
        time_gps = self.time_vec() + actual_time_gps

        # Local Sidereal Time (LST) of the interferometer
        gps_time = Time(time_gps, format = 'gps', scale = 'utc', location = location)
        lst = gps_time.sidereal_time('mean').radian
        # lst = gps_time.sidereal_time('apparent').radian
    
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
            self.plot(Omega_t, A,
                      'Interferometer response to GW plus polarization $A_{+}(t)$',
                      '$\Theta$ - right\_asc + interf\_long',
                      '$A_{+}(t)$')
            
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
            self.plot(Omega_t, B,
                      'Interferometer response to GW plus polarization $A_{x}(t)$',
                      '$\Theta$ - right\_asc + interf\_long',
                      '$A_{x}(t)$')
        
        else:
            pass
        
        return B
    
    
    def h_t(self, plot = False, values = 'real'):
        """
        GW signal total strain (h(t)).
        --------------------------------
        Par:
            plot (bool): it makes the plot of the total long transient GW strain h(t) (default = False)
            values (str): it specifies the values of h(t) to plot ('real', 'imag' or 'abs', default = 'real')
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
            
            self.plot(self.t/86400, h_values,
                      title,
                      'time [days]' + ' - start: ' + self.signal_date + 'UTC',
                      'strain')
        
        else:
            pass
    
        return h
    
    
    def spectr(self, lfft = 1024, image = False, directory = None):
        """
        It generates the spectrogram of the simulated data.
        --------------------------------
        Par:
            lfft (int): fft length for computing the FFT in the spectrogram (default = 1024)
            image (bool): if True the image of the spectrogram is shown (default = 'False')
            directory (str): directory where to save the spectrograms image (default = None)
        
        Return:
            f (ndarray): array of sample frequencies in [Hz]
            t (ndarray): array of segment times in [days]
            Sxx (ndarray): spectrogram of h(t) (by default, the last axis of Sxx corresponds
                           to the segment times)
        """
        
        x = self.h_t()
        
        # make spectrogram
        _, t, spectrogram = signal.spectrogram(x, fs=1/self.dt, window = 'cosine', nperseg = lfft - 1,
                                               noverlap=lfft//8, nfft=lfft, return_onesided=False,
                                               scaling='density', mode='psd')
        
        # shift frequency zero-component to correctly visualize the spectrogram
        spectr_shft = fftshift(spectrogram/np.max(spectrogram), axes=0)
        
        # translate the spectrogram 2D-array to remove the negative frequencies (as computed by SciPy) since h(t) is complex
        spectr_trs = np.vstack((spectr_shft[int(spectrogram.shape[0]/2):, :], spectr_shft[0:int(spectrogram.shape[0]/2), :]))
        
        freq_trs = np.arange(0, lfft)/(self.dt*lfft) + self.frequency()[-1] # define freq array
        
        if image: # spectrogram image
        
            if directory is None:
                raise ValueError('Specify the directory to save the spectrograms image')
            else:
                pass
                            
            plt.figure(num = None, figsize=(12, 12), tight_layout = True)        
            plt.pcolormesh(t, freq_trs, np.log10(spectr_trs + 1e-15), cmap='viridis', shading='gouraud')
            plt.axis('off')
            plt.title('')
            plt.savefig(directory + 'spectr_h(t)' + '.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        
        else: # spectrogram plot
            
            plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
            col_bar = plt.pcolormesh(t/86400, freq_trs, np.log10(spectr_trs + 1e-15),
                                     cmap='viridis', shading='gouraud')
            plt.colorbar(col_bar)
            plt.title('Spectrogram of $h(t)$')
            plt.ylabel('frequency [Hz]')
            plt.xlabel('time [days]' + ' - start: ' + self.signal_date + 'UTC')
            plt.show()
            
        return freq_trs, t/86400, spectr_trs
    
    
    def to_pdDataframe(self, objects = 'data', save_to_csv = False, directory = None):
        """
        It generates a pandas DataFrame with the simulated data.
        --------------------------------
        Par:
            objects (str): if 'data' the out DataFrame contains the simulated data, if 'parameters'
                           the simulation parameters will be moved in the out DataFrame (default = 'data')
            save_to_csv (bool): if True, it saves the DataFrame with the data (default = False)
            directory (str): path for the directory where the DataFrame will be saved (default = None)
        """
        
        # data dataframe
        if objects == 'data':
            df = pd.DataFrame([self.frequency(), self.Phi(), self.A_plus(),
                               self.A_cross(), self.h_t().real, self.h_t().imag]).transpose()
            
            df.columns = ['frequency', 'phase', 'A_plus', 'A_cross', 'h_real', 'h_imag']
        
        # parameters dataframe
        elif objects == 'parameters':
            par = vars(self)
            par['I'] = self.NS_parameters[0]
            par['eps'] = self.NS_parameters[1]
            par['dist'] = self.NS_parameters[2]
            del par['NS_parameters'], par['t'], par['lst']
            
            df = pd.DataFrame(list(par.items()), columns=['Keys', 'Values'])
            df = df.set_index('Keys')
        
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

class GW_injection:
    """
    This class injects a simulated long transient GW signal from an isolated neutron star
    into the selected interferometer's data (representing the noise) using the supplied parameters and options.
    -----------------------------------------------------------------------------
    Parameters:
        parameters (dict): dictionary from parameters function
        amp (float, int): amplitude of the signal wrt noise
        
    Attributes:
        dt (float, int): sampling time in [s] of the data
        gout (dict): dict with the info on the interferometer data
        band (float, int): frequency bandwidth [Hz]
        inifr (float, int): initial band frequency [Hz]
        t0_gout (float, int): starting time of the gout signal [mjd]
        y_gout (complex ndarray): data from the gout bsd
        amp (float, int): amplitude of the signal wrt noise
        Deltatime_data (float, int): time duration of the interferometer output [days]
        
        t0_signal (float, int): starting time of the simulated signal [mjd]                     
        frequency (float ndarray): frequency of the simulated signal [Hz]             
        Phi (float ndarray): phase of the simulated signal [rad]
        h_t (float ndarray): total strain of the simulated signal
        
        t0_iso_signal (str): starting time of the simulated GW signal [date iso-UTC]
        tini_iso_gout (str): starting time of the interferometer data [date iso-UTC]
        tend_iso_gout (str): ending time of the interferometer data [date iso-UTC]
            
    Methods:
        MJD_to_isoUTC: converts the MJD date in input to its iso-UTC format #[1]#
        frequency_band: extracts the signal frequency, phase and strain in the band which the interferometer's data covers
        data_chunk: selects the interferometer's data chunk in which the signal is injected
        injection: injects the signal in the interferometer's data
        spectr: generate the spectrogram of h(t) (to be careful with the signal frequency band - sampling time relation) #[2]#
        to_pdDataframe: gives a pandas DataFrame with the simulated data
    
    Ref:
        [1] The Astropy project (url: https://www.astropy.org/)
        [2] SciPy (scipy.signal module, url: https://docs.scipy.org/doc/scipy/reference/signal.html)
    """
    
    def __init__(self, parameters, amp):
        
        self.dt = parameters['dt']                               # sampling time in [s] of the data
        
        if not 'bsd_gout' in parameters.keys():
            raise ValueError("In order to inject the signal you have to set to 'True'\n",
                             "the 'signal_inj' input in the function parameters")
        else:
            self.gout = parameters['bsd_gout']                   # dict with the info on the interferometer data 
        
        self.band = self.gout['bandw']                           # frequency bandwidth [Hz]
        self.inifr = self.gout['inifr']                          # initial band frequency [Hz]
        self.t0_gout = self.gout['t0_gout']                      # starting time of the gout signal [mjd]
        self.y_gout = self.gout['y_gout']                        # data from the gout bsd
        self.amp = amp                                           # amplitude of the signal wrt noise
        
        self.Deltatime_data = self.gout['n']*self.dt/86400       # time duration of the interferometer output [days] 
        
        gws = GW_signal(parameters)                              # GW_signal class for dignal simulation
        self.t0_signal = gws.mjdsignal_date                      # starting time of the simulated signal [mjd]      
        self.frequency = gws.frequency()                         # frequency of the simulated signal [Hz] 
        self.Phi = gws.Phi()                                     # phase of the simulated signal [rad]
        self.h_t = gws.h_t()                                     # total strain of the simulated signal
        self.t = gws.t                                           # time array of the simulation
        
        self.t0_iso_signal = gws.signal_date                                              # starting time of the simulated GW signal [date iso-UTC]
        self.tini_iso_gout = self.MJD_to_isoUTC(self.t0_gout)                             # starting time of the interferometer data [date iso-UTC]
        self.tend_iso_gout = self.MJD_to_isoUTC(self.t0_gout + int(self.Deltatime_data))  # ending time of the interferometer data [date iso-UTC]
                
    
    ######## methods
    
    
    def MJD_to_isoUTC(self, mjd_date):
        """
        Conversion from MJD to iso-UTC.
        --------------------------------
        Par:
            mjd_date (int, float): date to convert [MJD]
        """
        
        # define mjd date
        t_mjd = Time(mjd_date, format='mjd', scale='utc')
        
        return t_mjd.iso
    
    
    def frequency_band(self, plot = False, observable = 'frequency', values = 'real'):
        """
        Extraction the signal frequency, phase and strain in the band
        which the interferometer's data covers.
        ------------------------------------------------------
        Par:
            plot (bool): it makes the plot of the long transient GW frequency, phase or strain
                         in the frequency band covered by the interferometer data (default = False)
            observable (str): 'frequency', 'phase', 'strain' for the plot (default = 'frequency')
            values (str): it specifies the values of the strain to plot ('real', 'imag' or 'abs', default = 'real')
            
        Return:
            obs_freq (ndarray): frequency f(t) in the interferometer band
            obs_phi (ndarray): phase Phi(t) in the interferometer band
            obs_h (ndarray): strain h(t) in the interferometer band
            obs_t (ndarray): time array chunk of the injection
        """
        
        try:
            # extract the frequency interval of the simulated signal
            obs_freq = self.frequency[(self.frequency > self.inifr) & (self.frequency < self.inifr + self.band)]
            
            # indexes of the frequency interval
            a = np.where(self.frequency == obs_freq[0])[0][0]
            z = np.where(self.frequency == obs_freq[-1])[0][0]
            
            # cut Phi_t and h_t in the right frequency band
            obs_phi = self.Phi[a:z + 1]
            obs_h = self.h_t[a:z + 1]
            obs_t = self.t[a:z + 1]
            
        except ValueError:
            raise ValueError(f"The simulated signal has no frequency values in the band [{self.inifr}, {self.inifr + self.band}][Hz].")
        
        # plot        
        if plot:
            
            if observable == 'frequency':
                title = 'Observed $f(t)$'
                y_values = obs_freq
                
            elif observable == 'phase':
                title = 'Observed $Phi(t)$'
                y_values = obs_phi
                
            elif observable == 'strain':
                title = 'Observed $h(t)$'
                
                if values == 'real':
                    y_values = obs_h.real
                    title += ' (real values)'
                elif values == 'imag':
                    y_values = obs_h.imag
                    title += ' (imag values)'
                elif values == 'abs':
                    y_values = np.abs(obs_h)
                    title += ' (abs values)'
                else:
                    raise ValueError("Must specify values: values = 'real', 'imag' or 'abs' (default = 'real')")
            else:
                raise ValueError("Must specify the observable: 'frequency', 'phase', 'strain' (default = 'frequency')")
            
            fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)    
            ax = fig.add_subplot(111)
            ax.plot(obs_t/86400, y_values, c = 'OrangeRed')
            plt.title(title)
            plt.ylabel(observable)
            plt.xlabel('time [days]' + ' - start: ' + self.t0_iso_signal + 'UTC')
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
        
        return obs_freq, obs_phi, obs_h, obs_t
    
    
    def data_chunk(self):
        """
        Interferometer's data chunk in which the signal is injected.
        """
        
        # choose right start time for the simulated signal
        if not (self.t0_signal > self.t0_gout and self.t0_signal < self.t0_gout + self.Deltatime_data):
            raise ValueError("The start time of the simulated long transient signal is not in the observation time\n",
                             f"interval: t0signal is {self.t0_signal} (MJD) - {self.t0_iso_signal} (iso-UTC),\n",
                             f"Dtgout is [{self.t0_gout}, {self.t0_gout + int(self.Deltatime_data)}] (MJD) -\n",
                             f"[{self.tini_iso_gout}, {self.tend_iso_gout}] (iso-UTC).")
        
        else:
            # index for y_gout
            ind = int((self.t0_signal - self.t0_gout)*86400/self.dt)
            
            # take the length of the obs h_t
            obs_h = self.frequency_band()[2]
            
            # detector's data chunk in which the signal would be observed
            y_chunk = self.y_gout[ind:ind + len(obs_h)]
        
        return y_chunk


    def injection(self, plot = False, values = 'real'):  # si potrebbe fare anche con h0factor, da vedere
        """
        Injection of the signal in the interferometer's data.
        --------------------------------
        Par:
            plot (bool): it makes the plot of the long transient GW strain + noise s(t) = amp*h(t) + n(t) (default = False)
            values (str): it specifies the values of s(t) to plot ('real', 'imag' or 'abs', default = 'real')
        
        Return:
            y (complex ndarray): total strain s(t) = amp*h(t) + n(t)
        """
        
        # take observed strain and interferometer data chunk
        obs_h, obs_t = self.frequency_band()[2], self.frequency_band()[3]
        y_chunk = self.data_chunk()
        
        # injection
        y = self.amp*obs_h + y_chunk
        
        # remove segment of non-data collection
        y_chunkzeros = np.where(y_chunk == 0)[0]
        for z in y_chunkzeros:
            y[z] = 0 + 0j
            
        # check amp
        print(f"Check on the reciprocal amplitudes: max_h (abs) is {self.amp*np.max(np.abs(obs_h))}",
              f"and max_noise (abs) is {np.max(np.abs(y_chunk))}")
        
        # plot        
        if plot:
            title = f'Observed data $s(t) = {self.amp} h(t) + n(t)$'
            
            if values == 'real':
                y_values = y.real
                title += ' (real values)'
            elif values == 'imag':
                y_values = y.imag
                title += ' (imag values)'
            elif values == 'abs':
                y_values = np.abs(y)
                title += ' (abs values)'
            else:
                raise ValueError("Must specify values: values = 'real', 'imag' or 'abs' (default = 'real')")
            
            fig = plt.figure(num = None, figsize = (12, 12), tight_layout = True)    
            ax = fig.add_subplot(111)
            ax.plot(obs_t/86400, y_values, c = 'OrangeRed')
            plt.title(title)
            plt.ylabel(f'strain - band [{self.inifr}, {self.inifr + self.band}][Hz]')
            plt.xlabel('time [days]' + ' - start: ' + self.t0_iso_signal + 'UTC')
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
        
        return y
    
    
    def spectr(self, lfft = 1024, image = False, directory = None):
        """
        It generates the spectrogram of the injected simulated data.
        --------------------------------
        Par:
            lfft (int): fft length for computing the FFT in the spectrogram  (default = 1024)
            image (bool): if True the image of the spectrogram is shown (default = 'False')
            directory (str): directory where to save the spectrograms image (default = None)
        
        Return:
            f (ndarray): array of sample frequencies in [Hz]
            t (ndarray): array of segment times in [days]
            Sxx (ndarray): spectrogram of s(t) (by default, the last axis of Sxx corresponds
                           to the segment times)
        """
        
        x = self.injection()
        
        # make spectrogram
        _, t, spectrogram = signal.spectrogram(x, fs=1/self.dt, window = 'cosine', nperseg = lfft - 1,
                                               noverlap=lfft//8, nfft=lfft, return_onesided=False,
                                               scaling='density', mode='psd')
        
        # shift frequency zero-component to correctly visualize the spectrogram
        spectr_shft = fftshift(spectrogram/np.max(spectrogram), axes=0)
        
        # translate the spectrogram 2D-array to remove the negative frequencies (as computed by SciPy) since h(t) is complex
        spectr = np.vstack((spectr_shft[int(spectrogram.shape[0]/2):, :], spectr_shft[0:int(spectrogram.shape[0]/2), :]))
        
        # define freq and time arrays
        freq = np.arange(0, lfft)/(self.dt*lfft) + self.inifr
        time = (t + self.frequency_band()[-1][0])/86400
        
        if image: # spectrogram image
        
            if directory is None:
                raise ValueError('Specify the directory to save the spectrograms image')
            else:
                pass
                            
            plt.figure(num = None, figsize=(12, 12), tight_layout = True)        
            plt.pcolormesh(t, freq, np.log10(spectr + 1e-15), cmap='viridis', shading='gouraud')
            plt.axis('off')
            plt.title('')
            plt.savefig(directory + 'spectr_s(t)' + '.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        
        else: # spectrogram plot
            
            plt.figure(num = None, figsize = (12, 12), tight_layout = True)             
            col_bar = plt.pcolormesh(time, freq, np.log10(spectr + 1e-15),
                                     cmap='viridis', shading='gouraud')
            plt.colorbar(col_bar)
            plt.title(f'Spectrogram of $s(t) = {self.amp} h(t) + n(t)$')
            plt.ylabel('frequency [Hz]')
            plt.xlabel('time [days]' + ' - start: ' + self.t0_iso_signal + 'UTC')
            plt.show()
            
        return freq, time, spectr
    
    
    def to_pdDataframe(self, objects = 'data', save_to_csv = False, directory = None):  
        """
        It generates a pandas DataFrame with the output data.
        --------------------------------
        Par:
            objects (str): if 'data' the out DataFrame contains the simulated data, if 'parameters'
                           the simulation parameters will be moved in the out DataFrame (default = 'data')
            save_to_csv (bool): if True, it saves the DataFrame with the data (default = False)
            directory (str): path for the directory where the DataFrame will be saved (default = None)
        """
        
        # data dataframe
        if objects == 'data':
            df = pd.DataFrame([self.frequency_band()[0], self.frequency_band()[1], self.frequency_band()[2].real,
                               self.frequency_band()[2].imag, self.data_chunk().real, self.data_chunk().imag,
                               self.injection().real, self.injection().imag]).transpose()
            
            df.columns = ['frequency', 'phase', 'h_real', 'h_imag', 'noise_real',
                          'noise_imag', 'total_out_real', 'total_out_imag']
        
        # parameters dataframe
        elif objects == 'parameters': 
            par = vars(self)
            del par['gout'], par['t'], par['h_t'], par['frequency'], par['Phi']
            
            df = pd.DataFrame(list(par.items()), columns=['Keys', 'Values'])
            df = df.set_index('Keys')
        
        else:
            raise ValueError("Specify objects: 'data' or 'parameters'")
            
        # save the dataframe
        if save_to_csv:
            if directory is None:
                raise ValueError('Specify the directory for the DataFrame')
            elif objects == 'data':
                df.to_csv(directory + 'LT_GWInjectedSignal_' + self.signal_date + '.csv')
            elif objects == 'parameters':
                df.to_csv(directory + 'InjectionParameters_' + self.signal_date + '.csv')
        
        return df


# end