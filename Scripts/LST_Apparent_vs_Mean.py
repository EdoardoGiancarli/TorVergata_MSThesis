# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:15:10 2023

@author: edosc
"""


import numpy as np
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic, EarthLocation, ITRS
import astropy.units as u
from astropy.time import Time
import time
import matplotlib.pyplot as plt



lat, long, azmt, interf_heigth = 0.53342313506, -1.58430937078, 4.40317772346, -6.574
signal_date = '2019-05-31 12:40:51.946'
time_vec = np.arange(0, 3*86400, 0.5)



# define the interferometer location
location = EarthLocation.from_geodetic(long*u.radian, lat*u.radian, interf_heigth*u.m)

# simulation UTC date
actual_time_utc = Time(signal_date, format='iso', scale='utc')

# GPS time array
actual_time_gps = actual_time_utc.gps
time_gps = time_vec + actual_time_gps

# Local Sidereal Time (LST) of the interferometer
gps_time = Time(time_gps, format = 'gps', scale = 'utc', location = location)
lst = gps_time.sidereal_time('apparent').radian

start_time = time.time()
lst = gps_time.sidereal_time('apparent').radian
end_time = time.time()
print("Elapsed time: {:.2f} s".format(end_time - start_time))


start_time = time.time()
gmst = gps_time.sidereal_time('mean').radian
end_time = time.time()
print("Elapsed time: {:.2f} s".format(end_time - start_time))

d = gmst - lst
# plt.plot(time_vec, d)
plt.plot(time_vec, lst)
# plt.plot(time_vec, gmst)
plt.show()



