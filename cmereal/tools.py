#tools.py for cme-real

import numpy as np
import pandas as pd
import scipy
import copy
import matplotlib.dates as mdates
import datetime
import urllib
import json
import os
import pdb
from sunpy.time import parse_time
import scipy.io
import scipy.signal
import pickle
import time
import sys
import cdflib
import matplotlib.pyplot as plt
#import heliosat  #not compatible with astrospice, problems with spiceypy in astrospice.generate
from numba import njit
from astropy.time import Time
import heliopy.data.cassini as cassinidata
import heliopy.data.helios as heliosdata
import heliopy.data.spice as spicedata
import heliopy.spice as spice
import astropy
import requests
import math
import h5py

#from config import data_path
#data_path='/nas/helio/data/insitu_python/'



def omni_loader():
   '''
   downloads all omni2 data into the "data" folder
   '''
   print('load OMNI2 .dat into "data" directory from')
   omni2_url='https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat'
   print(omni2_url)
   try: urllib.request.urlretrieve(omni2_url, 'data/omni2_all_years.dat')
   except urllib.error.URLError as e:
         print(' ', omni2_url,' ',e.reason)
         sys.exit()


def save_omni_data(path,file):
    '''
    save variables from OMNI2 dataset as pickle

    documentation https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text
    omni2_url='https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat'
    '''        
  
    print('start omni')
    
    omni_loader()
    #check how many rows exist in this file
    f=open('data/omni2_all_years.dat')
    dataset= len(f.readlines())
    
    #make array
    o=np.zeros(dataset,dtype=[('time',object),('bx', float),('by', float),\
                ('bz', float),('bygsm', float),('bzgsm', float),('bt', float),\
                ('vt', float),('np', float),('tp', float),('alpha', float),\
                ('dst', float),('kp', float),('spot', float),\
                ('ae', float),('ap', float),('f107', float),\
                ('pcn', float),('al', float),('au', float),\
                ('x', float),('y', float),('z', float),\
                ('r', float),('lat', float),('lon', float)])
                
    o=o.view(np.recarray)  
    print(dataset, ' datapoints')   #for reading data from OMNI file
    
    j=0
    with open('data/omni2_all_years.dat') as f:
        for line in f:
            line = line.split() # to deal with blank 
            
            #time - need to convert from year doy hour to datetime object
            o.time[j]=datetime.datetime(int(line[0]), 1, 1) + datetime.timedelta(int(line[1]) - 1) \
                              + datetime.timedelta(hours=int(line[2])) 

            #25 is bulkspeed F6.0, in km/s
            o.vt[j]=line[24]
            if o.vt[j] == 9999: o.vt[j]=np.NaN
            
            #24 in file, index 23 proton density /ccm
            o.np[j]=line[23]
            if o.np[j] == 999.9: o.np[j]=np.NaN
            
            #23 in file, index 22 Proton temperature  /ccm
            o.tp[j]=line[22]
            if o.tp[j] == 9999999.: o.tp[j]=np.NaN

            #28 in file, index 27 alpha to proton ratio
            o.alpha[j]=line[27]
            if o.alpha[j] == 9.999: o.alpha[j]=np.NaN
           
            #9 is total B  F6.1 also fill ist 999.9, in nT
            o.bt[j]=line[9]
            if o.bt[j] == 999.9: o.bt[j]=np.NaN

            #GSE components from 13 to 15, so 12 to 14 index, in nT
            o.bx[j]=line[12]
            if o.bx[j] == 999.9: o.bx[j]=np.NaN
            o.by[j]=line[13]
            if o.by[j] == 999.9: o.by[j]=np.NaN
            o.bz[j]=line[14]
            if o.bz[j] == 999.9: o.bz[j]=np.NaN
          
            #GSM
            o.bygsm[j]=line[15]
            if o.bygsm[j] == 999.9: o.bygsm[j]=np.NaN
          
            o.bzgsm[j]=line[16]
            if o.bzgsm[j] == 999.9: o.bzgsm[j]=np.NaN    
          

            o.kp[j]=line[38]
            if o.kp[j] == 99: o.kp[j]=np.nan
           
            o.spot[j]=line[39]
            if o.spot[j] ==  999: o.spot[j]=np.nan

            o.dst[j]=line[40]
            if o.dst[j] == 99999: o.dst[j]=np.nan

            o.ae[j]=line[41]
            if o.ae[j] ==  9999: o.ae[j]=np.nan
            
            o.ap[j]=line[49]
            if o.ap[j] ==  999: o.ap[j]=np.nan
            
            o.f107[j]=line[50]
            if o.f107[j] ==  999.9  : o.f107[j]=np.nan
            
            o.pcn[j]=line[51]
            if o.pcn[j] ==  999.9  : o.pcn[j]=np.nan

            o.al[j]=line[52]
            if o.al[j] ==   99999  : o.al[j]=np.nan

            o.au[j]=line[53]
            if o.au[j] ==  99999 : o.au[j]=np.nan
          
          
            j=j+1     
            
    #print('position start')
    #frame='HEEQ'
    #planet_kernel=spicedata.get_kernel('planet_trajectories')
    #earth=spice.Trajectory('399')  #399 for Earth, not barycenter (because of moon)
    #earth.generate_positions(o.time,'Sun',frame)
    #earth.change_units(astropy.units.AU)
    #[r, lat, lon]=cart2sphere(earth.x,earth.y,earth.z)
    #print('position end ')   
    
    
    #o.x=earth.x
    #o.y=earth.y
    #o.z=earth.z
    
    #o.r=r
    #o.lat=np.rad2deg(lat)
    #o.lon=np.rad2deg(lon)
    
    header='Near Earth OMNI2 1 hour solar wind and geomagnetic indices data since 1963. ' + \
    'Obtained from https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/  '+ \
    'Timerange: '+o.time[0].strftime("%Y-%b-%d %H:%M")+' to '+o.time[-1].strftime("%Y-%b-%d %H:%M")+'. '+\
    'The data are available in a numpy recarray, fields can be accessed by o.time, o.bx, o.vt etc. '+\
    'Missing data has been set to "np.nan". Total number of data points: '+str(o.size)+'. '+\
    'For units and documentation see: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text, the '+\
    'heliospheric position of Earth was added and is given in x/y/z/r/lon/lat [AU, degree, HEEQ]. '+\
    'Made with https://github.com/cmoestl/heliocats heliocats.data.save_omni_data (uses https://github.com/ajefweiss/HelioSat '+\
    'and https://github.com/heliopython/heliopy). '+\
    'By C. Moestl (twitter @chrisoutofspace), A. J. Weiss, and D. Stansby. File creation date: '+\
    datetime.datetime.utcnow().strftime("%Y-%b-%d %H:%M")+' UTC'
    
    
    pickle.dump([o,header], open(path+file, 'wb') )
    
    print('done omni')
    print()

    