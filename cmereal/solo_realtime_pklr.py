import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import spiceypy
import os.path
import urllib.request
import cdflib
import pickle


"""
SOLAR ORBITER SERVER DATA PATH
"""

#solo_path='/perm/aswo/data/solo/' #Note that we need to make a new directory within solo/mag called "ll" i.e. low latency 
#kernels_path='perm/aswo/data/kernels/'

solo_path='/Users/emmadavies/Documents/Data-test/solo/'
kernels_path='/Users/emmadavies/Documents/Data-test/kernels/'

"""
SOLO MAG DATA
# Potentially different SolO MAG file names: internal low latency, formagonly, and formagonly 1 min.
e.g.
- Internal files: solo_L2_mag-rtn-ll-internal_20230225_V00.cdf
- For MAG only files: solo_L2_mag-rtn-normal-formagonly_20200415_V01.cdf
- For MAG only 1 minute res files: solo_L2_mag-rtn-normal-1-minute-formagonly_20200419_V01.cdf
# All in RTN coords
# Should all follow same variable names within cdf
"""

def get_solomag(fp):
    """raw = rtn"""
    try:
        cdf = cdflib.CDF(fp)
        t1 = cdflib.cdfepoch.to_datetime(cdf.varget('EPOCH'))
        df = pd.DataFrame(t1, columns=['time'])
        bx, by, bz = cdf['B_RTN'][:].T
        df['bx'] = bx
        df['by'] = by
        df['bz'] = bz
        df['bt'] = np.linalg.norm(df[['bx', 'by', 'bz']], axis=1)
    except Exception as e:
        print('ERROR:', e, fp)
        df = None
    return df


def get_solomag_range(start_timestamp, end_timestamp=datetime.utcnow(), path=f'{solo_path}'+'mag/ll/'):
    """Pass two datetime objects and grab .cdf files between dates, from
    directory given."""
    df = None
    start = start_timestamp.date()
    end = end_timestamp.date() + timedelta(days=1)
    while start < end:
        date_str = f'{start.year}{start.month:02}{start.day:02}'
        fn = f'{path}solo_L2_mag-rtn-ll-internal_{date_str}_V00.cdf' #update filename depending on internal, formagonly etc. 
        _df = get_solomag(fn)
        if _df is not None:
            if df is None:
                df = _df.copy(deep=True)
            else:
                df = pd.concat([df, _df])
        start += timedelta(days=1)
    return df


"""
SOLO POSITION FUNCTIONS: coord maths, furnish kernels, and call position for each timestamp
"""

def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


#http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/kernels/fk/ for solo_ANC_soc-sci-fk_V08.tf
#http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/kernels/spk/ for solo orbit .bsp

def solo_furnish():
    """Main"""
    solo_path = kernels_path+'solo/'
    generic_path = kernels_path+'generic/'
    solo_kernels = os.listdir(solo_path)
    generic_kernels = os.listdir(generic_path)
    for kernel in solo_kernels:
        spiceypy.furnsh(os.path.join(solo_path, kernel))
    for kernel in generic_kernels:
        spiceypy.furnsh(os.path.join(generic_path, kernel))


def get_solo_pos(t):
    if spiceypy.ktotal('ALL') < 1:
        solo_furnish()
    pos = spiceypy.spkpos("SOLAR ORBITER", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
    r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
    position = t, pos[0], pos[1], pos[2], r, lat, lon
    return position


def get_solo_positions(time_series):
    positions = []
    for t in time_series:
        position = get_solo_pos(t)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['time', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


"""
FINAL FUNCTION TO CREATE PICKLE FILE: uses all above functions to create pickle file of 
data from input timestamp to now. 
Can be read in to DataFrame using:
obj = pd.read_pickle(solo_path+'solo_ll_rtn.p')
df = pd.DataFrame.from_records(obj[0])
"""


def create_solo_pkl(start_timestamp, end_timestamp):
    
    #load in mag data to DataFrame and resample, create empty mag and resampled DataFrame if no data
    # if empty, drop time column ready for concat
    df_mag = get_solomag_range(start_timestamp, end_timestamp)
    if df_mag is None:
        print(f'SolO MAG data is empty for this timerange')
        df_mag = pd.DataFrame({'time':[], 'bt':[], 'bx':[], 'by':[], 'bz':[]})
        mag_rdf = df_mag.drop(columns=['time'])
    else:
        mag_rdf = df_mag.set_index('time').resample('1min').mean().reset_index(drop=False)
        mag_rdf.set_index(pd.to_datetime(mag_rdf['time']), inplace=True)
        
    #get solo positions for corresponding timestamps
    solo_pos = get_solo_positions(mag_rdf['time'])
    solo_pos.set_index(pd.to_datetime(solo_pos['time']), inplace=True)
    solo_pos = solo_pos.drop(columns=['time'])

    #produce final combined DataFrame with correct ordering of columns 
    comb_df = pd.concat([mag_rdf, solo_pos], axis=1)

    #produce recarray with correct datatypes
    time_stamps = comb_df['time']
    dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format

    solo=np.zeros(len(dt_lst),dtype=[('time',object),('bx', float),('by', float),('bz', float),('bt', float),\
                ('x', float),('y', float),('z', float), ('r', float),('lat', float),('lon', float)])
    solo = solo.view(np.recarray) 

    solo.time=dt_lst
    solo.bx=comb_df['bx']
    solo.by=comb_df['by']
    solo.bz=comb_df['bz']
    solo.bt=comb_df['bt']
    solo.x=comb_df['x']
    solo.y=comb_df['y']
    solo.z=comb_df['z']
    solo.r=comb_df['r']
    solo.lat=comb_df['lat']
    solo.lon=comb_df['lon']
    
    #dump to pickle file
    header='Internal low latency solar wind magnetic field (MAG) from Solar Orbiter, ' + \
    'provided directly by Imperial College London SolO MAG Team '+ \
    'Timerange: '+solo.time[0].strftime("%Y-%b-%d %H:%M")+' to '+solo.time[-1].strftime("%Y-%b-%d %H:%M")+\
    ', resampled to a time resolution of 1 min. '+\
    'The data are available in a numpy recarray, fields can be accessed by solo.time, solo.bx, solo.r etc. '+\
    'Total number of data points: '+str(solo.size)+'. '+\
    'Units are btxyz [nT, RTN], heliospheric position x/y/z/r/lon/lat [AU, degree, HEEQ]. '+\
    'Made with [...] by E. Davies (twitter @spacedavies). File creation date: '+\
    datetime.utcnow().strftime("%Y-%b-%d %H:%M")+' UTC'

    pickle.dump([solo,header], open(solo_path+'solo_ll_rtn.p', "wb"))