from datetime import datetime, timedelta
import numpy as np
import spiceypy
import os
import pandas as pd


def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2) /1.495978707E8         
    theta = np.arctan2(z,np.sqrt(x**2+ y**2)) * 360 / 2 / np.pi
    phi = np.arctan2(y,x) * 360 / 2 / np.pi                   
    return (r, theta, phi)


def furnish():
    """Main"""
    base = "sta_kernels"
    kernels = ["naif0012.tls", "pck00010.tpc", "de434s.bsp", "heliospheric_v004u.tf", 
               "ahead_2010_208_01.depm.bsp", "ahead_2018_019_01.depm.bsp", "ahead_2015_219_01.depm.bsp",
               "ahead_2015_076_01.depm.bsp", "ahead_2012_138_01.depm.bsp", "ahead_2013_130_01.depm.bsp",
               "ahead_2020_224_01.depm.bsp", "ahead_2023_019_01.depm.bsp", "ahead_2023_026_01.depm.bsp",
               "ahead_2023_040_01.depm.bsp", "ahead_2023_054_01.depm.bsp", "ahead_2023_065_01.depm.bsp", 
               "ahead_science_09.sclk", "ahead_2017_061_5295day_predict.epm.bsp", "ahead_2023_019_01.epm.bsp",
               "ahead_2023_026_01.epm.bsp", "ahead_2023_040_01.epm.bsp", "ahead_2023_054_01.epm.bsp",
               "ahead_2023_065_01.epm.bsp"]
    for kernel in kernels:
        spiceypy.furnsh(os.path.join(base, kernel))    


def get_sta_pos(t):
    if spiceypy.ktotal('ALL') < 1:
        furnish()
    pos = spiceypy.spkpos("STEREO AHEAD", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
    r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
    position = t, pos[0], pos[1], pos[2], r, lat, lon
    return position


def get_sta_pos_range(start, end):
    if spiceypy.ktotal('ALL') < 1:
        furnish()
    t = start
    positions = []
    while t < end:
        pos = spiceypy.spkpos("STEREO AHEAD", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
        r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
        positions.append([t, pos[0], pos[1], pos[2], r, lat, lon])
        t += timedelta(minutes=1)
    return positions


def get_sta_positions(time_series):
    positions = []
    for t in time_series:
        position = get_sta_pos(t)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['timestamp', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions


def main():
    tfmt = r'%Y%m%dT%H%M%S'
    start = datetime.strptime(input('->'), tfmt)
    end = datetime.strptime(input('->'), tfmt)
    x = get_sta_pos(start, end)
    print(x)

if __name__ == '__main__':
    main()

