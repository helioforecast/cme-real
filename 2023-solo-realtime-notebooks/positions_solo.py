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
    base = "solo_kernels"
    kernels = [
        "de430.bsp", "naif0012.tls", "heliospheric_v004u.tf", "pck00010.tpc", 
        "solo_ANC_soc-orbit_20200210-20301120_L011_V1_00200_V01.bsp", "solo_ANC_soc-sci-fk_V08.tf"]
    for kernel in kernels:
        spiceypy.furnsh(os.path.join(base, kernel))


def get_solo_pos(t):
    if spiceypy.ktotal('ALL') < 1:
        furnish()
    pos = spiceypy.spkpos("SOLAR ORBITER", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
    r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
    position = t, pos[0], pos[1], pos[2], r, lat, lon
    return position


def get_solo_pos_range(start, end):
    if spiceypy.ktotal('ALL') < 1:
        furnish()
    t = start
    positions = []
    while t < end:
        pos = spiceypy.spkpos("SOLAR ORBITER", spiceypy.datetime2et(t), "HEEQ", "NONE", "SUN")[0]
        r, lat, lon = cart2sphere(pos[0],pos[1],pos[2])
        positions.append([t, pos[0], pos[1], pos[2], r, lat, lon])
        t += timedelta(minutes=1)
    return positions


def get_solo_positions(time_series):
    positions = []
    for t in time_series:
        position = get_solo_pos(t)
        positions.append(position)
    df_positions = pd.DataFrame(positions, columns=['timestamp', 'x', 'y', 'z', 'r', 'lat', 'lon'])
    return df_positions