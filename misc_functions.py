import re
import numpy as np
import datetime
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
import sunpy.coordinates
import astropy.units as u
import pickle
from scipy import interpolate

def gen_dt_arr(dt_init, dt_final, cadence_days=1) :
    """
    'Generate Datetime Array'
    Get array of datetime.datetime from {dt_init} to {dt_final} every 
    {cadence_days} days
    """
    dt_list = []
    while dt_init < dt_final :
        dt_list.append(dt_init)
        dt_init += datetime.timedelta(days=cadence_days)
    return np.array(dt_list)

def datetime2unix(dt_arr) :
    """Convert 1D array of `datetime.datetime` to unix timestamps"""
    return np.array([dt.timestamp() for dt in dt_arr])

def unix2datetime(ut_arr) : 
    """Convert 1D array of unix timestamps (float) to `datetime.datetime`"""
    return np.array([datetime.datetime.utcfromtimestamp(ut) for ut in ut_arr])

def interp_trajectory(dt_in, trajectory_in, dt_out) :
    trajectory_in.representation_type="spherical"
    radius_in = trajectory_in.radius.to("R_sun").value
    lon_in = trajectory_in.lon.to("deg").value
    lat_in = trajectory_in.lat.to("deg").value
    disconts = np.where(np.abs(np.diff(lon_in)) > 60)[0]
    nan_inds = []
    for discont in disconts: 
        nan_inds.append(discont)
        nan_inds.append(discont+1)
    radius_in[nan_inds] = np.nan
    lon_in[nan_inds] = np.nan
    lat_in[nan_inds] = np.nan
    return SkyCoord(
        radius=interp1d(datetime2unix(dt_in),radius_in,bounds_error=False)(datetime2unix(dt_out))*u.R_sun,
        lon=interp1d(datetime2unix(dt_in),lon_in,bounds_error=False)(datetime2unix(dt_out))*u.deg,
        lat=interp1d(datetime2unix(dt_in),lat_in,bounds_error=False)(datetime2unix(dt_out))*u.deg,        
        representation_type="spherical",
        frame=sunpy.coordinates.HeliographicCarrington,
        obstime=dt_out)

def spherical_distance(coor1, coor2):
    # unpacking the coordinates
    r1, theta1, phi1 = coor1.radius.to(u.km), coor1.lat.to(u.rad), coor1.lon.to(u.rad)
    r2, theta2, phi2 = coor2.radius.to(u.km), coor2.lat.to(u.rad), coor2.lon.to(u.rad)

    # calculating spherical distance
    dist_sq = r1**2 + r2**2 - 2*r1*r2*(np.sin(theta1)*np.sin(theta2)*np.cos(phi1-phi2)\
                            + np.cos(theta1)*np.cos(theta2))
    dist = np.sqrt(dist_sq)

    # returns distances in Mm
    return dist.to(u.Mm)

def angdist_on_sphere(coor1, coor2):
    '''
    Calculating the distance between two points assuming they are
    on a unit sphere.
    '''
    theta1, phi1 = coor1.lat.to(u.rad), coor1.lon.to(u.rad)
    theta2, phi2 = coor2.lat.to(u.rad), coor2.lon.to(u.rad)
    cos_theta = np.sin(theta1) * np.sin(theta2) + np.cos(theta1) * np.cos(theta2) * np.cos(phi1 - phi2)

    # using small angle approximation in angular minutes
    return np.sqrt(2 * (1 - cos_theta)) * 180 / np.pi * 60


def write_pickle(x, fname):
    with open(f'{fname}.pkl', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(fname):
    with open(f'{fname}.pkl', 'rb') as handle:
        x = pickle.load(handle)
    return x

def timestr_to_dt64(time_str):
    date, time = re.split('[/]', time_str)
    return np.datetime64(f'{date}T{time}')

def dt64_to_timestr(time_str):
    date, time = re.split('[T]', str(time_str))
    return f'{date}/{time}'

def get_radial_distance_from_time(hci_coords_km):
    # getting the distance in km
    dist_km = np.linalg.norm(hci_coords_km, axis=1) * u.km
    dist_Rsun = dist_km.to(u.Rsun)
    
    return dist_Rsun

def resampled_B(BR, BT, BN, time_fld, time_spc):
    dt_fld_mus = time_fld - np.datetime64('2001-01-01T00:00:00')
    dt_fld = dt_fld_mus / np.timedelta64(1, 'us')
    dt_spc_mus = time_spc - np.datetime64('2001-01-01T00:00:00')
    dt_spc = dt_spc_mus / np.timedelta64(1, 'us')
    BR_ = interpolate.interp1d(dt_fld, BR, bounds_error=False)
    BT_ = interpolate.interp1d(dt_fld, BT, bounds_error=False)
    BN_ = interpolate.interp1d(dt_fld, BN, bounds_error=False)
    BR = BR_(dt_spc)
    BT = BT_(dt_spc)
    BN = BN_(dt_spc)
    return BR, BT, BN