import re
import pandas
import pyspedas
import pytplot
import astropy.units as u
from astropy.time import Time, TimeDelta
from sunpy.coordinates import HeliocentricInertial
from astropy import coordinates as coor
import astrospice
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.axes_grid1 import make_axes_locatable

Rsun_km = u.Rsun.to('km')

def datetime(t):
    date, time = re.split('[ ]', t)
    time = re.split('[.]', time)[0]
    return f'{date}/{time}'

def convert_time_format(ta, tb):
    ta = datetime(ta)
    tb = datetime(tb)
    return ta, tb

def convert2datetime64(T1, T2, T3, T4, T5, T6):
    t1 = T1.to_datetime64()
    t2 = T2.to_datetime64()
    t3 = T3.to_datetime64()
    t4 = T4.to_datetime64()
    t5 = T5.to_datetime64()
    t6 = T6.to_datetime64()
    return t1, t2, t3, t4, t5, t6

def resampled_B(BR, BT, BN, time_fld, time_spc):
    dt_fld_mus = time_fld - np.datetime64('2001-01-01T00:00:00')
    dt_fld = dt_fld_mus / np.timedelta64(1, 'us')
    dt_spc_mus = time_spc - np.datetime64('2001-01-01T00:00:00')
    dt_spc = dt_spc_mus / np.timedelta64(1, 'us')
    BR_ = interpolate.interp1d(dt_fld, BR)
    BT_ = interpolate.interp1d(dt_fld, BT)
    BN_ = interpolate.interp1d(dt_fld, BN)
    BR = BR_(dt_spc)
    BT = BT_(dt_spc)
    BN = BN_(dt_spc)
    return BR, BT, BN

def get_radial_distance_from_time(hci_coords_km):
    # getting the distance in km
    dist_km = np.linalg.norm(hci_coords_km, axis=1) * u.km
    dist_Rsun = dist_km.to(u.Rsun)
    
    return dist_Rsun

def get_Bpolar_in_ParkerFrame(T1, T2, T3, T4, T5, T6):
    # converting the timestamps to yyyymmdd/hh:mm:ss format
    tstart, tend = convert_time_format(str(T1), str(T6))
    tstart, tend = '2019-03-30', '2019-04-12'
    # converting to axis units
    T1, T2, T3, T4, T5, T6 = convert2datetime64(T1, T2, T3, T4, T5, T6)

    #----------- read the V and B field data from pyspedas in the RTN frame ----------------#
    # print('Loading SPC variables.....')
    # spc_vars = pyspedas.psp.spc(trange=[tstart, tend], datatype='l3i', level='l3', time_clip=True, no_update=False)
    # print('Loading FIELDS variables.....')
    # fields_vars = pyspedas.psp.fields(trange=[tstart, tend], varnames=['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'],
    #                                   datatype='mag_RTN_4_Sa_per_Cyc', level='l2', time_clip=True, no_update=False)

    # reading the pytplot variables
    # time_fld = pytplot.data_quants['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'].time.data
    # BR, BT, BN = pytplot.data_quants['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'].data.T
    # B = np.sqrt(BR**2 + BT**2 + BN**2)
    # time_spc = pytplot.data_quants['psp_spc_vp_fit_RTN'].time.data
    # VR, VT, VN = pytplot.data_quants['psp_spc_vp_fit_RTN'].data.T

    time_fld = np.load('data_products/Enc2/time_fld_enc2.npy')
    time_spc = np.load('data_products/Enc2/time_spc_enc2.npy')
    BR, BT, BN = np.load('data_products/Enc2/B_enc2.npy')
    VR, VT, VN = np.load('data_products/Enc2/V_enc2.npy')
    hci_km_spc = np.load('data_products/Enc2/hci_spc_km.npy')

    # finding the radial distance
    r = get_radial_distance_from_time(hci_km_spc.data)

    # resampling BR, BT, BN onto the time_spc
    BR, BT, BN = resampled_B(BR, BT, BN, time_fld, time_spc)

    # calculate the angle between radial and parker spiral at each time (alpha_p)
    omega = 2.9e-6   # Sun's angular rotation at the equator in rad/seconds
    r0 = 10 * u.Rsun # in solar radius
    alpha_p = -1 * np.arctan2((-1. * omega * (r - r0) * Rsun_km).value, VR) # * 0.0

    # change the B to |B|, theta, phi frame using B (RTN) and alpha_p
    Bx = BR * np.cos(alpha_p) - BT * np.sin(alpha_p)
    By = BR * np.sin(alpha_p) + BT * np.cos(alpha_p)
    Bz = BN

    # Bx, By, Bz = np.random.rand(3,12890)

    # Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
    # theta = np.arctan2(np.sqrt(Bx**2 + By**2), Bz)
    # phi = np.arctan(By, Bx)
    Bmag, lat, lon = coor.cartesian_to_spherical(Bx, By, Bz)

    return Bmag.value, lat.value, lon.value, alpha_p # theta, phi

if __name__=='__main__':
    #-------selecting the time window-----#
    # from pytplot import tplot
    data = pandas.read_hdf("ApJ_SB_Database.h5")
    data.shape == (5370, 7) # true  
    events = data.groupby(["event number","encounter"])  # 1,074 events

    # event number and encounter
    event_number = 229
    encounter = 1

    # obtaining the six timestamps for the specific event
    T1, T2, T3, T4, T5 = data[(data['event number']==event_number) *\
                            (data['encounter']==encounter)]['start time'].array
    T6 = data[(data['event number']==event_number) *\
            (data['encounter']==encounter)]['end time'].array[-1]

    Bmag, theta, phi, alpha_p = get_Bpolar_in_ParkerFrame(T1, T2, T3, T4, T5, T6)

    # converting the ranges such that theta goes from -pi/2 to pi/2
    # theta = theta - np.pi/2

    # purging nan entries
    nan_idx_theta = np.isnan(theta)
    theta = theta[~nan_idx_theta]
    phi = phi[~nan_idx_theta]
    phi = phi - np.pi 
    
    # plot colormap of (theta, phi) which serves as data to emcee
    hist2D, xedges, yedges = np.histogram2d(theta, phi, bins=(180,360), density=True)
    hist2D[0] = 0.0
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    divider = make_axes_locatable(ax)
    im = ax.pcolormesh(yedges * 180/np.pi, xedges * 180/np.pi, hist2D, cmap='gnuplot')
    ax.axhline(0, color='gray')
    ax.axvline(0, color='gray')
    ax.set_xlabel(r'$\phi$ in degrees')
    ax.set_ylabel(r'$\theta$ in degrees')
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_aspect('equal')
    plt.tight_layout()