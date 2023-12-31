import re
import pandas
import pyspedas
import pytplot
import astropy.units as u
from sunpy.coordinates import HeliocentricInertial
import astrospice
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

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

def get_Bpolar_in_ParkerFrame(T1, T2, T3, T4, T5, T6):
    # converting the timestamps to yyyymmdd/hh:mm:ss format
    tstart, tend = convert_time_format(str(T1), str(T6))
    # converting to axis units
    T1, T2, T3, T4, T5, T6 = convert2datetime64(T1, T2, T3, T4, T5, T6)

    # finding the radial distance
    coords = astrospice.generate_coords('SOLAR PROBE PLUS', T4)
    new_frame = HeliocentricInertial()
    coords = coords.transform_to(new_frame)
    # using one radial distance for now since we are fixing the event and encounter     
    r = coords.distance.to(u.Rsun).value[0]

    #----------- read the V and B field data from pyspedas in the RTN frame ----------------#
    spc_vars = pyspedas.psp.spc(trange=[tstart, tend], datatype='l3i', level='l3', time_clip=True)
    fields_vars = pyspedas.psp.fields(trange=[tstart, tend], varnames=['psp_fld_l2_mag_RTN'], datatype='mag_rtn', level='l2', time_clip=True)

    # reading the pytplot variables
    time_fld = pytplot.data_quants['psp_fld_l2_mag_RTN'].time
    BR, BT, BN = pytplot.data_quants['psp_fld_l2_mag_RTN'].data.T
    B = np.sqrt(BR**2 + BT**2 + BN**2)
    time_spc = pytplot.data_quants['psp_spc_vp_fit_RTN'].time
    VR, VT, VN = pytplot.data_quants['psp_spc_vp_fit_RTN'].data.T

    # calculate the angle between radial and parker spiral at each time (alpha_p)
    omega = 2.9e-6   # Sun's angular rotation at the equator in rad/seconds
    r0 = 10 # in solar radius
    alpha_p = np.arctan2(-1. * omega * (r - r0), VR[0])

    # change the B to |B|, theta, phi frame using B (RTN) and alpha_p
    Bx = BR * np.cos(alpha_p) - BT * np.sin(alpha_p)
    By = BR * np.sin(alpha_p) + BT * np.cos(alpha_p)
    Bz = BN

    # Bx, By, Bz = np.random.rand(3,12890)

    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
    theta = np.arctan2(np.sqrt(Bx**2 + By**2), Bz)
    phi = np.arctan2(By, Bx)

    return theta, phi

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

    theta, phi = get_Bpolar_in_ParkerFrame(T1, T2, T3, T4, T5, T6)

    # plot colormap of (theta, phi) which serves as data to emcee
    hist2D, xedges,yedges = np.histogram2d(theta, phi, bins=(180,360), density=True)
    plt.figure()
    plt.pcolormesh(hist2D, cmap='gnuplot')