import sys
import pytz
from tqdm import tqdm
import pyspedas
import pytplot
import datetime
import astropy.units as u
from astropy.time import Time, TimeDelta
# from sunpy.coordinates import HeliocentricInertial
from sunpy.coordinates import HeliographicCarrington
from astropy import coordinates as coor
# import astrospice
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
plt.ion()

import get_enc_times
import Larosa_statistics_functions as Larosa_FN

import astrospice_demo

Rsun_km = u.Rsun.to('km')
utc=pytz.UTC

# new_frame = HeliocentricInertial()
new_frame = HeliographicCarrington(observer='self')


def get_statistic_for_event(arr, time_arr, T2, T3, T4, T5):

    T2 = np.datetime64(T2)
    T3 = np.datetime64(T3)
    T4 = np.datetime64(T4)
    T5 = np.datetime64(T5)

    before_mask = (time_arr >= T2) * (time_arr <= T3)
    SB_mask = (time_arr >= T3) *  (time_arr <= T4)
    after_mask = (time_arr >= T4) * (time_arr <= T5)

    val_SB = np.mean(arr[SB_mask])
    val_before = np.mean(arr[before_mask])
    val_after = np.mean(arr[after_mask])

    return val_SB / (0.5 * (val_before + val_after))

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

def get_Bpolar_in_ParkerFrame(T1, T2, T3, T4, T5, T6):
    # converting the timestamps to yyyy-mm-dd/hh:mm:ss format
    tstart = f'{T1.year}-{T1.month}-{T1.day}/{T1.hour}:{T1.minute}:{T1.second}'
    tend =  f'{T6.year}-{T6.month}-{T6.day}/{T6.hour}:{T6.minute}:{T6.second}'

    #----------- read the V and B field data from pyspedas in the RTN frame ----------------#
    print('Loading SPC variables.....')
    spc_vars = pyspedas.psp.spc(trange=[tstart, tend], varnames=['vp_fit_RTN', 'sc_pos_HCI'],
                                datatype='l3i', level='l3', time_clip=True, no_update=False)
    print('Loading FIELDS variables.....')
    fields_vars = pyspedas.psp.fields(trange=[tstart, tend], varnames=['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'],
                                      datatype='mag_RTN_4_Sa_per_Cyc', level='l2', time_clip=True, no_update=False)

    # reading the pytplot variables
    time_fld = pytplot.data_quants['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'].time.data
    BR, BT, BN = pytplot.data_quants['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'].data.T
    B = np.sqrt(BR**2 + BT**2 + BN**2)
    time_spc = pytplot.data_quants['psp_spc_vp_fit_RTN'].time.data
    VR, VT, VN = pytplot.data_quants['psp_spc_vp_fit_RTN'].data.T

    # resampling BR, BT, BN onto the time_spc
    BR, BT, BN = resampled_B(BR, BT, BN, time_fld, time_spc)

    return np.array([BR, BT, BN]), np.asarray([VR, VT, VN]), time_spc

if __name__=='__main__':
    # selecting encounter to analyze. E1,2,4-8 are available
    enc = int(sys.argv[1])

    enc_start_times, enc_end_times, enc_num = get_enc_times.get_PSP_enc()

    #-------selecting the time window-----#
    data = np.load('data_products/Larosa_Catalog/dates_lead_trail.npz', allow_pickle=True)

    enc_mask = (data['date_lead_vec'][0,:] > utc.localize(enc_start_times[enc-1])) *\
               (data['date_trail_vec'][1,:] < utc.localize(enc_end_times[enc-1]))

    # datetime array of SB events
    dt_SB_lead = data['date_lead_vec'][0,:][enc_mask]
    dt_SB_start = data['date_lead_vec'][1,:][enc_mask]
    dt_SB_end = data['date_trail_vec'][0,:][enc_mask]
    dt_SB_trail = data['date_trail_vec'][1,:][enc_mask]

    Nevents = np.sum(enc_mask)

    # arranging by time
    sort_idx = np.argsort(dt_SB_start)

    dt_SB_lead = dt_SB_lead[sort_idx]
    dt_SB_start = dt_SB_start[sort_idx]
    dt_SB_end = dt_SB_end[sort_idx]
    dt_SB_trail = dt_SB_trail[sort_idx]

    B_ratio = []

    for i in tqdm(range(Nevents)):
        # obtaining the six timestamps for the specific event
        # adding arbitrary 2 minutes to make up T1, T2, T5, T6 (only T3 and T4 are real)
        T1, T2, T3, T4, T5, T6 = dt_SB_lead[i] + datetime.timedelta(seconds=-60),\
                                 dt_SB_lead[i],\
                                 dt_SB_start[i],\
                                 dt_SB_end[i],\
                                 dt_SB_trail[i],\
                                 dt_SB_trail[i] + datetime.timedelta(seconds=60)

        # getting the deflection in Parker Frame
        Bmag, theta, phi, alpha_p, BRTN, VRTN, time_spc = get_Bpolar_in_ParkerFrame(T1, T2, T3, T4, T5, T6)

        try:
            B_ratio.append(get_statistic_for_event(np.linalg.norm(BRTN, axis=0), time_spc, T2, T3, T4, T5))

        except:
            continue

    B_ratio = np.asarray(B_ratio)

    plt.hist(B_ratio, bins=9, range=(0.75, 1.15))