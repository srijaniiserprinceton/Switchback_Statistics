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
import astrospice
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
plt.ion()

def get_statistic_for_event(arr, time_arr, T1, T2, T3, T4, T5, T6):

    # removing bad time entires
    nan_mask = np.isnan(arr)
    arr = arr[~nan_mask]
    time_arr = time_arr[~nan_mask]

    T1 = np.datetime64(T1)
    T2 = np.datetime64(T2)
    T3 = np.datetime64(T3)
    T4 = np.datetime64(T4)
    T5 = np.datetime64(T5)
    T6 = np.datetime64(T6)

    before_mask = (time_arr >= T1) * (time_arr <= T2)
    SB_mask = (time_arr >= T3) *  (time_arr <= T4)
    after_mask = (time_arr >= T5) * (time_arr <= T6)

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

def find_background_window(Bmag, time_arr, T2, T3, T4, T5, event_no):
    # removing bad time entires
    nan_mask = np.isnan(Bmag)
    Bmag = Bmag[~nan_mask]
    time_arr = time_arr[~nan_mask]

    T2 = np.datetime64(T2)
    T3 = np.datetime64(T3)
    T4 = np.datetime64(T4)
    T5 = np.datetime64(T5)

    # number of grid points in SB patch
    SB_mask = (time_arr >= T3) *  (time_arr <= T4)
    Nt_outside = np.min([np.sum(time_arr <= T2), np.sum(time_arr >= T5)])

    # SB edge idx
    left_edge_idx, right_edge_idx = np.argmin(np.abs(time_arr - T2)), np.argmin(np.abs(time_arr - T5))

    fig, ax = plt.subplots(1, 3, figsize=(12,5))
    ax[0].plot(time_arr, Bmag)
    ax[0].axvline(T2, ls='dashed', color='k')
    ax[0].axvline(T3, ls='solid', color='k')
    ax[0].axvline(T4, ls='solid', color='k')
    ax[0].axvline(T5, ls='dashed', color='k')
    ax[0].set_xlim([time_arr[0], time_arr[-1]])

    Bfit_avg_arr = []

    for i in range(1, Nt_outside):
        B_vals = Bmag[left_edge_idx - i: left_edge_idx]
        t_idx = np.arange(left_edge_idx - i, left_edge_idx)
        B_vals = np.append(B_vals, Bmag[right_edge_idx: right_edge_idx + i])
        t_idx = np.append(t_idx, np.arange(right_edge_idx, right_edge_idx + i))

        slope, intercept = np.polyfit(t_idx, B_vals, deg=1)

        Bfit = slope * t_idx + intercept

        ax[0].plot(time_arr[t_idx], Bfit, '--r', alpha=0.2)

        Bfit_avg_arr.append(Bfit.mean())
    
    ax[1].plot(Bfit_avg_arr, '.k')

    axt = ax[1].twiny()

    counts, vals, im = axt.hist(Bfit_avg_arr, 30, orientation='horizontal', alpha=0.5, color='k')

    ax[2].plot(time_arr, Bmag)
    ax[2].axvline(T2, ls='dashed', color='k')
    ax[2].axvline(T3, ls='solid', color='k')
    ax[2].axvline(T4, ls='solid', color='k')
    ax[2].axvline(T5, ls='dashed', color='k')

    maxcount_idx = np.argmax(counts)
    ax[2].axhline(vals[maxcount_idx], ls='solid', color='k')

    optimal_winidx = np.argmin(np.abs(Bfit_avg_arr - vals[maxcount_idx]))
    
    ax[1].axvline(optimal_winidx, color='red', ls='solid')

    ax[2].set_xlim([time_arr[left_edge_idx - optimal_winidx], time_arr[right_edge_idx + optimal_winidx]])

    plt.subplots_adjust(left= 0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.2, wspace=0.2)

    plt.savefig(f'plots_Larosa/fit_background/fit_{event_no}.png')
    plt.close()

    return time_arr[left_edge_idx] - time_arr[left_edge_idx - optimal_winidx]


def get_data_in_RTN(T1, T2, T3, T4, T5, T6):
    # converting the timestamps to yyyy-mm-dd/hh:mm:ss format
    tstart = f'{T1.year}-{T1.month}-{T1.day}/{T1.hour}:{T1.minute}:{T1.second}'
    tend =  f'{T6.year}-{T6.month}-{T6.day}/{T6.hour}:{T6.minute}:{T6.second}'

    #----------- read the V and B field data from pyspedas in the RTN frame ----------------#
    print('Loading SPC variables.....')
    spc_vars = pyspedas.psp.spc(trange=[tstart, tend], varnames=['vp_fit_RTN', 'sc_pos_HCI', 'wp_fit', 'vp_fit_SC'],
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
    V1_SC, V2_SC, V3_SC = pytplot.data_quants['psp_spc_vp_fit_SC'].data.T
    WP =  pytplot.data_quants['psp_spc_wp_fit'].data

    # resampling BR, BT, BN onto the time_spc
    BR, BT, BN = resampled_B(BR, BT, BN, time_fld, time_spc)

    return np.array([BR, BT, BN]), np.asarray([VR, VT, VN]), np.asarray([V1_SC, V2_SC, V3_SC]), WP, time_spc

if __name__=='__main__':
    #-------selecting the time window-----#
    data = np.load('data_products/Larosa_Catalog/dates_lead_trail.npz', allow_pickle=True)

    # datetime array of SB events
    dt_SB_lead = data['date_lead_vec'][0,:]
    dt_SB_start = data['date_lead_vec'][1,:]
    dt_SB_end = data['date_trail_vec'][0,:]
    dt_SB_trail = data['date_trail_vec'][1,:]

    # total number of SB events in encounter 1
    Nevents = len(dt_SB_start)

    # arranging by time
    sort_idx = np.argsort(dt_SB_start)

    dt_SB_lead = dt_SB_lead[sort_idx]
    dt_SB_start = dt_SB_start[sort_idx]
    dt_SB_end = dt_SB_end[sort_idx]
    dt_SB_trail = dt_SB_trail[sort_idx]

    # list of necessary statistics
    B_ratio = []
    V_ratio = []
    beta_ratio = []

    for i in tqdm(range(Nevents)):
        SB_duration = dt_SB_end[i] - dt_SB_start[i]
        dt_quiet = np.max([0.5 * SB_duration, datetime.timedelta(seconds=120)])
        # obtaining the six timestamps for the specific event
        # adding arbitrary 1 minutes to make up T1 and T6 (only T2, T3, T4, T5 are from catalog)
        # T1, T2, T3, T4, T5, T6 = dt_SB_lead[i] + datetime.timedelta(seconds=-180),\
        #                          dt_SB_lead[i],\
        #                          dt_SB_start[i],\
        #                          dt_SB_end[i],\
        #                          dt_SB_trail[i],\
        #                          dt_SB_trail[i] + datetime.timedelta(seconds=180)
        T1, T2, T3, T4, T5, T6 = dt_SB_lead[i] - dt_quiet,\
                                 dt_SB_lead[i],\
                                 dt_SB_start[i],\
                                 dt_SB_end[i],\
                                 dt_SB_trail[i],\
                                 dt_SB_trail[i] + dt_quiet

        # getting the deflection in Parker Frame
        BRTN, VRTN, VSC, WP, time_spc = get_data_in_RTN(T1, T2, T3, T4, T5, T6)

        # try:
        optimal_windur = find_background_window(np.linalg.norm(BRTN, axis=0), time_spc, T2, T3, T4, T5, i)
        optimal_windur_sec = optimal_windur / np.timedelta64(1, 's')

        T1, T6 = dt_SB_lead[i] - datetime.timedelta(seconds=optimal_windur_sec),\
                dt_SB_trail[i] + datetime.timedelta(seconds=optimal_windur_sec)

        Bmag = np.linalg.norm(BRTN, axis=0)

        B_ratio.append(get_statistic_for_event(Bmag, time_spc, T1, T2, T3, T4, T5, T6))
        V_ratio.append(get_statistic_for_event(np.linalg.norm(VRTN, axis=0), time_spc, T1, T2, T3, T4, T5, T6))

        beta = WP**2 / Bmag**2
        beta_ratio.append(get_statistic_for_event(beta, time_spc, T1, T2, T3, T4, T5, T6))

        # except:
        #     continue

    B_ratio = np.asarray(B_ratio)
    V_ratio = np.asarray(V_ratio)
    beta = np.asarray(beta_ratio)

    fig, ax = plt.subplots(1, 3, figsize=(10,5), sharey=True)
    ax[0].hist(B_ratio, bins=10, range=(0.75, 1.15), color='grey', histtype='bar', ec='black')
    ax[0].set_xlim([0.75, 1.15])
    ax[0].set_ylim([0, 34])
    ax[0].set_xlabel(r'$B_{\rm{ins}} / 0.5 (B_{\rm{bef}} + B_{\rm{aft}})$')
    ax[0].set_ylabel('# events')
    # plt.savefig('B_ratio.png')

    ax[1].hist(V_ratio, bins=10, range=(0.95, 1.45), color='grey', histtype='bar', ec='black')
    ax[1].set_xlim([0.95, 1.45])
    ax[1].set_ylim([0, 34])
    ax[1].set_xlabel(r'$V_{\rm{ins}} / 0.5 (V_{\rm{bef}} + V_{\rm{aft}})$')
    ax[1].set_ylabel('# events')

    ax[2].hist(beta_ratio, bins=10, range=(0, 4.5), color='grey', histtype='bar', ec='black')
    ax[2].set_xlim([0, 4.5])
    ax[2].set_ylim([0, 34])
    ax[2].set_xlabel(r'$\beta_{\rm{ins}} / 0.5 (\beta_{\rm{bef}} + \beta_{\rm{aft}})$')
    ax[2].set_ylabel('# events')
    plt.savefig('Larosa_ratios.png')