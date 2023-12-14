import re
import cdflib
import pandas
import pyspedas
import pytplot
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def get_params(tstart, tend, cdf_file):
    cdf_data = cdflib.CDF(cdf_file)

    # reading the timestamps array
    time = cdf_data.varget('epoch_mag_RTN')

    # reading the magnetic field products in RTN format
    B = cdf_data.varget('psp_fld_l2_mag_RTN')

    return B, time

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

def plot_psp_profiles(T1, T2, T3, T4, T5, T6):
    # converting the timestamps to yyyymmdd/hh:mm:ss format
    tstart, tend = convert_time_format(str(T1), str(T6))
    # converting to axis units
    T1, T2, T3, T4, T5, T6 = convert2datetime64(T1, T2, T3, T4, T5, T6)

    spc_vars = pyspedas.psp.spc(trange=[tstart, tend], datatype='l3i', level='l3', time_clip=True)
    fields_vars = pyspedas.psp.fields(trange=[tstart, tend], varnames=['psp_fld_l2_mag_RTN'], datatype='mag_rtn', level='l2', time_clip=True)
    # pytplot.tplot(['psp_fld_l2_mag_RTN', 'psp_spc_vp_fit_RTN'], combine_axes=True)

    # reading the pytplot variables
    time_fld = pytplot.data_quants['psp_fld_l2_mag_RTN'].time
    BR, BT, BN = pytplot.data_quants['psp_fld_l2_mag_RTN'].data.T
    B = np.sqrt(BR**2 + BT**2 + BN**2)
    time_spc = pytplot.data_quants['psp_spc_vp_fit_RTN'].time
    VR, VT, VN = pytplot.data_quants['psp_spc_vp_fit_RTN'].data.T
    V = np.sqrt(VR**2 + VT**2 + VN**2)

    # trying out manual plot for more flexibility
    plt.figure()
    fig, ax = plt.subplots(2, 1, figsize=(10,6), sharex=True)

    ax[0].plot(time_fld, B, label=r'$B\,[\mathrm{nT}]$')
    ax[0].plot(time_fld, BR, label=r'$B_R\,[\mathrm{nT}]$')
    ax[0].plot(time_fld, BT, label=r'$B_T\,[\mathrm{nT}]$')
    ax[0].plot(time_fld, BN, label=r'$B_N\,[\mathrm{nT}]$')
    ax[0].set_xlim([time_fld[0], time_fld[-1]])
    ax[0].legend(ncol=4)
    ax[0].set_ylabel('B[nT]')
    ax[0].axvline(T2, ls='--', color='black')
    ax[0].axvline(T3, ls='--', color='black')
    ax[0].axvline(T4, ls='--', color='black')
    ax[0].axvline(T5, ls='--', color='black')
    ax[0].axvspan(T1, T2, alpha=0.2, color='red')
    ax[0].axvspan(T2, T3, alpha=0.2, color='pink')
    ax[0].axvspan(T3, T4, alpha=0.2, color='blue')
    ax[0].axvspan(T4, T5, alpha=0.2, color='orange')
    ax[0].axvspan(T5, T6, alpha=0.2, color='green')

    ax[1].plot(time_spc, V, label=r'$V_{\mathrm{sw}}\,[\mathrm{km/s}]$')
    ax[1].plot(time_spc, VR, label=r'$V_R\,[\mathrm{km/s}]$')
    ax[1].plot(time_spc, VT, label=r'$V_T\,[\mathrm{km/s}]$')
    ax[1].plot(time_spc, VN, label=r'$V_N\,[\mathrm{km/s}]$')
    ax[1].set_xlim([time_spc[0], time_spc[-1]])
    ax[1].legend(ncol=4)
    ax[1].set_ylabel(r'$\mathrm{V}_{\mathrm{sw}}$[km/s]')
    ax[1].set_xlabel('Time')
    ax[1].axvline(T2, ls='--', color='black')
    ax[1].axvline(T3, ls='--', color='black')
    ax[1].axvline(T4, ls='--', color='black')
    ax[1].axvline(T5, ls='--', color='black')
    ax[1].axvspan(T1, T2, alpha=0.2, color='red')
    ax[1].axvspan(T2, T3, alpha=0.2, color='pink')
    ax[1].axvspan(T3, T4, alpha=0.2, color='blue')
    ax[1].axvspan(T4, T5, alpha=0.2, color='orange')
    ax[1].axvspan(T5, T6, alpha=0.2, color='green')

    plt.subplots_adjust(wspace=0.0, hspace=0.0)


if __name__=='__main__':
    # from pytplot import tplot
    data = pandas.read_hdf("ApJ_SB_Database.h5")
    data.shape == (5370, 7) # true  
    events = data.groupby(["event number","encounter"])  # 1,074 events

    # event number and encounter
    event_number = 229
    encounter = 1

    # obtaining the six timestamps for the specific event
    T1, T2, T3, T4, T5 = data[(data['event number'] == 229) * (data['encounter']==1)]['start time'].array
    T6 = data[(data['event number'] == 229) * (data['encounter']==1)]['end time'].array[-1]

    # choosing a specific event and its corresponding times
    # tstart, tend = '2018-11-5/12:52:00', '2018-11-5/13:25:00'
    plot_psp_profiles(T1, T2, T3, T4, T5, T6)


    
