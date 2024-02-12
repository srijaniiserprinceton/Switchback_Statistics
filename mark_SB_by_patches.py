import re
import sys
import pandas
import pytplot
import pyspedas
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import misc_functions as misc_FN

# loading the metadict for the desired encounter
enc = 'E01'
metadict = misc_FN.read_pickle(f'{enc}_metadict')

onset_time = []
xy_cen = []

for key in metadict.keys():
    onset_time.append(metadict[f'{key}']['onset_time'])
    xy_cen.append(metadict[f'{key}']['xy_cen'])

# converting them to arrays from lists
onset_time = np.asarray(onset_time)
xy_cen = np.asarray(xy_cen)

SB_start_times = []
SB_end_times = []

# loading the csv file from the Huang catalog
data = pandas.read_csv(f"./data_products/Huang2023Catalog/{enc}_PSP_switchback_event_list.csv") 
T3_arr = data['spike Start Time']
T4_arr = data['spike End Time']

for i in range(len(onset_time)):
    T3_idx = np.where(T3_arr == onset_time[i])[0][0]
    T4 = T4_arr[T3_idx]

    T3_dt64 = misc_FN.timestr_to_dt64(T3_arr[T3_idx])
    T4_dt64 = misc_FN.timestr_to_dt64(T4_arr[T3_idx])

    SB_start_times.append(T3_dt64)
    SB_end_times.append(T4_dt64)

# converting list to arrays and excluding the very first SB
SB_start_times = np.asarray(SB_start_times)[1:]
SB_end_times = np.asarray(SB_end_times)[1:]


# constructing the array of angular distances of z-angles
z_offset = []

for i in range(1, len(xy_cen)):
    phi1, theta1 = xy_cen[i,-1] * np.pi / 180
    phi2, theta2 = xy_cen[i-1,-1] * np.pi / 180
    # phi2, theta2 = 0.0, 0.0
    z_offset.append(np.arccos(np.sin(theta1) * np.sin(theta2) + np.cos(theta1) * np.cos(theta2) * np.cos(phi1 - phi2)))

z_offset = np.asarray(z_offset)

# sys.exit()

z_offset_norm = (z_offset - z_offset.min()) / (z_offset.max() - z_offset.min())
color = cm.jet(z_offset_norm)

# splitting up the plot into 3days each
plot_start_times = np.arange(SB_start_times[0], SB_start_times[-1]+1, np.timedelta64(3, 'D'))
plot_end_times = plot_start_times[1:]
plot_end_times = np.append(plot_end_times, SB_start_times[-1])

for i in range(len(plot_start_times)):
    tstart, tend = misc_FN.dt64_to_timestr(plot_start_times[i]) , misc_FN.dt64_to_timestr(plot_end_times[i])

    # getting the timeseries of B and V to plot
    spc_vars = pyspedas.psp.spc(trange=[tstart, tend], varnames=['vp_fit_RTN', 'sc_pos_HCI'],
                                datatype='l3i', level='l3', time_clip=True, no_update=True)
    fields_vars = pyspedas.psp.fields(trange=[tstart, tend], varnames=['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'],
                                      datatype='mag_RTN_4_Sa_per_Cyc', level='l2', time_clip=True, no_update=True)

    # reading the pytplot variables
    time_fld = pytplot.data_quants['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'].time.data
    time_spc = pytplot.data_quants['psp_spc_vp_fit_RTN'].time.data
    hci_km_spc = pytplot.data_quants['psp_spc_sc_pos_HCI']
    r = misc_FN.get_radial_distance_from_time(hci_km_spc.data)

    BR, BT, BN = pytplot.data_quants['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'].data.T
    # getting the resampled BRTN according to spc times
    BR, BT, BN = misc_FN.resampled_B(BR, BT, BN, time_fld, time_spc)
    B = np.sqrt(BR**2 + BT**2 + BN**2)
    VR, VT, VN = pytplot.data_quants['psp_spc_vp_fit_RTN'].data.T
    V = np.sqrt(VR**2 + VT**2 + VN**2)

    # creating subplots
    fig, ax = plt.subplots(2, 1, figsize=(10,8))

    # coloring the patches by z-angle deflection of SB patches
    SB_timemask = (SB_start_times >= plot_start_times[i]) * (SB_end_times <= plot_end_times[i])
    SB_start_times_thiswindow = SB_start_times[SB_timemask]
    SB_end_times_thiswindow = SB_end_times[SB_timemask]
    color_thiswindow = color[SB_timemask]

    for j in range(len(SB_start_times_thiswindow)):
        ax[0].axvspan(SB_start_times_thiswindow[j], SB_end_times_thiswindow[j],
                      color=color[j], alpha=0.5)
        im = ax[1].axvspan(SB_start_times_thiswindow[j], SB_end_times_thiswindow[j],
                      color=color[j], alpha=0.5)

    # divider = make_axes_locatable(ax[1])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im, cax=cax, orientation='vertical')

    # normalizing with respect to the first point in the time series
    ax[0].plot(time_spc, BR * (r / r[0])**2, 'k')
    ax[1].plot(time_spc, VR, 'k')

    ax[0].set_xlim([plot_start_times[i], plot_end_times[i]])
    ax[1].set_xlim([plot_start_times[i], plot_end_times[i]])
    ax[1].set_ylim([0, None])

    ax[0].set_ylabel(r'$B_r$ [nT]')
    ax[1].set_ylabel(r'$V_r$ [km/s]')

    plt.subplots_adjust(wspace=0.0)

    plt.savefig(f'plots/{enc}_{i}.pdf')