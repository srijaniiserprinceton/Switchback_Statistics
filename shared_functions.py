import numpy as np
import pyspedas
import pytplot
import astropy.units as u
import astropy.constants as const
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
plt.ion()
from scipy import interpolate
from astropy import coordinates as coor
from photutils.centroids import (centroid_1dg, centroid_2dg,
                                 centroid_com, centroid_quadratic)

import misc_functions as misc_FN

Rsun_km = u.Rsun.to('km')

def get_Bpolar_in_ParkerFrame(T1, T6):
    tstart, tend = T1, T6

    #----------- read the V and B field data from pyspedas in the RTN frame ----------------#
    print('Loading SPC variables.....')
    spc_vars = pyspedas.psp.spc(trange=[tstart, tend], varnames=['np_fit', 'vp_fit_RTN', 'sc_pos_HCI', 'wp_fit'],
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
    WP =  pytplot.data_quants['psp_spc_wp_fit'].data
    N_p = pytplot.data_quants['psp_spc_np_fit'].data

    # finding the radial distance
    hci_km_spc = pytplot.data_quants['psp_spc_sc_pos_HCI']
    r = get_radial_distance_Rsun_from_hci(hci_km_spc.data)

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

    Bmag, lat, lon = coor.cartesian_to_spherical(Bx, By, Bz)

    return lat.value, lon.value, alpha_p, np.array([BR, BT, BN]), np.asarray([VR, VT, VN]), WP, N_p, time_spc

def get_radial_distance_Rsun_from_hci(hci_coords_km):
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

def make_BRTN_plots(ax, time_event, BRTN, T1, T2, T3, T4, T5, T6):
    B = np.linalg.norm(BRTN, axis=0)
    BR, BT, BN = BRTN
    T1, T2, T3, T4, T5, T6 = misc_FN.convert2datetime64(T1, T2, T3, T4, T5, T6)
    ax.plot(time_event, B, label=r'$B\,[\mathrm{nT}]$')
    ax.plot(time_event, BR, label=r'$B_R\,[\mathrm{nT}]$')
    ax.plot(time_event, BT, label=r'$B_T\,[\mathrm{nT}]$')
    ax.plot(time_event, BN, label=r'$B_N\,[\mathrm{nT}]$')
    # ax.set_xlim([time_event[0], time_event[-1]])
    ax.set_xlim([T1, T6])
    # ax.legend(ncol=4)
    ax.set_ylabel('B[nT]')
    ax.axvline(T2, ls='--', color='black')
    ax.axvline(T3, ls='--', color='black')
    ax.axvline(T4, ls='--', color='black')
    ax.axvline(T5, ls='--', color='black')
    ax.axvspan(T1, T2, alpha=0.2, color='red')
    ax.axvspan(T2, T3, alpha=0.2, color='pink')
    ax.axvspan(T3, T4, alpha=0.2, color='blue')
    ax.axvspan(T4, T5, alpha=0.2, color='orange')
    ax.axvspan(T5, T6, alpha=0.2, color='green')

def make_theta_phi_zangle_plot(fig, ax, ztheta, zphi):
    divider = make_axes_locatable(ax)
    h, __, __, im = ax.hist2d(zphi * 180/np.pi, ztheta * 180/np.pi, bins=[50,25], range=[[-180, 180],[-90, 90]], cmap='gnuplot', rasterized=True)
    ax.axhline(0, color='gray')
    ax.axvline(0, color='gray')
    ax.set_xlabel(r'$\phi$ in degrees')
    ax.set_ylabel(r'$\theta$ in degrees')
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_aspect('equal')

    # calculating and plotting the centroids
    h = h.T
    xycen1 = misc_FN.grid2coor(centroid_com(h))
    xycen2 = misc_FN.grid2coor(centroid_quadratic(h))
    xycen3 = misc_FN.grid2coor(centroid_1dg(h))
    xycen4 = misc_FN.grid2coor(centroid_2dg(h))
    xycens = [xycen1, xycen2, xycen3, xycen4]
    colors = ('white', 'black', 'red', 'blue')
    for xycen, color in zip(xycens, colors):
        ax.plot(*xycen, color=color, marker='+', ms=15, mew=2.0)

    # returns the different estimates of centroids
    return xycens

def get_statistic_for_event(arr, time_arr, T1, T2, T3, T4, T5, T6):

    # removing bad time entires
    nan_mask = np.isnan(arr)
    arr = arr[~nan_mask]
    time_arr = time_arr[~nan_mask]

    try:
        T1 = np.datetime64(T1)
        T2 = np.datetime64(T2)
        T3 = np.datetime64(T3)
        T4 = np.datetime64(T4)
        T5 = np.datetime64(T5)
        T6 = np.datetime64(T6)
    except: 
        T1 = misc_FN.timestr_to_dt64(T1)
        T2 = misc_FN.timestr_to_dt64(T2)
        T3 = misc_FN.timestr_to_dt64(T3)
        T4 = misc_FN.timestr_to_dt64(T4)
        T5 = misc_FN.timestr_to_dt64(T5)
        T6 = misc_FN.timestr_to_dt64(T6)

    before_mask = (time_arr >= T1) * (time_arr <= T2)
    SB_mask = (time_arr >= T3) *  (time_arr <= T4)
    after_mask = (time_arr >= T5) * (time_arr <= T6)

    val_SB = np.mean(arr[SB_mask])
    val_before = np.mean(arr[before_mask])
    val_after = np.mean(arr[after_mask])

    return val_SB / (0.5 * (val_before + val_after))

def get_DV_DB_for_event(N_p, V_RTN, B_RTN, time_arr, T1, T2):

    # removing bad time entires
    nan_mask = np.isnan(V_RTN) + np.isnan(B_RTN)
    nan_mask = np.sum(nan_mask, axis=0)
    nan_mask = nan_mask > 0
    V_RTN = V_RTN[:, ~nan_mask]
    B_RTN = B_RTN[:, ~nan_mask]
    N_p = N_p[~nan_mask]
    time_arr = time_arr[~nan_mask]

    try:
        T1 = np.datetime64(T1)
        T2 = np.datetime64(T2)
    except:
        T1 = misc_FN.timestr_to_dt64(T1)
        T2 = misc_FN.timestr_to_dt64(T2)

    # Agapitov+2023 say that they only consider the step with respect to the before SB conditions
    before_mask = (time_arr >= T1) * (time_arr <= T2)

    T_bef_mean_idx = np.argmin(np.abs(time_arr - T2))

    # below is a choice I am making in terms of what I consider background solar wind
    # velocity
    # V_bef_mean = np.mean(V_RTN[:, before_mask], axis=1)
    V_bef_mean = V_RTN[:, T_bef_mean_idx] * np.random.rand(1)
    DV = V_RTN - V_bef_mean[:, np.newaxis]

    # magnetic field
    # B_bef_mean = np.mean(B_RTN[:, before_mask], axis=1)
    B_bef_mean = B_RTN[:, T_bef_mean_idx] * np.random.rand(1)
    DB = B_RTN - B_bef_mean[:, np.newaxis]

    return N_p, DV, DB, B_bef_mean, time_arr

def get_Agapitov_params(N_p, V_RTN, B_RTN, time_arr, T1, T2):

    N_p, DV, DB, B_bef_mean, time_arr = get_DV_DB_for_event(N_p, V_RTN, B_RTN, time_arr, T1, T2)

    DB_u = misc_FN.unit_vector(DB)

    V_par = np.sum(DV * DB_u, axis=0)
    V_A = (DB * u.nT / np.sqrt(const.mu0 * N_p * u.cm**-3 * const.m_p)).to("km/s")
    V_A = V_A.value

    B_bef_mean_arr = np.repeat([B_bef_mean], DB_u.shape[1], axis=0).T
    B_bef_mean_u = misc_FN.unit_vector(B_bef_mean_arr)
    theta = np.arccos(np.clip(np.sum(DB_u * B_bef_mean_u, axis=0), -1.0, 1.0))
    alpha = V_par / np.linalg.norm(V_A, axis=0)
    sig_c = 2 * np.sum(V_A * DV, axis=0) / (np.linalg.norm(V_A, axis=0)**2 + np.linalg.norm(DV, axis=0)**2)
    sig_r = (np.linalg.norm(V_A, axis=0)**2 - np.linalg.norm(DV, axis=0)**2)\
            /(np.linalg.norm(V_A, axis=0)**2 + np.linalg.norm(DV, axis=0)**2)

    return theta, alpha, sig_c, sig_r, time_arr