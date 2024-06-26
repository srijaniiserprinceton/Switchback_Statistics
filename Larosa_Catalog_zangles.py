import re
import sys
import pytz
import pandas
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from photutils.centroids import (centroid_1dg, centroid_2dg,
                                 centroid_com, centroid_quadratic)

import PSP_orbit
import misc_functions as misc_FN
import get_enc_times

import astrospice_demo

Rsun_km = u.Rsun.to('km')
utc=pytz.UTC

# new_frame = HeliocentricInertial()
new_frame = HeliographicCarrington(observer='self')

def to_datetime64(t):
    # date, time = re.split('[/]', t)
    return np.datetime64(t)

def convert_time_format(ta, tb):
    ta = datetime(ta)
    tb = datetime(tb)
    return ta, tb

def convert2datetime64(T1, T2, T3, T4, T5, T6):
    t1 = to_datetime64(T1)
    t2 = to_datetime64(T2)
    t3 = to_datetime64(T3)
    t4 = to_datetime64(T4)
    t5 = to_datetime64(T5)
    t6 = to_datetime64(T6)
    return t1, t2, t3, t4, t5, t6

def get_event_times(event):
    T1 = event['LQP Start Time'].values[0]
    T2 = event['LQP End Time'].values[0]
    T3 = event['spike Start Time'].values[0]
    T4 = event['spike End Time'].values[0]
    T5 = event['TQP Start Time'].values[0]
    T6 = event['TQP End Time'].values[0]

    return T1, T2, T3, T4, T5, T6

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

def get_radial_distance_from_time(hci_coords_km):
    # getting the distance in km
    dist_km = np.linalg.norm(hci_coords_km, axis=1) * u.km
    dist_Rsun = dist_km.to(u.Rsun)
    
    return dist_Rsun

def get_Bpolar_in_ParkerFrame(T1, T2, T3, T4, T5, T6):
    # converting the timestamps to yyyymmdd/hh:mm:ss format
    # tstart, tend = convert_time_format(str(T1), str(T6))
    # converting to axis units
    # T1, T2, T3, T4, T5, T6 = convert2datetime64(T1, T2, T3, T4, T5, T6)
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

    # time_fld = np.load('data_products/Enc2/time_fld_enc2.npy')
    # time_spc = np.load('data_products/Enc2/time_spc_enc2.npy')
    # BR, BT, BN = np.load('data_products/Enc2/B_enc2.npy')
    # VR, VT, VN = np.load('data_products/Enc2/V_enc2.npy')
    # hci_km_spc = np.load('data_products/Enc2/hci_spc_km.npy')

    # finding the radial distance
    hci_km_spc = pytplot.data_quants['psp_spc_sc_pos_HCI']
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

    return Bmag.value, lat.value, lon.value, alpha_p, np.array([BR, BT, BN]), np.asarray([VR, VT, VN]), time_spc

def grid2coor(xy):
    x, y = xy
    X = x * 360/50 - 180
    Y = y * 180/25 - 90
    return (X, Y)

# def make_theta_phi_zangle_plot(fig, ax, xedges, yedges, hist2D):
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
    xycen1 = grid2coor(centroid_com(h))
    xycen2 = grid2coor(centroid_quadratic(h))
    xycen3 = grid2coor(centroid_1dg(h))
    xycen4 = grid2coor(centroid_2dg(h))
    xycens = [xycen1, xycen2, xycen3, xycen4]
    colors = ('white', 'black', 'red', 'blue')
    for xycen, color in zip(xycens, colors):
        ax.plot(*xycen, color=color, marker='+', ms=15, mew=2.0)

    # returns the different estimates of centroids
    return xycens


def make_BRTN_plots(ax, time_event, BRTN, T1, T2, T3, T4, T5, T6):
    B = np.linalg.norm(BRTN, axis=0)
    BR, BT, BN = BRTN
    T1, T2, T3, T4, T5, T6 = convert2datetime64(T1, T2, T3, T4, T5, T6)
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

if __name__=='__main__':
    # selecting encounter to analyze. E1,2,4-8 are available
    enc = int(sys.argv[1])

    enc_start_times, enc_end_times, enc_num = get_enc_times.get_PSP_enc()

    #-------selecting the time window-----#
    data = np.load('data_products/Larosa_Catalog/dates_lead_trail.npz', allow_pickle=True)

    enc_mask = (data['date_lead_vec'][0,:] > utc.localize(enc_start_times[enc-1])) *\
               (data['date_trail_vec'][1,:] < utc.localize(enc_end_times[enc-1]))

    Nevents = np.sum(enc_mask)

    # enc_orb_times, enc_orb_coords = PSP_orbit.get_PSP_orbit_coords(tstart_enc, tend_enc)
    dt_start_enc, dt_end_enc = utc.localize(enc_start_times[enc-1]), utc.localize(enc_end_times[enc-1])
    dt_mission = PSP_orbit.get_PSP_orbit_spice(dt_start_enc, dt_end_enc)

    # datetime array of SB events
    dt_SB_lead = data['date_lead_vec'][0,:][enc_mask]
    dt_SB_start = data['date_lead_vec'][1,:][enc_mask]
    dt_SB_end = data['date_trail_vec'][0,:][enc_mask]
    dt_SB_trail = data['date_trail_vec'][1,:][enc_mask]

    # arranging by time
    sort_idx = np.argsort(dt_SB_start)

    dt_SB_lead = dt_SB_lead[sort_idx]
    dt_SB_start = dt_SB_start[sort_idx]
    dt_SB_end = dt_SB_end[sort_idx]
    dt_SB_trail = dt_SB_trail[sort_idx]

    # obtaining the interpolated PSP coorinates
    PSP_coor_SB_events = PSP_orbit.get_PSP_orbit_highres(dt_mission, dt_SB_start)

    # distance from last SB in-situ location
    last_event_orb_coords = PSP_coor_SB_events[0]
    last_event_time = dt_SB_start[0]

    # metadata dictionary
    metadict = {}

    for i in tqdm(range(Nevents)):
        # obtaining the six timestamps for the specific event
        # adding arbitrary 2 minutes to make up T1, T2, T5, T6 (only T3 and T4 are real)
        T1, T2, T3, T4, T5, T6 = dt_SB_lead[i] + datetime.timedelta(seconds=-60),\
                                 dt_SB_lead[i],\
                                 dt_SB_start[i],\
                                 dt_SB_end[i],\
                                 dt_SB_trail[i],\
                                 dt_SB_trail[i] + datetime.timedelta(seconds=60)
        # event_orb_coords = astrospice.generate_coords('SOLAR PROBE PLUS', re.split('[/]', T3)[0])
        # event_orb_coords = event_orb_coords.transform_to(new_frame)
        event_orb_coords = PSP_coor_SB_events[i]
        # PSP_dist_Rsun = event_orb_coords.distance.to(u.Rsun).value[0]
        # PSP_dist_Rsun = event_orb_coords.radius.to(u.Rsun).value[0]
        PSP_dist_Rsun = event_orb_coords.radius.to(u.Rsun).value

        # calculating distance from last event
        # print(event_orb_coords, last_event_orb_coords)
        distance_offset = misc_FN.spherical_distance(event_orb_coords, last_event_orb_coords)
        angdist_onsphr_offset = misc_FN.angdist_on_sphere(event_orb_coords, last_event_orb_coords)
        time_offset = (dt_SB_start[i] - last_event_time).total_seconds()

        # getting the deflection in Parker Frame
        Bmag, theta, phi, alpha_p, BRTN, VRTN, time_spc = get_Bpolar_in_ParkerFrame(T1, T2, T3, T4, T5, T6)

        # clipping the theta and phi arrays to the SB spike interval only
        T3_datetime64, T4_datetime64 = np.datetime64(T3), np.datetime64(T4) #to_datetime64(T3), to_datetime64(T4)
        SB_duration = (T4_datetime64 - T3_datetime64)/np.timedelta64(1, 's')
        time_mask = (time_spc >= T3_datetime64) * (time_spc <= T4_datetime64)
        theta, phi = theta[time_mask], phi[time_mask]

        # purging nan entries
        nan_idx_theta = np.isnan(theta)
        theta = theta[~nan_idx_theta]
        phi = phi[~nan_idx_theta]
        phi = phi - np.pi 
        
        # plot colormap of (theta, phi) which serves as data to emcee
        fig = plt.figure(figsize=(8,5))
        axs = fig.add_subplot(221)
        axs.text(0.1, 0.8, f'Encounter: {enc}')
        T3_plot = datetime.datetime(T3.year, T3.month, T3.day, T3.hour, T3.minute, T3.second)
        axs.text(0.1, 0.7, f'SB onset Date/Time: {T3_plot}')
        axs.text(0.1, 0.6, f'SB duration [s]: {SB_duration:.2f}')
        axs.text(0.1, 0.5, f'Distance from Sun [Rsun]: {PSP_dist_Rsun:.2f}')
        axs.text(0.1, 0.4, f'Distance offset [Rsun]: {distance_offset:.2f}')
        axs.text(0.1, 0.3, f'Ang angle offset [ang min]: {angdist_onsphr_offset:.2f}')
        axs.text(0.1, 0.2, f'Time offset [sec]: {time_offset:.2f}')

        axs.axis('off')
        axs = fig.add_subplot(222)
        try:
            xy_cen = make_theta_phi_zangle_plot(fig, axs, theta, phi)
            axs = fig.add_subplot(223)
            # axs.plot(enc_orb_coords.lon.to(u.rad), enc_orb_coords.distance.to(u.au), 'grey', zorder=0)
            # axs.scatter(event_orb_coords.lon.to(u.rad), event_orb_coords.distance.to(u.au), c='red', s=15, zorder=1)
            # axs.plot(enc_orb_coords.lon.to(u.rad), enc_orb_coords.radius.to(u.au), 'grey', zorder=0)
            axs.scatter(PSP_coor_SB_events.lon, PSP_coor_SB_events.lat, color="black", s=1)
            axs.scatter(PSP_coor_SB_events.lon[i], PSP_coor_SB_events.lat[i], c='red', s=15, zorder=1)
            axs = fig.add_subplot(224)
            make_BRTN_plots(axs, time_spc, BRTN, T1, T2, T3, T4, T5, T6)
            plt.tight_layout()
            plt.savefig(f'plots_Larosa/{enc}_{i}.pdf')
            plt.close()

            last_event_orb_coords = event_orb_coords
            last_event_time = dt_SB_events[i]
            
            # adding to the dictionary
            metadict[f'{enc}_{i}'] = {}
            metadict[f'{enc}_{i}']['onset_time'] = T3
            metadict[f'{enc}_{i}']['duration'] = SB_duration
            metadict[f'{enc}_{i}']['dist_from_Sun'] = PSP_dist_Rsun
            metadict[f'{enc}_{i}']['distance_offset'] = distance_offset
            metadict[f'{enc}_{i}']['angdist_onsphr_offset'] = angdist_onsphr_offset
            metadict[f'{enc}_{i}']['time_offset'] = time_offset
            metadict[f'{enc}_{i}']['xy_cen'] = xy_cen
        except:
            continue

    
    # saving the dictionary in a pickle file
    misc_FN.write_pickle(metadict, f'{enc}_metadict_Larosa')