import numpy as np
import pytz
import datetime
from tqdm import tqdm
import astropy.units as u
import matplotlib.pyplot as plt
plt.ion()

import PSP_orbit
import shared_functions as shared_FN
import misc_functions as misc_FN
import get_enc_times

utc=pytz.UTC

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

def get_statistics():
    enc = 1
    enc_start_times, enc_end_times, enc_num = get_enc_times.get_PSP_enc()

    # loading the catalog
    data = np.load('data_products/Larosa_Catalog/dates_lead_trail.npz', allow_pickle=True)

    # datetime array of SB events
    dt_SB_lead = data['date_lead_vec'][0,:]
    dt_SB_start = data['date_lead_vec'][1,:]
    dt_SB_end = data['date_trail_vec'][0,:]
    dt_SB_trail = data['date_trail_vec'][1,:]

    # arranging by time
    sort_idx = np.argsort(dt_SB_start)

    dt_SB_lead = dt_SB_lead[sort_idx]
    dt_SB_start = dt_SB_start[sort_idx]
    dt_SB_end = dt_SB_end[sort_idx]
    dt_SB_trail = dt_SB_trail[sort_idx]

    # total number of SB events in encounter 1
    Nevents = len(dt_SB_start)

    # enc_orb_times, enc_orb_coords = PSP_orbit.get_PSP_orbit_coords(tstart_enc, tend_enc)
    dt_start_enc, dt_end_enc = utc.localize(enc_start_times[enc-1]), utc.localize(enc_end_times[enc-1])
    dt_mission = PSP_orbit.get_PSP_orbit_spice(dt_start_enc, dt_end_enc)

    # obtaining the interpolated PSP coorinates
    PSP_coor_SB_events = PSP_orbit.get_PSP_orbit_highres(dt_mission, dt_SB_start)

    # distance from last SB in-situ location
    last_event_orb_coords = PSP_coor_SB_events[0]
    last_event_time = dt_SB_start[0]

    # metadata dictionary
    metadict = {}

    for i in tqdm(range(Nevents)):
        SB_duration = dt_SB_end[i] - dt_SB_start[i]
        dt_quiet = np.max([0.5 * SB_duration, datetime.timedelta(seconds=120)])

        T1, T2, T3, T4, T5, T6 = dt_SB_lead[i] - dt_quiet,\
                                 dt_SB_lead[i],\
                                 dt_SB_start[i],\
                                 dt_SB_end[i],\
                                 dt_SB_trail[i],\
                                 dt_SB_trail[i] + dt_quiet

        event_orb_coords = PSP_coor_SB_events[i]
        PSP_dist_Rsun = event_orb_coords.radius.to(u.Rsun).value

        # calculating distance from last event
        distance_offset = misc_FN.spherical_distance(event_orb_coords, last_event_orb_coords)
        angdist_onsphr_offset = misc_FN.angdist_on_sphere(event_orb_coords, last_event_orb_coords)
        time_offset = (dt_SB_start[i] - last_event_time).total_seconds()

        # try:
        # getting the deflection in Parker Frame
        T1_formatted = f'{T1.year}-{T1.month}-{T1.day}/{T1.hour}:{T1.minute}:{T1.second}'
        T6_formatted = f'{T6.year}-{T6.month}-{T6.day}/{T6.hour}:{T6.minute}:{T6.second}'
        theta, phi, alpha_p, BRTN, VRTN, WP, N_p, time_spc = shared_FN.get_Bpolar_in_ParkerFrame(T1_formatted, T6_formatted)

        optimal_windur = find_background_window(np.linalg.norm(BRTN, axis=0), time_spc, T2, T3, T4, T5, i)
        optimal_windur_sec = optimal_windur / np.timedelta64(1, 's')

        T1, T6 = dt_SB_lead[i] - datetime.timedelta(seconds=optimal_windur_sec),\
                 dt_SB_trail[i] + datetime.timedelta(seconds=optimal_windur_sec)

        Bmag = np.linalg.norm(BRTN, axis=0)
        Vmag = np.linalg.norm(VRTN, axis=0)
        beta = WP**2 / Bmag**2

        B_ratio = shared_FN.get_statistic_for_event(Bmag, time_spc, T1, T2, T3, T4, T5, T6)
        V_ratio = shared_FN.get_statistic_for_event(Vmag, time_spc, T1, T2, T3, T4, T5, T6)
        beta_ratio = shared_FN.get_statistic_for_event(beta, time_spc, T1, T2, T3, T4, T5, T6)

        # clipping the theta and phi arrays to the SB spike interval only
        T3_datetime64, T4_datetime64 = np.datetime64(T3), np.datetime64(T4)
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
        axs.text(0.1, 0.8, f'Encounter: E01')
        axs.text(0.1, 0.7, f'SB onset Date/Time: {T3}')
        axs.text(0.1, 0.6, f'SB duration [s]: {SB_duration:.2f}')
        axs.text(0.1, 0.5, f'Distance from Sun [Rsun]: {PSP_dist_Rsun:.2f}')
        axs.text(0.1, 0.4, f'Distance offset [Rsun]: {distance_offset:.2f}')
        axs.text(0.1, 0.3, f'Ang angle offset [ang min]: {angdist_onsphr_offset:.2f}')
        axs.text(0.1, 0.2, f'Time offset [sec]: {time_offset:.2f}')

        axs.axis('off')
        axs = fig.add_subplot(222)
        xy_cen = shared_FN.make_theta_phi_zangle_plot(fig, axs, theta, phi)
        axs = fig.add_subplot(223)
        axs.scatter(PSP_coor_SB_events.lon, PSP_coor_SB_events.lat, color="black", s=1)
        axs.scatter(PSP_coor_SB_events.lon[i], PSP_coor_SB_events.lat[i], c='red', s=15, zorder=1)
        axs = fig.add_subplot(224)
        shared_FN.make_BRTN_plots(axs, time_spc, BRTN, T1, T2, T3, T4, T5, T6)
        plt.tight_layout()
        plt.savefig(f'Larosa_plots/{enc}_{i}.pdf')
        plt.close()

        last_event_orb_coords = event_orb_coords
        last_event_time = dt_SB_end[i]

        #------ Agapitov+2023 statistics ----------#
        theta_agp, alpha_agp, sig_c, sig_r, time_nonan = shared_FN.get_Agapitov_params(N_p, VRTN, BRTN, time_spc, T1, T2)
        
        # adding to the dictionary
        metadict[f'{enc}_{i}'] = {}
        metadict[f'{enc}_{i}']['onset_time'] = T3
        metadict[f'{enc}_{i}']['duration'] = SB_duration
        metadict[f'{enc}_{i}']['dist_from_Sun'] = PSP_dist_Rsun
        metadict[f'{enc}_{i}']['distance_offset'] = distance_offset
        metadict[f'{enc}_{i}']['angdist_onsphr_offset'] = angdist_onsphr_offset
        metadict[f'{enc}_{i}']['time_offset'] = time_offset
        metadict[f'{enc}_{i}']['xy_cen'] = xy_cen
        metadict[f'{enc}_{i}']['B_ratio'] = B_ratio
        metadict[f'{enc}_{i}']['V_ratio'] = V_ratio
        metadict[f'{enc}_{i}']['beta_ratio'] = beta_ratio
        metadict[f'{enc}_{i}']['theta_arr'] = theta_agp
        metadict[f'{enc}_{i}']['alpha_arr'] = alpha_agp
        metadict[f'{enc}_{i}']['sig_c_arr'] = sig_c
        metadict[f'{enc}_{i}']['sig_r_arr'] = sig_r
        metadict[f'{enc}_{i}']['time_nonan'] = time_nonan

        # except:
        #     continue

    
    # saving the dictionary in a pickle file
    misc_FN.write_pickle(metadict, f'{enc}_metadict_Larosa')