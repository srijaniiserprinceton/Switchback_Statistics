import numpy as np
import pandas 
import pytz
import datetime
from tqdm import tqdm
import astropy.units as u
import matplotlib.pyplot as plt
plt.ion()

import PSP_orbit
import misc_functions as misc_FN
import shared_functions as shared_FN
import get_enc_times

utc=pytz.UTC

def get_statistics(enc):
    enc_start_times, enc_end_times, enc_num = get_enc_times.get_PSP_enc()

    # loading the catalog
    data = np.load('data_products/Vamsee_Catalog/dictionary_Switch_back_analysis_srijan_das.npy', allow_pickle=True)

    enc_mask = (data.item()['New_Time_SB_patch_starting'] > utc.localize(enc_start_times[enc-1])) *\
               (data.item()['New_Time_SB_patch_starting'] < utc.localize(enc_end_times[enc-1]))

    Nevents = np.sum(enc_mask)

    # enc_orb_times, enc_orb_coords = PSP_orbit.get_PSP_orbit_coords(tstart_enc, tend_enc)
    dt_start_enc, dt_end_enc = utc.localize(enc_start_times[enc-1]), utc.localize(enc_end_times[enc-1])
    dt_mission = PSP_orbit.get_PSP_orbit_spice(dt_start_enc, dt_end_enc)

    # datetime array of SB events
    dt_SB_events = data.item()['New_Time_SB_patch_starting'][enc_mask]
    dt_SB_window = data.item()['life_time_SB_patch'][enc_mask] # in seconds

    # obtaining the interpolated PSP coorinates
    PSP_coor_SB_events = PSP_orbit.get_PSP_orbit_highres(dt_mission, dt_SB_events)

    # distance from last SB in-situ location
    last_event_orb_coords = PSP_coor_SB_events[0]
    last_event_time = dt_SB_events[0]

    # metadata dictionary
    metadict = {}

    for i in tqdm(range(Nevents)):
        # obtaining the six timestamps for the specific event
        # adding arbitrary 2 minutes to make up T1, T2, T5, T6 (only T3 and T4 are real)
        T1, T2, T3, T4, T5, T6 = dt_SB_events[i] + datetime.timedelta(seconds=-240),\
                                 dt_SB_events[i] + datetime.timedelta(seconds=-120),\
                                 dt_SB_events[i],\
                                 dt_SB_events[i] + datetime.timedelta(seconds=int(dt_SB_window[i])),\
                                 dt_SB_events[i] + datetime.timedelta(seconds=int(dt_SB_window[i])+120),\
                                 dt_SB_events[i] + datetime.timedelta(seconds=int(dt_SB_window[i])+240)

        event_orb_coords = PSP_coor_SB_events[i]
        PSP_dist_Rsun = event_orb_coords.radius.to(u.Rsun).value

        # calculating distance from last event
        distance_offset = misc_FN.spherical_distance(event_orb_coords, last_event_orb_coords)
        angdist_onsphr_offset = misc_FN.angdist_on_sphere(event_orb_coords, last_event_orb_coords)
        time_offset = (dt_SB_events[i] - last_event_time).total_seconds()

        # getting the deflection in Parker Frame
        T1_formatted = f'{T1.year}-{T1.month}-{T1.day}/{T1.hour}:{T1.minute}:{T1.second}'
        T6_formatted = f'{T6.year}-{T6.month}-{T6.day}/{T6.hour}:{T6.minute}:{T6.second}'
        theta, phi, alpha_p, BRTN, VRTN, WP, time_spc = shared_FN.get_Bpolar_in_ParkerFrame(T1_formatted, T6_formatted)

        # getting Larosa statistics
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
        # try:
        xy_cen = shared_FN.make_theta_phi_zangle_plot(fig, axs, theta, phi)
        axs = fig.add_subplot(223)
        axs.scatter(PSP_coor_SB_events.lon, PSP_coor_SB_events.lat, color="black", s=1)
        axs.scatter(PSP_coor_SB_events.lon[i], PSP_coor_SB_events.lat[i], c='red', s=15, zorder=1)
        axs = fig.add_subplot(224)
        shared_FN.make_BRTN_plots(axs, time_spc, BRTN, T1, T2, T3, T4, T5, T6)
        plt.tight_layout()
        plt.savefig(f'Vamsee_plots/{enc}_{i}.pdf')
        plt.close()

        last_event_orb_coords = event_orb_coords
        last_event_time = T4
        
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
        # except:
        #     continue

    
    # saving the dictionary in a pickle file
    misc_FN.write_pickle(metadict, f'{enc}_metadict_Vamsee')