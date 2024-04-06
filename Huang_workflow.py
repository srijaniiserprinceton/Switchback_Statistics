import re
import numpy as np
import pandas 
import datetime
from tqdm import tqdm
import astropy.units as u
import matplotlib.pyplot as plt
plt.ion()

import PSP_orbit
import misc_functions as misc_FN
import shared_functions as shared_FN

def get_event_times(event):
    T1 = event['LQP Start Time'].values[0]
    T2 = event['LQP End Time'].values[0]
    T3 = event['spike Start Time'].values[0]
    T4 = event['spike End Time'].values[0]
    T5 = event['TQP Start Time'].values[0]
    T6 = event['TQP End Time'].values[0]

    return T1, T2, T3, T4, T5, T6

def get_statistics(enc):
    if(enc < 10):
        enc = f'E0{enc}'
    else:
        enc = f'E{enc}'

    # loading the catalog
    data = pandas.read_csv(f"./data_products/Huang2023Catalog/{enc}_PSP_switchback_event_list.csv") 

    # minimum Quality Flag value
    QF_threshold = 3
    data = data[data['Quality Flag']>=QF_threshold]

    # total number of filtered SB events in encounter
    Nevents = data.shape[0]

    # getting the PSP orbit coordinates at resolution of half a day
    start_case_no, end_case_no = data['Case Number'].keys()[0], data['Case Number'].keys()[-1]
    tstart_enc, tend_enc = data['LQP Start Time'][start_case_no], data['TQP End Time'][end_case_no]
    tstart_enc = np.asarray(re.split('[-,/,:]', tstart_enc)).astype('int')
    tend_enc = np.asarray(re.split('[-,/,:]', tend_enc)).astype('int')
    dt_start_enc, dt_end_enc = datetime.datetime(*tstart_enc), datetime.datetime(*tend_enc)
    dt_mission = PSP_orbit.get_PSP_orbit_spice(dt_start_enc, dt_end_enc)

    # datetime array of SB events
    SB_times = data['spike Start Time']
    dt_SB_events = []
    for idx_key in SB_times.keys():
        dt_SB_events.append(datetime.datetime(*np.asarray(re.split('[-,/,:]', SB_times[idx_key])).astype('int')))
    dt_SB_events = np.asarray(dt_SB_events)

    # obtaining the interpolated PSP coorinates
    PSP_coor_SB_events = PSP_orbit.get_PSP_orbit_highres(dt_mission, dt_SB_events)

    last_event_orb_coords = PSP_coor_SB_events[0]
    last_event_time = np.datetime64(dt_SB_events[0])

    # metadata dictionary
    metadict = {}

    for i, event_key in tqdm(enumerate(data['Case Number'].keys())):
        # event number and encounter
        case_number = event_key + 1

        # Quality factor of event
        QF = data['Quality Flag'][event_key]

        # obtaining the six timestamps for the specific event
        T1, T2, T3, T4, T5, T6 = get_event_times(data[data['Case Number']==case_number])
        event_orb_coords = PSP_coor_SB_events[i]
        PSP_dist_Rsun = event_orb_coords.radius.to(u.Rsun).value

        # calculating distance from last event
        distance_offset = misc_FN.spherical_distance(event_orb_coords, last_event_orb_coords)
        angdist_onsphr_offset = misc_FN.angdist_on_sphere(event_orb_coords, last_event_orb_coords)
        time_offset = (np.datetime64(dt_SB_events[i]) - last_event_time) / np.timedelta64(1, 's')

        # getting the deflection in Parker Frame
        theta, phi, alpha_p, BRTN, VRTN, WP, time_spc = shared_FN.get_Bpolar_in_ParkerFrame(T1, T6)

        # getting Larosa statistics
        Bmag = np.linalg.norm(BRTN, axis=0)
        Vmag = np.linalg.norm(VRTN, axis=0)
        beta = WP**2 / Bmag**2

        B_ratio = shared_FN.get_statistic_for_event(Bmag, time_spc, T1, T2, T3, T4, T5, T6)
        V_ratio = shared_FN.get_statistic_for_event(Vmag, time_spc, T1, T2, T3, T4, T5, T6)
        beta_ratio = shared_FN.get_statistic_for_event(beta, time_spc, T1, T2, T3, T4, T5, T6)

        # clipping the theta and phi arrays to the SB spike interval only
        T3_datetime64, T4_datetime64 = misc_FN.timestr_to_dt64(T3), misc_FN.timestr_to_dt64(T4)
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
        axs.text(0.1, 0.7, f'SB onset Date/Time: {T3}')
        axs.text(0.1, 0.6, f'SB duration [s]: {SB_duration:.2f}')
        axs.text(0.1, 0.5, f'Distance from Sun [Rsun]: {PSP_dist_Rsun:.2f}')
        axs.text(0.1, 0.4, f'QF: {QF} [Min QF: {QF_threshold}]')
        axs.text(0.1, 0.3, f'Distance offset [Rsun]: {distance_offset:.2f}')
        axs.text(0.1, 0.2, f'Ang angle offset [ang min]: {angdist_onsphr_offset:.2f}')
        axs.text(0.1, 0.1, f'Time offset [sec]: {time_offset:.2f}')

        axs.axis('off')
        axs = fig.add_subplot(222)
        xy_cen = shared_FN.make_theta_phi_zangle_plot(fig, axs, theta, phi)
        axs = fig.add_subplot(223)
        axs.scatter(PSP_coor_SB_events.lon, PSP_coor_SB_events.lat, color="black", s=1)
        axs.scatter(PSP_coor_SB_events.lon[i], PSP_coor_SB_events.lat[i], c='red', s=15, zorder=1)
        axs = fig.add_subplot(224)
        shared_FN.make_BRTN_plots(axs, time_spc, BRTN, T1, T2, T3, T4, T5, T6)
        plt.tight_layout()
        plt.savefig(f'Huang_plots/{enc}_{case_number}.pdf')
        plt.close()

        last_event_orb_coords = event_orb_coords
        last_event_time = T4_datetime64
        
        # adding to the dictionary
        metadict[f'{enc}_{case_number}'] = {}
        metadict[f'{enc}_{case_number}']['onset_time'] = T3
        metadict[f'{enc}_{case_number}']['duration'] = SB_duration
        metadict[f'{enc}_{case_number}']['dist_from_Sun'] = PSP_dist_Rsun
        metadict[f'{enc}_{case_number}']['QF'] = QF
        metadict[f'{enc}_{case_number}']['distance_offset'] = distance_offset
        metadict[f'{enc}_{case_number}']['angdist_onsphr_offset'] = angdist_onsphr_offset
        metadict[f'{enc}_{case_number}']['time_offset'] = time_offset
        metadict[f'{enc}_{case_number}']['xy_cen'] = xy_cen
        metadict[f'{enc}_{case_number}']['B_ratio'] = B_ratio
        metadict[f'{enc}_{case_number}']['V_ratio'] = V_ratio
        metadict[f'{enc}_{case_number}']['beta_ratio'] = beta_ratio

        print(case_number)

    # saving the dictionary in a pickle file
    misc_FN.write_pickle(metadict, f'{enc}_metadict_Huang')