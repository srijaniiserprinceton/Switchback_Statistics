import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import datetime
from astropy.time import Time, TimeDelta
from sunpy.coordinates import HeliocentricInertial
from sunpy.coordinates import HeliographicCarrington
from sunpy.coordinates import spice
import astrospice
plt.ion()

import misc_functions as misc_fn

def get_PSP_orbit_coords(tstart, tend, tstep='half_day', frame='HeliographicCarrington'):
    kernels = astrospice.registry.get_kernels('psp', 'predict')
    psp_kernel = kernels[0]

    if(tstep=='half_day'):
        dt = TimeDelta(0.5 * u.day)

    # times = Time(np.arange(Time(tstart), Time(tend), dt))
    times = Time(np.linspace(Time(tstart), Time(tend), 100))
    coords = astrospice.generate_coords('SOLAR PROBE PLUS', times)
    if(frame=='HeliographicCarrington'):
        new_frame = HeliographicCarrington(observer='self')
    if(frame=='HeliocentricIntertial'):
        new_frame = HeliocentricInertial()
    coords = coords.transform_to(new_frame)

    return times, coords

def get_PSP_orbit_spice(dt_start, dt_end, cadence_days=1/24):
    return misc_fn.gen_dt_arr(dt_start, dt_end, cadence_days=cadence_days)
    # return misc_fn.gen_dt_arr(datetime.datetime(2018, 8, 15),
    #                           datetime.datetime(2025, 8, 31),
    #                           cadence_days=cadence_days)

def get_PSP_orbit_highres(dt_mission, dt_queried):
    '''
    Function to return interpolated SPC trajectory for the custom datetime
    array queried. The mission's spice kernels are used as input function
    for the interpolation.

    Paramters:
    ----------
    dt_mission : datetime array for the SPP mission returned from 
                 spice kernels with a cadence of 1/24 days
    dt_queried : datetime array queried

    Returns:
    --------
    trajectory_spc : interpolated trajectory of SPC onto dt_queried
    '''
    # finding default trajectoory from spice kernels for SPP
    parker_trajectory_inertial = spice.get_body('SPP', dt_mission)
    # transforming to heliographic carrington coordinates (co-rotating with solar equatorial mean velocity)
    parker_trajectory_carrington = parker_trajectory_inertial.transform_to(HeliographicCarrington(observer="self"))

    # interpolating onto required time grid
    trajectory_spc = misc_fn.interp_trajectory(dt_mission, parker_trajectory_carrington, dt_queried)

    return trajectory_spc


