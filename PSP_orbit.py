import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta
from sunpy.coordinates import HeliocentricInertial
import astrospice
plt.ion()

def get_PSP_orbit_coords(tstart, tend, tstep='half_day'):
    kernels = astrospice.registry.get_kernels('psp', 'predict')
    psp_kernel = kernels[0]

    if(tstep=='half_day'):
        dt = TimeDelta(0.5 * u.day)

    # times = Time(np.arange(Time(tstart), Time(tend), dt))
    times = Time(np.linspace(Time(tstart), Time(tend), 100))
    coords = astrospice.generate_coords('SOLAR PROBE PLUS', times)
    new_frame = HeliocentricInertial()
    coords = coords.transform_to(new_frame)

    return times, coords


