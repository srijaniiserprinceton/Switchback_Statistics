import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.visualization import quantity_support
from sunpy.coordinates import HeliocentricInertial
from sunpy.coordinates import HeliographicCarrington
import astrospice
plt.ion()

kernels = astrospice.registry.get_kernels('psp', 'predict')
psp_kernel = kernels[0]
coverage = psp_kernel.coverage('SOLAR PROBE PLUS')
print(coverage.iso)

dt = TimeDelta(0.5 * u.day)
# times = Time(np.arange(Time('2018-11-01'), coverage[1], dt))
times = Time(np.arange(Time('2021-06-01'), Time('2022-06-30'), dt))
coords = astrospice.generate_coords('SOLAR PROBE PLUS', times)
print(coords[0:4])

new_frame1 = HeliocentricInertial()
new_frame2 = HeliographicCarrington(observer='self')
coords1 = coords.transform_to(new_frame1)
coords2 = coords.transform_to(new_frame2)
# print(coords[0:4])

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.scatter(coords1.lon.to(u.rad), coords1.distance.to(u.au), c=times.jd, s=2)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
# ax.scatter(coords2.lon.to(u.rad), coords2.radius.to(u.au), c=times.jd, s=2)
ax.plot(coords2.lon.to(u.rad), coords2.radius.to(u.au), 'k')
plt.show()

