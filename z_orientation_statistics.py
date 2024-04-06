import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import misc_functions as misc_FN

# loading the metadict for the desired encounter
enc = 10 #'E02'

metadict = misc_FN.read_pickle(f'{enc}_metadict_Vamsee')

# reading and storing the elements
onset_time = []
duration = []
dist_from_Sun = []
QF = []
distance_offset = []
angdist_onsphr_offset = []
time_offset = []
xy_cen = []

for key in metadict.keys():
    onset_time.append(metadict[f'{key}']['onset_time'])
    duration.append(metadict[f'{key}']['duration'])
    dist_from_Sun.append(metadict[f'{key}']['dist_from_Sun'])
    # QF.append(metadict[f'{key}']['QF'])
    distance_offset.append(metadict[f'{key}']['distance_offset'].value)
    angdist_onsphr_offset.append(metadict[f'{key}']['angdist_onsphr_offset'].value)
    time_offset.append(metadict[f'{key}']['time_offset'])
    xy_cen.append(metadict[f'{key}']['xy_cen'])

# converting them to arrays from lists
onset_time = np.asarray(onset_time)
duration = np.asarray(duration)
dist_from_Sun = np.asarray(dist_from_Sun)
# QF = np.asarray(QF)
distance_offset = np.asarray(distance_offset)
angdist_onsphr_offset = np.asarray(angdist_onsphr_offset)
time_offset = np.asarray(time_offset)
xy_cen = np.asarray(xy_cen)

# constructing the array of angular distances of z-angles
z_offset = []

for i in range(1, len(xy_cen)):
    phi1, theta1 = xy_cen[i,1] * np.pi / 180
    phi2, theta2 = xy_cen[i-1,1] * np.pi / 180
    z_offset.append(np.arccos(np.sin(theta1) * np.sin(theta2) + np.cos(theta1) * np.cos(theta2) * np.cos(phi1 - phi2)))

# in degrees
z_offset = np.asarray(z_offset) * 180 / np.pi

# making the necessary plots
plt.figure()
plt.scatter(z_offset, angdist_onsphr_offset[1:], c=np.log10(time_offset[1:]))
plt.xlabel('z offset in degrees')
plt.ylabel('Angular distance of PSP in minutes')


plt.figure()
plt.hist2d(z_offset, angdist_onsphr_offset[1:], range=[[0,180],[0,180]], bins=[20,20])

# histogram of deflection change when PSP moves by less than a degree
mask_PSP_movement = angdist_onsphr_offset[1:] < 60
z_offset_masked = z_offset[mask_PSP_movement]

plt.figure()
plt.hist(z_offset_masked, bins=20)
plt.xlabel('Subsequent SB deflection offset [deg]')
plt.savefig(f'Vamsee_zangle_hist1d_{enc}.pdf')