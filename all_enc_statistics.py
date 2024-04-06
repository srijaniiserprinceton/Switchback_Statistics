import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import misc_functions as misc_FN

def all_hist_SB_orientation(enc_arr, catalog='Vamsee'):
    plt.figure(figsize=(8,6))

    for enc in enc_arr:
        metadict = misc_FN.read_pickle(f'{enc}_metadict_Vamsee')
        angdist_onsphr_offset = []
        xy_cen = []

        for key in metadict.keys():
            angdist_onsphr_offset.append(metadict[f'{key}']['angdist_onsphr_offset'].value)
            xy_cen.append(metadict[f'{key}']['xy_cen'])
        
        angdist_onsphr_offset = np.asarray(angdist_onsphr_offset)
        xy_cen = np.asarray(xy_cen)

        # constructing the array of angular distances of z-angles
        z_offset = []

        for i in range(1, len(xy_cen)):
            phi1, theta1 = xy_cen[i,1] * np.pi / 180
            phi2, theta2 = xy_cen[i-1,1] * np.pi / 180
            z_offset.append(np.arccos(np.sin(theta1) * np.sin(theta2) + np.cos(theta1) * np.cos(theta2) * np.cos(phi1 - phi2)))

        # in degrees
        z_offset = np.asarray(z_offset) * 180 / np.pi

        # histogram of deflection change when PSP moves by less than a degree
        mask_PSP_movement = angdist_onsphr_offset[1:] < 60
        z_offset_masked = z_offset[mask_PSP_movement]

        if(enc < 10):
            enc = f'0{enc}'

        plt.hist(z_offset_masked, bins=20, histtype='step', density=True, label=f'E{enc}')

    plt.legend()
    plt.xlabel('Subsequent SB deflection offset [deg]')
    plt.ylabel('Normalized switchback count')
    plt.tight_layout()
    plt.savefig(f'Vamsee_zangle_hist1d_all_enc.pdf')  

if __name__=='__main__':
    enc_arr = np.array([1, 2, 4, 5, 6, 7, 8, 9, 10])
    all_hist_SB_orientation(enc_arr)