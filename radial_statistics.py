import pandas
import numpy as np
import astropy.units as u
from astropy.time import Time
from sunpy.coordinates import HeliocentricInertial
import astrospice
import matplotlib.pyplot as plt
plt.ion()

def time2radialdistances(data, min_QF=5):
    # filtering the events that meet the quality factor
    mask_by_QF = np.where(data['QF'] >= min_QF)

    # total number of events that meet the quality critera
    valid_event_numbers = np.unique(data['event number'].array[mask_by_QF])
    Nevents = len(valid_event_numbers)

    # the list to store the radial distances
    r_in_AU = []

    # the number of problematic indices
    nfaulty = 0

    for i in range(Nevents):
        # finding the index for the SB start
        SB_idx = np.where(data['event number'] == valid_event_numbers[i])[0][2]
        # finding the corresponding time from astrospice
        try:
            time = Time([data['start time'][SB_idx]])
        except:
            nfaulty += 1 
            print('Faulty index: ', SB_idx)
            print(f'Ignored {nfaulty} points uptil now.')
            continue

        # coordinates in HeliocentricIntertial
        coords = astrospice.generate_coords('SOLAR PROBE PLUS', time)
        new_frame = HeliocentricInertial()
        coords = coords.transform_to(new_frame)
        # converting to AU
        r_in_AU.append(coords.distance.to(u.au).value[0])

    # returning the radial distance in AU
    return np.asarray(r_in_AU)


if __name__=='__main__':
    # loading the data for the marked switchback encounters
    data = pandas.read_hdf("ApJ_SB_Database.h5")
    data.shape == (5370, 7) # true  
    events = data.groupby(["event number","encounter"])  # 1,074 events

    plt.figure()

    # looping over the min_QF
    for min_QF in range(1, 6):
        print(f'min_QF = {min_QF}')
        # array of radial distances in AU for each SB above quality factor min_QF
        r_in_AU = time2radialdistances(data, min_QF=min_QF)

        # plotting the the SB counts as a function of distance
        plt.hist(r_in_AU, alpha=0.3, density=True, label=f'min_QF={min_QF}')
    
    plt.xlabel('Radial distance [A.U.]')
    plt.ylabel('Normalized counts')
    plt.legend()
    plt.tight_layout()