import pyspedas
import pytplot
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()

def compare_fc_density(tstart, tend, i):
    spc_vars = pyspedas.psp.spc(trange=[tstart, tend], varnames=['np_fit', 'DQF'],
                                datatype='l3i', level='l3', time_clip=True, no_update=False, get_support_data=True)
    
    fc_flag = pytplot.data_quants['psp_spc_DQF'].data[:,16]
    N = pytplot.data_quants['psp_spc_np_fit'].data
    time = pytplot.data_quants['psp_spc_np_fit'].time.data
    N_fc = N[fc_flag]
    time_fc = time[fc_flag] 

    plt.figure()
    plt.plot(time, N, '.k', alpha=0.5)
    plt.plot(time_fc, N_fc, 'r')
    plt.tight_layout()
    plt.savefig(f'compare_fc_plots/compare_fc_{i}.png')

if __name__=='__main__':
    #-------selecting the time window-----#
    data = np.load('data_products/Larosa_Catalog/dates_lead_trail.npz', allow_pickle=True)

    # datetime array of SB events
    dt_SB_lead = data['date_lead_vec'][0,:]
    dt_SB_start = data['date_lead_vec'][1,:]
    dt_SB_end = data['date_trail_vec'][0,:]
    dt_SB_trail = data['date_trail_vec'][1,:]

    # total number of SB events in encounter 1
    Nevents = len(dt_SB_start)

    # arranging by time
    sort_idx = np.argsort(dt_SB_start)

    dt_SB_lead = dt_SB_lead[sort_idx]
    dt_SB_start = dt_SB_start[sort_idx]
    dt_SB_end = dt_SB_end[sort_idx]
    dt_SB_trail = dt_SB_trail[sort_idx]

    for i in tqdm(range(Nevents)):
        T1, T2 = dt_SB_lead[i], dt_SB_trail[i]
        T1_formatted = f'{T1.year}-{T1.month}-{T1.day}/{T1.hour}:{T1.minute}:{T1.second}'
        T2_formatted = f'{T2.year}-{T2.month}-{T2.day}/{T2.hour}:{T2.minute}:{T2.second}'
        compare_fc_density(T1_formatted, T2_formatted, i)