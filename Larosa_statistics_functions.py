import numpy as np

def get_statistic_for_event(arr, time_arr, T2, T3, T4, T5):

    T2 = np.datetime64(T2)
    T3 = np.datetime64(T3)
    T4 = np.datetime64(T4)
    T5 = np.datetime64(T5)

    before_mask = (time_arr >= T2) * (time_arr <= T3)
    SB_mask = (time_arr >= T3) *  (time_arr <= T4)
    after_mask = (time_arr >= T4) * (time_arr <= T5)

    val_SB = np.mean(arr[SB_mask])
    val_before = np.mean(arr[before_mask])
    val_after = np.mean(arr[after_mask])

    return val_SB / (0.5 * (val_before + val_after))