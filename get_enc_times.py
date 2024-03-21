import numpy as np
import re
import pandas as pd
import datetime

def get_PSP_enc():
    enc_file = 'Enc_times.txt'

    enc_data = pd.read_csv(enc_file, sep=" ")

    enc_data = np.asarray(enc_data)

    enc_start_times = []
    enc_end_times = []
    enc_num = []

    for i in range(len(enc_data)):
        start_time_str, end_time_str, enc_str = re.split('[\t]', enc_data[i][0])
        enc_start_times.append(datetime.datetime(*np.asarray(re.split('[-,T,:,Z]', start_time_str)[:-1]).astype('int')))
        enc_end_times.append(datetime.datetime(*np.asarray(re.split('[-,T,:,Z]', end_time_str)[:-1]).astype('int')))
        enc_num.append(int(enc_str))

    return np.asarray(enc_start_times), np.asarray(enc_end_times), np.asarray(enc_num)
        