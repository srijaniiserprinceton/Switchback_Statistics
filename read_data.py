import cdflib
import pandas
data = pandas.read_hdf("ApJ_SB_Database.h5")
data.shape == (5370, 7) # true  
events = data.groupby(["event number","encounter"])  # 1,074 events

def get_params(tstart, tend, cdf_file):
    cdf_data = cdflib.CDF(cdf_file)

    # reading the timestamps array
    time = cdf_data.varget()

    # reading the magnetic field products in RTN format
    BR, BT, BN = cdf_data.varget(psp_fld_l2_mag_RTN)

    
