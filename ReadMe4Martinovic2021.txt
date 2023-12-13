================================================================================
Title: Multiscale Solar Wind Turbulence Properties inside and near Switchbacks 
       measured by Parker Solar Probe 
Authors: Martinovic M.M., Klein K.G., Huang J., Chandran B.D.G., Kasper J.C., 
    Lichko E., Bowen T., Chen C.H.K., Matteini L., Stevens M., Case A.W., 
    Bale S.D. 
================================================================================
Description of contents: Switchback (SB), spike, or jet is defined as a
    structure in which the solar wind plasma flow changes direction with
    respect to the background solar wind. It is observed by the spacecraft
    as a magnetic field rotation, persisting for a brief period of time.

    This database contains the list of 1,074 visaully selected events from
    Parker Solar Probe Encounters 1 and 2. Each event is contained of five
    sampled regions: 

    1) Leading Quiet Region (LQR) with stable velocities and magnetic fields 
        before the SB;
    2) Leading Transition Region (LTR), where the magnetic field rotates from 
        LQR toward its SB orientation;
    3) the SB itself with stable field orientation;
    4) Trailing Transition Region (TTR); and
    5) Trailing Quiet Region (TQR), with conditions which are, in general, 
        not very different from the ones in LQR.

    There are seven data fields for each event and sampled region
        1) "encounter":     integer encounter number (1 or 2)
        2) "event number":  integer, number of an event within the given encounter
        3) "QF":            integer, rate of an event clarity (0-5). 
                            Events with QF < 2 are considered unreliable.
        4) "region":        what region is sampled. Can take string values:
                                "lead quiet" - LQR
                                "lead trans" - LTR
                                "SB"         - SB
                                "post trans" - TTR
                                "post quiet" - TQR
        5) "start time":    timestamp of the region sampling start time
        6) "end time":      timestamp of the region sampling end time
        7) "duration [s]":  integer, time interval between start time and and 
                            time, for  each region

System requirements: The database was loaded and verified using pandas v1.0.5. 
    A short access script is given below:
    
    import pandas
    data = pandas.read_hdf("ApJ_SB_Database.h5")
    data.shape == (5370, 7) # true  
    events = data.groupby(["event number","encounter"])  # 1,074 events

Additional comments: 

================================================================================
