import sys

import Huang_workflow as Huang
import Vamsee_workflow as Vamsee
import Larosa_workflow as Larosa

import astrospice_demo

if __name__=='__main__':
    # selecting encounter to analyze. E1,2,4-8 are available
    enc = int(sys.argv[1])
    catalog = sys.argv[2]
    
    #-------selecting the input file according to the requested catalog-----#
    if(catalog == 'Huang'):
        Huang.get_statistics(enc)

    if(catalog == 'Vamsee'):
        Vamsee.get_statistics(enc)

    if(catalog == 'Larosa'):
        if(enc > 1): print('Larosa catalog only upto E01.')
        Larosa.get_statistics()