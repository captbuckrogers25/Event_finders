# Script to identify candidate flapping current sheets using MMS data

import numpy as np
import pandas as pd
from mms_curvature.mms_load_data_shims import mms_load_fgm, mms_load_fpi
from signalsmoothing import smooth
import datetime as dt
# setup data 
########################################
#
trange = ['2017-06-01', '2017-06-03']   # Time range to search in
probe = '1'                             # Which MMS probe to use the data from
prefix = 'FCS_'                         # Text in filename before time range 
suffix = '_v1'                          # Text in filename after time range
save_csv = False                         # if output should be saved as a CSV file
save_h5 = False                         # if output should be saved as a HDF5 file
#
min_dbdt = 0.3                          # minimum value for |dBx/dt| to count as CS crossing (may need some experimentation)
min_bx = 0.25                            # maximum magnitude of Bx at CS crossing (also needs experimentation)
min_vz = 25.0                           # minimum ion velocity in the CS normal direction; proxy for current
#
trimout = True                          # output a trimmed output file with only possible CS crossings
########################################

# Generate sane filename
if len(trange[0]) > 10 or len(trange[1]) > 10:
        filename = prefix+trange[0][:10]+"_"+trange[0][11:13]+trange[0][14:16]+"--"+trange[1][:10]+"_"+trange[1][11:13]+trange[1][14:16]+suffix+".csv"
else:
        filename = prefix+trange[0][:10]+"--"+trange[1][:10]+suffix+".csv"

# Load the magnetometer and particle instrument data
fgmdata = mms_load_fgm(trange=trange, probe=probe)[0]
fpidata = mms_load_fpi(trange=trange, probe=probe)[0]

# Generate a 1s rolling average of the X_GSE component of the magnetometer data
bxdata_1s = smooth(fgmdata['mms'+probe+'_fgm_b_gse_srvy_l2']['y'][:,0], window_len=16, window='flat')[1]

# Generate a discrete time derivative of Bx (dBx = Bx[i] - Bx[i-1])
#(NOTE: survey level FGM data cadance is 16 samples/s so dt=(1/16) --> dB/dt = 16 * dB)
dBxdt = 16. * np.subtract(np.concatenate((bxdata_1s, np.array([bxdata_1s[-1]]))), np.concatenate((np.array([bxdata_1s[0]]), bxdata_1s)))[:-1]

# interpolate the ion velocity data to FGM cadance
vzdata = np.interp(fgmdata['mms'+probe+'_fgm_b_gse_srvy_l2']['x'], fpidata['mms'+probe+'_dis_bulkv_gse_fast']['x'], fpidata['mms'+probe+'_dis_bulkv_gse_fast']['y'][:,2])

# Require CS crossing to have |dBx/dt| > 1 nT/s
CS_crossing = np.logical_and(np.abs(dBxdt) > min_dbdt, np.abs(bxdata_1s) < min_bx)

# For flapping CS candidate, add requirement that 
FCS = np.logical_and(CS_crossing, vzdata > min_vz)

# Generate Pandas dataframe for easy exporting
outputdf = pd.DataFrame({'Bx_1s':bxdata_1s, 'dBx/dt':dBxdt, 'CS_crossing':CS_crossing, 'CS_wVz':FCS}, index=fgmdata['mms'+probe+'_fgm_b_gse_srvy_l2']['x'])
outputdf.index.name='Time'

# save output file
if save_csv: outputdf.to_csv(filename)
if save_h5: ouputdf.to_hdf(filename[:-3]+'h5', key='df')



# Select only FGM-cadance timesteps 
strtime = np.datetime_as_string(((fgmdata['mms'+probe+'_fgm_b_gse_srvy_l2']['x'] * 1e6).astype(int)).astype('datetime64[us]'))

# Output the 'Trimmed' data with human-readable dates.
if trimout: pd.DataFrame({'Date':strtime[CS_crossing], 'Bx_1s':bxdata_1s[CS_crossing], 'dB/dt':dBxdt[CS_crossing], 'FCS':FCS[CS_crossing]}, index=fgmdata['mms'+probe+'_fgm_b_gse_srvy_l2']['x'][CS_crossing]).to_csv('Trimmed_'+filename)


#end
