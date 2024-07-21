import numpy as np
import pandas as pd
from mms_curvature import mms_load_fgm, mms_load_mec
import matplotlib.pyplot as plt
from flap_fit_v0 import *

#### general run parameters ###############

prefix = "./fits/autoSinFit_GSM_"
suffix = "_v0"
probe= '1'
eventfile = './fits/clean_CS_flap_2017_v1.csv'
coords = 'gsm'

###########################################

# Load in csv of flapping event time ranges
eventdf = pd.read_csv(eventfile, index_col=0)

# initialize lists to temporarily hold fit parameters and event locations
outlst = []
rlst = []

print('Calculating fits and generating plots...')
# for-loop to calculate the fit parameters for each event and generate a figure of each
for idx in range(eventdf.index.values.shape[0]):
    event = idx + 1
    print('Event:  '+str(event))
    trange = [eventdf['Tstart'].values[idx], eventdf['Tend'].values[idx]]
    filename = prefix+trange[0][:10]+"_"+trange[0][11:13]+trange[0][14:16]+"--"+trange[1][11:13]+trange[1][14:16]+suffix
    # Load the magnetic field data
    bdata = mms_load_fgm(trange=trange, probe=probe, time_clip=True)[0]['mms'+probe+'_fgm_b_'+coords+'_srvy_l2']
    # Load positional data and append MLT and vector data to rlst
    mecdata = mms_load_mec(trange=trange, probe=probe, time_clip=True)[0]
    rlst.append((mecdata['mms'+probe+'_mec_r_'+coords]['y'][0,:], mecdata['mms'+probe+'_mec_mlt']['y'][0]))
    # Calculate the sinusoidal fit for the timerage listed for the event and append the full output to the outlst
    outlst.append(FitSine(bdata['x'], bdata['y'])[0])
    params = outlst[-1][1]
    sigma = outlst[-1][2]
    # Generate a figure of the current event with fit and parameters
    plt.close()
    plt.xlabel("UT(s)")
    plt.ylabel("Bx(nT)")
    plt.title("Sinusoidal fit\n"+trange[0]+"--"+trange[1][11:16])
    plt.plot(bdata['x'], bdata['y'][:,0], label='Bx')
    plt.plot(bdata['x'], sinFunc(bdata['x'], *params), color='r', label='Sin fit')
    plt.plot(bdata['x'], np.zeros(bdata['x'].shape[0]), color='k')
    txtA = "amp: "+np.format_float_scientific(params[0], precision=5, exp_digits=1)+" +/- "+np.format_float_scientific(sigma[0], precision=5, exp_digits=1)
    txtT = "\nT: "+np.format_float_scientific(1/params[1], precision=5, exp_digits=1)+" +/- "+np.format_float_scientific(1/sigma[1], precision=5, exp_digits=1)
    txtO = "\nphase: "+np.format_float_scientific(params[2], precision=5, exp_digits=1)+"  offset: "+np.format_float_scientific(params[3], precision=5, exp_digits=1)
    plt.figtext(0.45, 0.12, txtA+txtT+txtO, wrap=True, horizontalalignment='center', fontsize=10)
    plt.savefig(filename+".png")

# Initialize lists for all the fit parameters to save to the event CSV
# Amplitude
eventdf['gA'] = [i[0][0] for i in outlst]  # guessed amplitude
eventdf['fA'] = [i[1][0] for i in outlst] # fitted amplitude
eventdf['sA'] = [i[2][0] for i in outlst] # sigma of fitted amplitude

# frequency
eventdf['gF'] = [i[0][1] for i in outlst] # guessed frequency
eventdf['fF'] = [i[1][1] for i in outlst] # fitted frequency
eventdf['sF'] = [i[2][1] for i in outlst] # sigma of fitted frequency

# phase
eventdf['gP'] = [i[0][2] for i in outlst] # guessed phase
eventdf['fP'] = [i[1][2] for i in outlst] # fitted phase
eventdf['sP'] = [i[2][2] for i in outlst] # sigma of fitted phase

# offset
eventdf['gO'] = [i[0][3] for i in outlst] # guessed offset
eventdf['fO'] = [i[1][3] for i in outlst] # fitted offset
eventdf['sO'] = [i[2][3] for i in outlst] # sigma of fitted offset

# position 
eventdf['Rx'] = [i[0][0] for i in rlst]
eventdf['Ry'] = [i[0][1] for i in rlst]
eventdf['Rz'] = [i[0][2] for i in rlst]

eventdf['Rmlt'] = [i[1] for i in rlst]

# Store the new calculated event list
print("\n\nSaving "+prefix+"2017_params"+suffix+".csv")
eventdf.to_csv(prefix+"2017_params"+suffix+".csv")

# end
