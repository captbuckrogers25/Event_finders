'''
script to calculate desited parameters associated with tail current sheet flapping.

AJR -- 19.11.2020
'''
import time
import numpy as np
import pandas as pd
from mms_curvature.mms_curvature import mms_Grad, mms_Curvature, mms_CurlB
from mms_curvature.mms_load_data_shims import mms_load_fgm
from MVA import minvar
from dateutil.parser import parse
from datetime import timedelta, datetime, timezone

#########################################################
####### input parameters ################################
trange_file = 'flapping_2017_CS_crossings.old'

out_file = 'flapping_2017_stats_v4.csv'
agu_file = 'AGU_flapping_2017_stats_v4.csv'

# mu_0 with units correction for nT and km
ucorr = (1e-12)/(1.257e-6)

########################################################
########################################################



# Funtion for calculating all the desired parameters for each CS crossing

def flap_params(trange=['2017-06-08/19:33', '2017-06-08/19:37'], probe='1', data_rate='srvy'):

    # magnetic field gradient products calculation taken from 'Curvature_Runner_v5.py'

    # Get B-field data and calculate curvature
    numProbes = 4

    # Type check for trange
    tRangeDT = [None,None]
    if type(trange[0]) == datetime: # Already a datetime.
        tRangeDT = trange
    elif type(trange[0]) in (int,float,np.float32,np.float64): # Convert from posix timestamp if provided.
        tRangeDT = [datetime.fromtimestamp(trange[0], timezone.utc),datetime.fromtimestamp(trange[1], timezone.utc)]
    else: # Assuming a string-like.  Parse it and generate a datetime.
        tRangeDT = [parse(trange[0]),parse(trange[1])]
    
    # Expand time range initially loaded to ensure we get at least a pair of positions for interpolation.
    minute_delta = timedelta(seconds=60)
    tmpRange = [tRangeDT[0]-minute_delta, tRangeDT[1]+minute_delta]
    fgmdata = mms_load_fgm(trange=tmpRange, probe=['1', '2', '3', '4'], data_rate=data_rate, time_clip=True)[0]
    
    shapes = []
    # Trim the mag data to only the desired time range
    for bird in range(numProbes):
        slice1 = np.greater_equal(fgmdata['mms'+str(bird+1)+'_fgm_b_gsm_'+str(data_rate)+'_l2']['x'], tRangeDT[0].replace(tzinfo=timezone.utc).timestamp())
        slice2 = np.less_equal(fgmdata['mms'+str(bird+1)+'_fgm_b_gsm_'+str(data_rate)+'_l2']['x'], tRangeDT[1].replace(tzinfo=timezone.utc).timestamp())
        slice3 = np.logical_and(slice1, slice2)
        fgmdata['mms'+str(bird+1)+'_fgm_b_gsm_'+str(data_rate)+'_l2']['x'] = fgmdata['mms'+str(bird+1)+'_fgm_b_gsm_'+str(data_rate)+'_l2']['x'][slice3]
        fgmdata['mms'+str(bird+1)+'_fgm_b_gsm_'+str(data_rate)+'_l2']['y'] = fgmdata['mms'+str(bird+1)+'_fgm_b_gsm_'+str(data_rate)+'_l2']['y'][slice3]
        # This is just a collator for use in a dozen lines or so.
        shapes += [fgmdata['mms'+str(bird+1)+'_fgm_b_gsm_'+str(data_rate)+'_l2']['x'].shape[0] > 0]

    # calculate minimum variance analysis eigenvalues and eigenvectors
    eigvals, eigvecs = minvar(fgmdata['mms'+probe+'_fgm_b_gsm_'+data_rate+'_l2']['y'][:,0:3])

    pos_times = [None]*numProbes
    b_times = [None]*numProbes
    pos_values = [None]*numProbes
    b_values = [None]*numProbes

    #populate master arrays for reuse
    for bird in range(numProbes):
            pos_times[bird] = np.copy(fgmdata['mms'+str(bird+1)+'_fgm_r_gsm_'+str(data_rate)+'_l2']['x'])
            b_times[bird] = np.copy(fgmdata['mms'+str(bird+1)+'_fgm_b_gsm_'+str(data_rate)+'_l2']['x'])
            pos_values[bird] = np.copy(fgmdata['mms'+str(bird+1)+'_fgm_r_gsm_'+str(data_rate)+'_l2']['y'])
            b_values[bird] = np.copy(fgmdata['mms'+str(bird+1)+'_fgm_b_gsm_'+str(data_rate)+'_l2']['y'])


    if np.all(shapes):
        # We have data from all birds
        # Calculate nominal values for grad(B) products; no uncertainty
        grad_0n, bm_0, Bmag_0, rm_0, t_master = mms_Grad(postimes=pos_times, posvalues=pos_values, magtimes=b_times, magvalues=b_values, normalize=True)
        grad_0f, Bm_0 = mms_Grad(postimes=pos_times, posvalues=pos_values, magtimes=b_times, magvalues=b_values, normalize=False)[:2]
        
        kvec = mms_Curvature(grad_0n, bm_0)  # uses gradiant of normalized magnetic field
        
        jvec = mms_CurlB(grad_0f) * ucorr             # Curl uses gradiant of NOT-normalized magnetic field

        jmag = np.linalg.norm(jvec, axis=1)
        kmag = np.linalg.norm(kvec, axis=1)
    
        jmaxi = np.nanargmax(jmag)
        kmaxi = np.nanargmax(kmag)
    else:
        # We're missing data from at least one bird, so mms_Grad would fail.
        # We will instead just insert a minimal set of data from the requested probe
        #  and return lots of NaN.
        t_master = fgmdata['mms'+probe+'_fgm_b_gsm_'+data_rate+'_l2']['x']
        rm_0 = np.ndarray((t_master.shape[0],3))
        for dim in range(3): # Currently, everything measures in 3 spacial dimensions
            rm_0[:,dim] = np.interp(t_master, fgmdata['mms'+probe+'_fgm_r_gsm_'+data_rate+'_l2']['x'], fgmdata['mms'+probe+'_fgm_r_gsm_'+data_rate+'_l2']['y'][:,dim])

        kvec = np.full((t_master.shape[0],3), np.nan)
        jvec = np.full((t_master.shape[0],3), np.nan)

        jmag = np.linalg.norm(jvec, axis=1)
        kmag = np.linalg.norm(kvec, axis=1)
    
        jmaxi = 0
        kmaxi = 0

    jmaxtime = t_master[jmaxi]
    kmaxtime = t_master[kmaxi]

    pos = rm_0[jmaxi]

    kvecmax = kvec[kmaxi]
    jvecmax = jvec[jmaxi]


    outlist = [jmaxtime, jmag[jmaxi], jvecmax[0], jvecmax[1], jvecmax[2], kmaxtime, kmag[kmaxi], kvecmax[0], kvecmax[1], kvecmax[2], pos[0], pos[1], pos[2], eigvals[0], eigvecs[0][0], eigvecs[0][1], eigvecs[0][2], eigvals[1], eigvecs[1][0], eigvecs[1][1], eigvecs[1][2], eigvals[2], eigvecs[2][0], eigvecs[2][1], eigvecs[2][2]]

    #############################
    ######  AGU stats ###########
    #############################

    rval = eigvals[1]/eigvals[2]
    if (rval > 10) and np.all(shapes):
        Nvec = np.divide(np.cross(bm_0, kvec), np.linalg.norm(np.cross(bm_0, kvec), axis=1).reshape(bm_0.shape[0],1))
        gamma_N_deg = np.arccos(np.einsum('...i,...i', np.divide(jvec, jmag.reshape(jmag.shape[0], 1)), Nvec)) * (180/np.pi)
        gamma = gamma_N_deg[(kmaxi + jmaxi)//2]

        n_ang = np.arccos(eigvecs[2][2]) * (180/np.pi)
        ang_ratio = n_ang/gamma

        kl = np.einsum('...i,...i', kvecmax, eigvecs[0])
        km = np.einsum('...i,...i', kvecmax, eigvecs[1])
        kn = np.einsum('...i,...i', kvecmax, eigvecs[2])

        jl = np.einsum('...i,...i', jvecmax, eigvecs[0])
        jm = np.einsum('...i,...i', jvecmax, eigvecs[1])
        jn = np.einsum('...i,...i', jvecmax, eigvecs[2])
    else:
        gamma = n_ang = ang_ratio = kl = km = kn = jl = jm = jn = np.nan

    statslist = [rval, n_ang, gamma, ang_ratio, kl, km, kn, jl, jm, jn, kmag[kmaxi]]




    return outlist, statslist

# end flap_params ###################################################



# column names for end dataframe
columnnames = ['jtime', 'jmax', 'jx', 'jy', 'jz', 'ktime', 'kmax', 'kx', 'ky', 'kz', 'rx', 'ry', 'rz', 'l0', 'l0x', 'l0y', 'l0z', 'l1', 'l1x', 'l1y', 'l1z', 'l2', 'l2x', 'l2y', 'l2z']

agucolumns = ['rval', 'n_ang', 'gamma_N', 'ang_ratio', 'kl', 'km', 'kn', 'jl', 'jm', 'jn', 'kmax']

# read in CSV file with tranges
crossings = pd.read_csv(trange_file, header=0, index_col=0)

# initialize a list to hold the calculated parameters
data=[]
agustats=[]

# run loop to calculate parameters for each crossing in the input file
for i in range(len(crossings.index.values)):
    print(crossings.index.values[i])
    outlst, agulst = flap_params(trange=[crossings['tstart'].values[i], crossings['tstop'].values[i]])
    data.append(outlst)
    agustats.append(agulst)

# generate output file
paramsDF = pd.DataFrame(data, columns=columnnames, index=crossings.index.values)
paramsDF.to_csv(out_file)

aguDF = pd.DataFrame(agustats, columns=agucolumns, index=crossings.index.values)
aguDF.to_csv(agu_file)









