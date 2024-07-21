
import numpy as np
#from scipy.optimize import least_squares
from scipy.optimize import curve_fit
#import matplotlib.pyplot as plt

# our paramaterized function to try fitting
def sinFunc(time,amp,freq,phase,offset):
    return amp*np.sin((freq*2*np.pi)*(time - phase)) + offset

# least_squares requires a root-finding function:
# 0 = amp*np.sin(freq*time + phase) + offset - y
# x = [amplitude, frequency, phase, offset]
def sinFuncLS(x, time, y):
    #return x[0]*np.sin(x[1]*time + x[2]) + x[3] - y
    return sinFunc(time, *x) - y

'''
This function is only lightly modified from the scipy cookbook recipe found here:
  https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
This version is limited to the 'flat' smoothing style to avoid complexity.
'''
def localSmooth(x, window=7):
    if x.size < window or window < 3:
        return x
    s=np.r_[x[window-1:0:-1],x,x[-2:-window-1:-1]]
    w=np.ones(window,'d')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

'''
The following is derived from an answer to a StackOverflow question,
 found here:  https://stackoverflow.com/a/4625132
Explaination of the format in the following lines:
Each segment in the 'and' (&) construct is a single-direction comparison
 pass of each value relative to its nth neighbor.
This means that by performing the comparison pass once in each direction,
 we get the relative valuation from bi-directionally nearby sample values.

np.r_[[True]*n, inputArray[n:] <comparison_op> inputArray[:-n]]
      -- This is the forward comparison, which compares each value
         with its nth forward neighbor.

np.r_[inputArray[:-n] <comparison_op> inputArray[n:], [True]*n]
      -- This is the backward comparison, which compares each value
         with its nth previous neighbor.
'''
def localCompare(inputArray, radius=1, op=np.less):
    output = np.full_like(inputArray, True, dtype=np.bool)
    for n in range(1,radius+1):
        output &= np.r_[[True]*n, op(inputArray[n:], inputArray[:-n])] & np.r_[op(inputArray[:-n], inputArray[n:]), [True]*n]
    return output

'''
Assumes the following regarding parameters:
	timesteps - array of timesteps, shape (steps, )
	data - array of data, shape (steps, data_axis)
This code currently operates only on the first data axis [:,0].  This will get paramaterized later.
This function will return a list of tuples, where:
  output[data_axis][0] == the calculated guess parameters
  output[data_axis][1] == the array returned from least_squares()
  output[data_axis][2] == the function used with least_squares() (intended for plotting use)
'''
def FitSine(timesteps, data, frequencyResolutionHz=0.0001):
    timeStepSize = np.mean(timesteps[1:] - timesteps[:-1]) # average timestep duration
    output = []
    for dim in range(data.shape[-1]):
        g_amp = g_freq = g_phase = g_offset = 0 # initialize the guesses for this fit
        # Determine sample padding size to reach desired frequency resolution of FFT
        steps = data.shape[0]
        freqs = np.fft.rfftfreq(steps, d=timeStepSize)[1:]
        scale = (freqs[1]-freqs[0]) / frequencyResolutionHz
        if scale > 1:
            scale = np.ceil(np.log2(scale))
            steps = int(steps * np.power(2,scale))
            freqs = np.fft.rfftfreq(int(steps), d=timeStepSize)[1:]
        
        # Run FFT to locate dominant frequency
        # Notes:
        #  - We drop index 0 because this is the (nominal) mean signal across the
        #    sample.  As we're padding the sample, we calculate this elsewhere.
        #    This is also why we dropped index 0 from the frequency list above.
        dimFFT = np.fft.rfft(data[:,dim], n=steps)[1:]
        # This calculates the actual "amplitude" of the sample at each frequency bin,
        #  defined by |sqrt(freq_real^2 + freq_imag^2)|, then returns the frequency
        #  bin displaying greatest amplitude.
        FFTmag = np.abs(np.sqrt(np.power(dimFFT.real,2)+np.power(dimFFT.imag,2)))
        FFTsort = np.flip(np.argsort(FFTmag))
        fBin = FFTsort[0]
        # Dominant frequency is frequency guess for later fit
        g_freq = freqs[fBin]  
        #
        # Approximate guess offset (DC component of sine wave) = mean(max,min)
        g_offset = np.mean([np.max(data[:,dim]),np.min(data[:,dim])])
        #
        # Approximate amplitude from offset = mean(abs(max),abs(min))
        g_amp = np.mean([np.abs(np.max(data[:,dim])),np.abs(np.min(data[:,dim]))])
        #
        # The phase shift guess is where things get dicey.  The below is all 
        # Tim Rogers
        #
        ######################################################################
        # Approximate guess for phase shift:
        #  - Roughly center the wave by reversing the guessed offset, then smooth.
        #  - Find a local max(s) and/or min(s) in seriously smoothed sample
        #  - Find likely root(s)
        #  - Locate a partial sine wave and base the phase shift on a root location
        clip = int(np.ceil(data.shape[0]/100))
        dimTemp = localSmooth(data[:,dim] - g_offset, window=(clip*2)+1)[clip:-clip]
        # 
        # Root-finding:
        #  1.  find semi-local minima across the waveform, limit possible
        #  2.  limit potential roots to those close enough to 0
        #
        # Defining "close enough" to a root as below 5% of max amplitude.
        close_enough_root = .05 * g_amp
        # 
        # Defining "close enough" to a peak/valley as beyond 90% of the guessed amplitude
        close_enough_pv = .9 * g_amp
        # 
        # Locate likely roots
        roots = localCompare(np.abs(dimTemp), radius=3, op=np.less)
        roots = np.argwhere(roots & np.less(np.abs(dimTemp),close_enough_root)).flatten()
        # Locate likely peaks
        peaks = localCompare(dimTemp, radius=3, op=np.greater)
        peaks = np.argwhere(peaks & np.greater(dimTemp,close_enough_pv)).flatten()
        # Locate likely valleys
        valleys = localCompare(dimTemp, radius=3, op=np.less)
        valleys = np.argwhere(valleys & np.less(dimTemp,-close_enough_pv)).flatten()
        #
        # Naive wave locator:
        #  if multiple roots, cycle through root pairs, find peaks or valleys that fall between them.
        #  if only 1 root, try finding a pair of peaks or valley that have the root between them
        #
        half_wave = 1 / (2*g_freq)
        half_wave_in_steps = half_wave / timeStepSize
        g_phase = 0
        if roots.shape[0]>1:
            for idx in range(roots.shape[0]-1):
                # make sure this pair of roots is at least a quarter-wave apart
                if (roots[idx+1]-roots[idx]) < (half_wave_in_steps/2):
                    continue # Less than half the expected spacing.  Skip to next pair.
                match_peaks = np.logical_and(peaks > roots[idx], peaks < roots[idx+1])
                match_valleys = np.logical_and(valleys > roots[idx], valleys < roots[idx+1])
                if np.any(match_peaks):
                    # Found a peak between roots
                    # Naive approach:
                    #  No consideration is made about *where*
                    #  between the roots the peak is.
                    g_phase = timesteps[roots[idx]]
                    break
                if np.any(match_valleys):
                    # Found a valley between roots
                    # Naive approach:
                    #  No consideration is made about *where*
                    #  between the roots the peak is.
                    g_phase = timesteps[roots[idx]] + half_wave
                    break
        # Only case where we should have only a single root is if we have
        #  just from a peak to a valley (or vice versa).  Determine direction.
        elif roots.shape[0] == 1:
            # No matter what else we find, we've only got one root to source from.
            # Initialize the phase to start upswing at the root.
            g_phase = timesteps[roots[0]]
            # If we've only got one root, we should only have peaks on one side
            #  of the root.
            if np.any(peaks < roots[0]):
                # Peaks behind the root.  Shift phase to reverse waveform estimate.
                g_phase += half_wave
            # Sanity check.  If above was not true, we'll only check valleys if
            #  there are also no peaks forward of the root.
            # If the following condition fails, we use the already assigned phase.
            elif np.all(np.logical_not(peaks > roots[0])):
                # Check for valleys behind the root. No-op if true.
                if np.any(valleys < roots[0]):
                    pass
                # Check for valleys ahead of the root. Shift phase if true.
                elif np.any(valleys > roots[0]):
                    g_phase += half_wave
                else:
                    # Umm... tilt?  no valleys, no peaks, one root.
                    # Maybe this is a line, not a sine?  Either way,
                    # no useful result possible.
                    output.append((None, None, None))
                    continue # skip to next dim
        # If we have no likely roots, this is likely
        #  not a sine-like curve within the sample area.
        else:
            #raise 'Not a sine-like waveform'
            output.append((None, None, None))
            continue # skip to next dim
        #
        ##########################################################
        # 
        # 
        # Now that we have initial guesses for all parameters for a sinusoidal function,
        # we can apply a fit.
        #
        guess = [g_amp, g_freq, g_phase, g_offset]
        try:
            params, pcov = curve_fit(sinFunc, timesteps, data[:,dim], p0=guess)
        except:
            print("'curve_fit()' HAS FAILED.  REPORTING GUESSES.")
            params = guess
            pcov = np.inf * np.ones([4,4])
        # resid = least_squares(sinFuncLS, guess, args=(timesteps,data[:,dim]))
        # output.append((guess, resid, sinFuncLS))
        # Output tuple for each data dimension is of the initial guess, the fitted parameters, and the sigma for each parameter
        output.append((guess, params, np.sqrt(np.diag(pcov))))
        #end foreach dim in data
    return output

##original guess generator
#guess = [
#        np.max(np.abs(data[:,dim])),
#        (timesteps[-1]-timesteps[0]),
#        timesteps[0]/np.pi,
#        data[np.argmin(np.abs(data[:,dim]))][0]]

## 2nd guess generator
#g_offset = np.mean([np.max(data[:,dim]),np.min(data[:,dim])]) # offset = mean(max,min)
#g_frequency = fftfr[np.argmax(np.abs(np.sqrt(np.power(fft2.real,2)+np.power(fft2.imag,2))))] *2*np.pi # frequency, seek fft peak bin, multiply by 2pi
#top_point = np.argmax(data[:,dim])
#root_point = np.argmin(np.abs(data[:,dim] - g_offset))
#phase_shift = timesteps[root_point]*g_frequency
#if top_point < root_point:
#    phase_shift += np.pi
#guess = [
#        np.mean([np.abs(np.max(data[:,dim])),np.abs(np.min(data[:,dim]))]), # amplitude = mean(abs(max),abs(min))
#        g_frequency, 
#        phase_shift,
#        g_offset]


# Have scipy try to fit the curve from our guess
#fit = curve_fit(sinFunc,timesteps,data[:,dim], p0=guess)
#from scipy.optimize import least_squares
#resid = least_squares(sinFunc, guess, args=(timesteps,data[:,dim]))

# plot and save
#plt.close()
#plt.scatter(timesteps,data[:,dim],color='k')
#plt.plot(timesteps,sinFunc(timesteps, *guess), color='c')
#plt.plot(timesteps,sinFuncLS(resid.x,timesteps,0), color='g')
#plt.savefig('./fit_test_guess.png')


'''
my runner:

import time
import numpy as np
import pandas as pd
from mms_curvature.mms_curvature import mms_Grad, mms_Curvature, mms_CurlB
from mms_curvature.mms_load_data_shims import mms_load_fgm
from MVA import minvar
from dateutil.parser import parse
from datetime import timedelta, datetime, timezone
#from flap_runner_v4 import flap_params
from flap_proc import flap_params, flap_segment
import re
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
import string

trange_file = 'flapping_2017_CS_crossings_veryshort.csv'
ucorr = (1e-12)/(1.257e-6)
columnnames = ['jtime', 'jmax', 'jx', 'jy', 'jz', 'ktime', 'kmax', 'kx', 'ky', 'kz', 'rx', 'ry', 'rz', 'l0', 'l0x', 'l0y', 'l0z', 'l1', 'l1x', 'l1y', 'l1z', 'l2', 'l2x', 'l2y', 'l2z']
agucolumns = ['rval', 'n_ang', 'gamma_N', 'ang_ratio', 'kl', 'km', 'kn', 'jl', 'jm', 'jn', 'kmax']
crossings = pd.read_csv(trange_file, header=0, index_col=0)
data=[]
agustats=[]
partials=[]
rawfgm=[]
events = set([(re.match('(\d+).*',element))[1] for element in crossings.index])
for event in events:
    subset = crossings[[bool(re.match((event+'.'),element)) for element in crossings.index]]
    times = subset.values.tolist()
    print("Bulk loading and processing event "+event)
    outlst, agulst, partslst, fgmraw = flap_params(tranges=times)
    data.append(outlst)
    agustats.append(agulst)
    partials.append(partslst)
    rawfgm.append(fgmraw)

# missing loading bits go here...

tslices = []
for trange in tRangeDT:
    pos_times = [None]*numProbes
    b_times = [None]*numProbes
    pos_values = [None]*numProbes
    b_values = [None]*numProbes
    for bird in range(numProbes):
        slice1 = np.greater_equal(fgmdata['mms'+str(bird+1)+'_fgm_b_gsm_'+str(data_rate)+'_l2']['x'], trange[0].replace(tzinfo=timezone.utc).timestamp())
        slice2 = np.less_equal(fgmdata['mms'+str(bird+1)+'_fgm_b_gsm_'+str(data_rate)+'_l2']['x'], trange[1].replace(tzinfo=timezone.utc).timestamp())
        slice3 = np.logical_and(slice1, slice2)
        b_times[bird] = np.copy(fgmdata['mms'+str(bird+1)+'_fgm_b_gsm_'+str(data_rate)+'_l2']['x'][slice3])
        b_values[bird] = np.copy(fgmdata['mms'+str(bird+1)+'_fgm_b_gsm_'+str(data_rate)+'_l2']['y'][slice3])
        pos_times[bird] = np.copy(fgmdata['mms'+str(bird+1)+'_fgm_r_gsm_'+str(data_rate)+'_l2']['x'])
        pos_values[bird] = np.copy(fgmdata['mms'+str(bird+1)+'_fgm_r_gsm_'+str(data_rate)+'_l2']['y'])
    tslices.append((pos_times,b_times,pos_values,b_values))

for s in range(len(tslices)):
    grad_0n, bm_0, Bmag_0, rm_0, t_master = mms_Grad(postimes=tslices[s][0], posvalues=tslices[s][2], magtimes=tslices[s][1], magvalues=tslices[s][3], normalize=True)
    fits = FitSine(t_master, bm_0)
    for i in range(len(fits)):
        plt.close()
        plt.scatter(t_master,bm_0[:,i],color='k')
        if fits[i][2] is not None:
            plt.plot(t_master,fits[i][2](fits[i][0], t_master, 0), color='c')
            plt.plot(t_master,fits[i][2](fits[i][1].x, t_master, 0), color='r')
            fitx = fits[i][1].x
            txt =  'amp:'+np.format_float_scientific(fitx[0], precision=5, exp_digits=1)
            txt += '  freq:'+  np.format_float_scientific(fitx[1], precision=5, exp_digits=1)
            txt += '\nphase:'+ np.format_float_scientific(fitx[2], precision=5, exp_digits=1)
            txt += '  offset:'+np.format_float_scientific(fitx[3], precision=5, exp_digits=1)
            plt.figtext(0.45, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=10)
        plt.savefig('./autofit_evt36'+string.ascii_lowercase[s]+'_dim'+str(i)+'.png')

'''

