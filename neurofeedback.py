#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 19:37:15 2024

@author: hungyunlu
"""


import os
import tables
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from scipy import stats
from itertools import groupby
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import statsmodels.api as sm
import warnings; warnings.filterwarnings("ignore")

DATA_FOLDER = '/Users/hungyunlu/Library/CloudStorage/Box-Box/Hung-Yun Lu Research File/Projects/Neurofeedback/data'
PLOT_FOLDER = '/Users/hungyunlu/Library/CloudStorage/Box-Box/Hung-Yun Lu Research File/Projects/Neurofeedback/results'

SESSIONS = [

    #### Brazos ####
    'braz20230404','braz20230406','braz20230407','braz20230411',
    'braz20230413','braz20230417','braz20230424','braz20230425',
    'braz20230427','braz20230428','braz20230502','braz20230503',
    'braz20230504','braz20230508','braz20230509','braz20230510',

    #### Airport ####
    'airp20231216','airp20231218','airp20231220','airp20231221',
    'airp20231223','airp20231226','airp20240226','airp20240227',
    'airp20240313','airp20240314','airp20240315','airp20240318',
    'airp20240320','airp20240321','airp20240322',
    ]

bands      = {'alpha':[8,12],'low_beta':[12,20],'high_beta':[20,35],'gamma':[35,100],'broad':[0,100]}  # CAP
block_clr  = {'Control':'g','Main':'r','Transfer':'b','Washout':'k'} # CAP
block_info = [['Control','Main','Transfer','Washout'] for _ in range(len(SESSIONS))]

targ_label = ['High','Center','Low'] # CAP
n_targ     = len(targ_label) # CAP
colors     = dict(zip(targ_label,['blue','green','orange'])) # CAP
SAVEFIG    = False

# Brazos adjustment
block_info[1].pop(3)
block_info[2].pop(2)
block_info[4].pop(2)
block_info[9].pop(2)
block_info[10].pop(2)
block_info[13].pop(2)
block_info[14].pop(2)

# Airport adjustment
block_info[0+16].pop(2)
block_info[2+16].pop(2)
block_info[4+16].pop(2)
block_info[5+16].pop(3)
block_info[5+16].pop(2)
block_info[6+16].pop(2)
block_info[9+16].pop(2)
block_info[10+16].pop(2)
block_info[11+16].pop(2)

# Make it a global constant
BLOCK_INFO = dict(zip(SESSIONS,block_info))

def nonan(array):
    """Remove nan from the array."""
    return array[~np.isnan(array)]


def slide_avg(array, n):
    avg = np.zeros(array.shape)
    for i in range(len(array)):
        if i < n:
            avg[i] = np.sum(array[:i+1],axis=0)/float(i+1)
        else:
            avg[i] = np.sum(array[i-n+1:i+1],axis=0)/float(n)
    return avg


class Time_align:
    '''
    A class designed to align data.
    The input means how many seconds of data to align before and after that aligned metric.
    '''
    def __init__(self,time):
        self.t_second  = time
        self.t_before  = int(60 * self.t_second) # Sampling rate at 60 Hz
        self.t_after   = int(60 * self.t_second)
        self.t_samples = self.t_before + self.t_after
        self.time      = np.linspace(-self.t_second, self.t_second, self.t_samples)


class Neurofeedback_Block:
    """
    Read VTA LFP neruofeedback behavioral data (HDF) for each BLOCK.
    Since there are multiple blocks in a session, each block is read separately
    with provided block name.
    The block information for each session is stored in the BLOCK_INFO constant.

    filename: '{session}/{long_filename}.hdf' format.    
    block_name: 'Control', 'Main', 'Transfer', or 'Washout'.
    """

    def __init__(self,filename,block_name):
        
        print(f'Reading file: {filename}, a {block_name} block.')

        self.table = tables.open_file(filename)

        # [Basic info]
        self.session_name = filename.split('/')[0] # Should use re but this works.
        self.subject = filename[:4]
        self.block_name = block_name

        # [Behavioral info]
        self.lfp_cursor = None
        self.lfp_target = None
        self.lfp_power = None
        self.state = None
        self.state_time = None
        self.time_wait = None
        self.time_reward = None 
        self.n_success = None
        self.targs = None
        self.targ_per_trial = None
        self.read_behavior()
        
        # [Trial durations]
        self.times_table = None
        self.calc_duration()
    
        
    def read_behavior(self):

        # Cursor is moved along the third dimension
        self.lfp_cursor    = self.table.root.task[:]['lfp_cursor'][:,2]

        # The LFP target: first item is stored -100, replacing it w actual numbers
        self.lfp_target    = self.table.root.task[:]['lfp_target'][:,2]
        self.lfp_target[0] = self.lfp_target[1]
        self.lfp_power     = self.table.root.task[:]['lfp_power']
        self.state         = self.table.root.task_msgs[:]['msg']
        self.state_time    = self.table.root.task_msgs[:]['time']

        # Unique trials in time
        self.time_wait   = self.trial_time(b'wait')
        self.time_reward = self.trial_time(b'reward')
        self.n_success = len(self.time_reward)

        # All targets
        self.targs = np.unique(self.lfp_target)[::-1]
        self.targ_per_trial = self.lfp_target[self.time_reward]
        if len(self.targs) == n_targ:
            self.rule = dict(zip(self.targs,targ_label))
            self.targ_per_trial = np.vectorize(self.rule.get)(self.targ_per_trial)


    def calc_duration(self):
        """Trial duration profile for neurofeedback blocks"""
        reward = self.time_reward
        wait   = self.time_wait[:-1]
        duration = (reward-wait)/60.
        df = pd.DataFrame({'target':self.targ_per_trial,
                           'time':duration,
                           'reward':reward,
                           'wait':wait})
        self.times_table = df


    @property
    def band_range(self):
        """
        Manually caclulate Fourier transformation.
        Use to calculate the bins and indices of the frequency bands
        """
        npts = 0.2*1000
        nfft = 2**int(np.ceil(np.log2(npts)))
        freqs = np.arange(0,1000,float(1000)/nfft)#[0:int(nfft/2)+1]
        return freqs


    def power(self,start,end):
        """Find power of a certain band given starting and ending frequencies"""
        start = np.argmin(abs(start-self.band_range)) # Find index for that freq
        end   = np.argmin(abs(end-self.band_range))   # Find index for that freq
        res   = np.zeros((self.lfp_power.shape[0],))
        for channel in range(3):
            # Average over the three selected channels.
            res += self.lfp_power[:,start+129*channel:end+129*channel].sum(axis=1).flatten()
        res /= 3
        return res


    def trial_ind(self, state_type):
        return np.where(self.state==state_type)[0]


    def trial_time(self, state_type):
        return self.state_time[self.trial_ind(state_type)]
    
    
    def profile(self,metric):
        """25 Hz corresponds to the index 6 in the lfp_power matrix."""
        beta  = self.power(25,35)
        gamma = self.power(35,100)
        theta = self.power(4,8)
        alpha = self.power(8,12)
        delta = self.power(1,4)
        fraction = beta/(beta+gamma)
        
        match metric:
            case 'beta':  return beta
            case 'gamma': return gamma
            case 'theta': return theta
            case 'alpha': return alpha
            case 'delta': return delta
            case 'fraction': return fraction

        
    def index(self,metric):
        
        def find_index(start,end):
            start_ind = np.argmin(abs(start-self.band_range)) # Find index for that freq
            end_ind   = np.argmin(abs(end-self.band_range))
            return [start_ind, end_ind]
        
        match metric:
            case 'beta':  return find_index(25,35)
            case 'gamma': return find_index(35,100)
            case 'theta': return find_index(4,8)
            case 'alpha': return find_index(8,12)
            case 'delta': return find_index(1,4)

            
class Neurofeedback_Session:
    
    def __init__(self, session):
        
        """session: SUBJECT+YYYYMMDD, e.g., braz20230404"""
        self.session = session
        print(f'Reading session {self.session}')
        
        self.has_control = False
        self.has_main = False
        self.has_transfer = False
        self.has_washout = False
        self.C = None
        self.M = None
        self.T = None
        self.W = None
        self.read_session(self.session)
        
        self.location = None
        self.find_location(block='Control')

    def read_session(self,session):
        os.chdir(DATA_FOLDER)
        files = sorted(glob.glob(f'{session}/{session}*.hdf'))
        names = BLOCK_INFO[session]
        
        # First read all the blocks, and then assign them based on the block name
        blocks = [Neurofeedback_Block(files[i],names[i]) for i in range(len(files))]
        
        match names:
            case ['Control', 'Main', 'Transfer', 'Washout']: 
                self.C, self.M, self.T, self.W = blocks
                self.has_control = True
                self.has_main = True
                self.has_transfer = True
                self.has_washout = True
            case ['Control','Main','Transfer']: 
                self.C, self.M, self.T = blocks
                self.has_control = True
                self.has_main = True
                self.has_transfer = True
                self.has_washout = False
            case ['Control','Main','Washout']: 
                self.C, self.M, self.W = blocks
                self.has_control = True
                self.has_main = True
                self.has_transfer = False
                self.has_washout = True
            case ['Control','Main']: 
                self.C, self.M = blocks
                self.has_control = True
                self.has_main = True
                self.has_transfer = False
                self.has_washout = False
                
                
    def find_location(self, block: str = 'Control'):
        """Return the [5,50,95] percentiles of the power fraction in a given block."""
        match block:
            case 'Control':
                l = np.percentile(nonan(self.C.profile('fraction')),[5,50,95])
            case 'Main':
                l = np.percentile(nonan(self.M.profile('fraction')),[5,50,95])
            case 'Transfer':
                l = np.percentile(nonan(self.T.profile('fraction')),[5,50,95])
            case 'Washout':
                l = np.percentile(nonan(self.W.profile('fraction')),[5,50,95])
        return np.round(l,3)
    
                
    def __repr__(self):
        return f'Neurofeedback_Session: {self.session}'
    
    
    def get_block_data(self, block):
        match block:
            case 'Control':
                block_data = self.C
            case 'Main': 
                block_data = self.M
            case 'Transfer':
                if self.has_transfer:
                    block_data = self.T
                else:
                    raise ValueError('This session does not have a Transfer block.')
            case 'Washout':
                if self.has_washout:
                    block_data = self.W
                else:
                    raise ValueError('This session does not have a Washout block.')
                
        return block_data
    
    
    def find_spectrogram(self, block: str):
        block_data = self.get_block_data(block)
        result = np.zeros((block_data.lfp_power.shape[0],129))
        for channel in range(3): # The recorded data has three channels.
            result += block_data.lfp_power[:,129*channel:129+129*channel].squeeze()
        result /= 3 # Average over three channels
        data = result[:,:26] # Take only 0-100 Hz
        return data
    
    
    def find_aligned(self, 
                     block: str,
                     band: str, 
                     target: str | None, 
                     behavioral_metric: str | list[int], 
                     T: Time_align = Time_align(1)):
        """
        Align data to behavioral metrics. 
        For example, we may want to align "LFP power" or "spectrogram" to
        "reward" or "begin of the trial" in the "Main" or "Transfer" blocks.
        
        Return unaveraged aligned data. 
        First two dimensions are number of trials and number of samples in one align.
        For example, if aligned for 0.5 seconds, then there will be 0.5*2*60 = 60 samples in the second dimension.
        """
        
        # Grab data from the selected block
        block_data = self.get_block_data(block)
                
        # Make sure what data is needed
        assert band in ['beta','gamma','theta','alpha','delta','fraction','spectrogram'], \
            ValueError('Wrong band option.')
        if band == 'spectrogram':
            data = self.find_spectrogram(block)
        else:
            data = block_data.profile(band)
            
        # The times table for that block.
        df = block_data.times_table
        if target is not None:
            assert target in ['High','Center','Low'], ValueError('Wrong target option.')
            df = df[df.target == target]
            
        # The points to align
        if type(behavioral_metric) is str:
            align_pts = df[behavioral_metric].values
        else:
            # Manually input a list of points to be aligned.
            # Be flexible!
            align_pts = behavioral_metric
        
        # Initiate an empty targ_data for different conditions.
        if data.ndim == 1: # lfp power
            targ_data = np.empty([len(align_pts), T.t_samples])
        elif data.ndim == 2: # spectrogram
            targ_data = np.empty([len(align_pts), T.t_samples, data.shape[1]])
            
        targ_data[:] = np.nan
        for j in range(len(align_pts)):
            # Use these criteria to make sure we have enough samples for aligning.
            criteria1 = align_pts[j]-T.t_before > 0
            criteria2 = align_pts[j]+T.t_after < block_data.state_time[-1]
            if criteria1 and criteria2:
                targ_data[j] = data[align_pts[j]-T.t_before:align_pts[j]+T.t_after]
                
        return targ_data
    
    
class Neurofeedback_All:
    
    def __init__(self, subject):
        self.subject = subject
        self.session_names = [s for s in SESSIONS if s[:4] == self.subject]
        self.sessions = None
        self.Cs = None
        self.Ms = None
        self.Ts = None
        self.Ws = None
        self.read_all_sessions()
        

    def read_all_sessions(self):
        self.Cs = []
        self.Ms = []
        self.Ts = []
        self.Ws = []
        
        for session in self.session_names:
            data = Neurofeedback_Session(session)
            setattr(self, session, data) # Set it this way for easy access.
            if data.has_control: self.Cs.append(data.C)
            if data.has_main: self.Ms.append(data.M)
            if data.has_transfer: self.Ts.append(data.T)
            if data.has_washout: self.Ws.append(data.W)
            
    
    @staticmethod
    def find_trialset(lst_neurofeedback_blocks,n_samples):
        """
        Find the trial sets of given neurofeedback blocks (Main or Transfer) by a given n_samples in a set.
        Trial sets are calculated individually for each target.
        For a given block (e.g., Main block), the first trial set would be trial 0 to n_samples.
        If the number of trials cannot be divided by n_samples, then the remaining trials are
        contained in the last trial set.
    
        INPUT
        lst_neurofeedback_blocks:
            list of neurofeedback blocks. Ms, Ts, ...
            Can take multiple sessions of data. If used in single-session analysis,
        n_samples:
            int. number of samples in a trial set.
            If n_samples = 20, then there will be 5 trial sets.
    
        OUTPUT
        trial_set:
            pd.DataFrame. An extension of the times_table for the neurofeedback blocks.
            If multile sessions are given, then those extended times_table are concatenated.
        """
    
        trialset = []
        for i in range(len(lst_neurofeedback_blocks)):
            df = lst_neurofeedback_blocks[i].times_table
            for targ in targ_label:
                df_ = df[df.target==targ]
                df_['time'] = slide_avg(df_['time'],20)
                ## len(df_)//10*10 will be 90 or 100 in the case of shorter Transfer block
                n_sets = len(df_)//10*10 // n_samples
    
                ## Do this in case for that target, there are more than 100 trials.
                ## The remaining trials will be labeled the last trial set.
                sets = np.ones(len(df_),)*(n_sets-1)
                sets[:n_sets*n_samples] = np.repeat(np.arange(n_sets),n_samples)
    
                df_['set'] = sets
                df_.set = df_.set.astype('int')
                df_['session'] = lst_neurofeedback_blocks[i].session_name
    
                trialset.append(df_)
        trialset = pd.concat(trialset,ignore_index=True)
        return trialset

#%%

airp = Neurofeedback_All('airp')
braz = Neurofeedback_All('braz')
    
#%%

def plot_nbp_pdf_target_location(nf_session: Neurofeedback_Session):
    """
    Probability density and target location
    """
    C = nf_session.C
    loc = nf_session.find_location()
    plt.figure(figsize=(4,3))
    ratio = nonan(C.profile('fraction'))
    xx = np.linspace(0,1,300)
    c1 = gaussian_kde(ratio)(xx)
    plt.plot(xx,c1,color='k')
    
    for i in range(3):
        plt.axvline(loc[i],c=list(colors.values())[i],ls='--')
        plt.axvspan(loc[i]-0.04, loc[i]+0.04, alpha=0.2, color=list(colors.values())[i])
    plt.xlabel('Normalized Beta Power')
    plt.ylabel('Probability Density')
    if SAVEFIG:
        plt.savefig(f'{nf_session.session}_Control_target_example.svg')
    plt.show()
    

def plot_NBP_SegmentControl(c: Neurofeedback_Block, show_stats: bool = True):
    """
    Stability of NBP within a Control block.
    """
    title = c.session_name+"-"+c.block_name
    overlap,n_segment,color_adjust = 0.45,5,1
    cmap = plt.cm.binary
    ratio = nonan(c.profile('fraction'))
    profile = np.zeros((n_segment,3))

    ## Restructure PF into minute-by-minute
    m = ratio.shape[0]//n_segment
    data = [ratio[m*i:m*(i+1)] for i in range(n_segment)]
    data = sorted(data,key=lambda x: np.mean(x))
    for i in range(n_segment):
        profile[i] = np.percentile(data[i],[95,50,5])

    ## To approximate the KDE
    n_points = 300
    xx = np.linspace(0,1,n_points)

    fig,ax = plt.subplots(figsize=(4,0.1))
    for i in range(n_segment):
        plt.plot([i,i+1],[0,0],c=cmap((i+color_adjust)/(n_segment+color_adjust)),lw=5)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Time')
    plt.title('Segment Control block')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.show()

    fig,ax = plt.subplots(figsize=(4,3))
    for i, d in enumerate(data):
        pdf = gaussian_kde(d)
        y = -i*(1.0-overlap)
        curve = pdf(xx)
        plt.fill_between(xx,np.ones(n_points)*y,curve+y,
                         color=cmap((i+color_adjust)/(n_segment+color_adjust)),alpha=1)
        plt.plot(xx, curve+y, c='k')
    plt.xlabel('Normalized Beta Power')
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylim([y,None])
    plt.title(title)
    if SAVEFIG:
        plt.savefig(f'{title}_NBP_SegmentControl_Dist.svg')
    plt.show()

    fig,ax = plt.subplots(figsize=(1,1))
    plt.eventplot(profile,lineoffsets=-1,
                  colors=cmap(np.arange(color_adjust,color_adjust+n_segment)/(n_segment+color_adjust)),
                  linewidths=5)
    plt.xlim([0,1])
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xticks([0,0.5,1])
    if SAVEFIG:
        plt.savefig(f'{title}_NBP_SegmentControl_Loc.svg')
    plt.show()
    
    if show_stats:
        n_segment = 5 # Divide 5 minutes of data into 5 segments.
        ratio = nonan(c.profile('fraction'))
        profile = np.zeros((n_segment,3))
        m = ratio.shape[0]//n_segment
        data = [ratio[m*i:m*(i+1)] for i in range(n_segment)]
        for i in range(n_segment):
            profile[i] = np.percentile(data[i],[95,50,5]) # Find percentiles based on segmented data
        
        df = pd.DataFrame({'Target':np.tile([1,2,3],n_segment),
                           'Time':np.repeat(np.arange(n_segment),3),
                           'NBP':profile.ravel()})
        print(pg.friedman(data=df, dv='NBP', within='Time', subject='Target'))
            
        
def plot_time_smoothed_trial_duration(nf_all: Neurofeedback_All, block: str = 'Main'):
    """
    Time-smoothed trial duration over time.
    """
    
    if nf_all is None: return 
    
    match block:
        case 'Main': 
            lst = nf_all.Ms
            crop = 100
            xticks = [0,25,50,75,100]
        case 'Transfer': 
            lst = nf_all.Ts
            crop = 90 # For transfer blocks, take only the first 90 trials since not all are complete.
            xticks = [0,30,60,90]
        
    plt.figure(figsize=(4,3))
    for targ in targ_label:
        combined = []
        for i in range(len(lst)):
            df = getattr(lst[i],'times_table')
            data = slide_avg(df[df.target==targ]['time'].values,20)
            combined.append(data[:crop])
        combined = np.array(combined)
        avg = np.mean(combined,0)
        std = np.std(combined,0) / np.sqrt(len(combined)-1)
        plt.plot(range(crop),avg,c=colors[targ],label=targ)
        plt.fill_between(range(crop),
                         avg+std,avg-std,
                         alpha=0.2,color=colors[targ])
    plt.legend(frameon=False,loc='upper right')
    plt.xlabel('Trial Number')
    plt.ylabel('Avg trial duration (s)')
    plt.title(f'Trial duration of all sessions - {block}')
    plt.xticks(xticks)
    if SAVEFIG:
        plt.savefig(f'{nf_all.subject}-{block}_TrialDuration_NoGaussian.svg')
    plt.show()
    

def plot_time_smoothed_trial_duration_combined(nf_all: Neurofeedback_All):
    """
    Time-smoothed trial durations, not separating targets.
    """
    
    if nf_all is None: return 

    def combined(lst,n_trials):
        res = []
        for i in range(len(lst)):
            df = getattr(lst[i],'times_table')
            # data = gaussian_filter(slide_avg(df['time'].values,20),3,mode='nearest')
            data = slide_avg(df['time'].values,60)
            res.append(data[:n_trials])
        res = np.array(res)
        return res
    
    plt.figure(figsize=(4,3))
    n_trials = 300
    ms = combined(nf_all.Ms,n_trials)
    avg = np.mean(ms,0)
    std = np.std(ms,0) / np.sqrt(len(ms)-1)
    plt.plot(range(n_trials),avg,c='k')
    plt.fill_between(range(n_trials),avg+std,avg-std,alpha=0.2,color='k',label='Main')
    
    n_trials = 270
    ts = combined(nf_all.Ts,270)
    avg = np.mean(ts,0)
    std = np.std(ts,0) / np.sqrt(len(ts)-1)
    plt.plot(range(n_trials),avg,c='b')
    plt.fill_between(range(n_trials),avg+std,avg-std,alpha=0.2,color='b',label='Transfer')
    plt.legend(frameon=False)
    plt.xlabel('Trial Number')
    plt.ylabel('Avg trial duration (s)')
    plt.title('Trial duration of all sessions over trials')
    if SAVEFIG:
        plt.savefig(f'{nf_all.subject}-TrialDuration_Merge.svg')
    plt.show()
    

def plot_max_improvement(nf_all: Neurofeedback_All, 
                        block: str = 'Main', 
                        show_stats: bool = True):
    """
    Maximal improvement.
    """
    
    if nf_all is None: return 

    match block:
        case 'Main': 
            lst = nf_all.Ms
        case 'Transfer': 
            lst = nf_all.Ts
    
    df = nf_all.find_trialset(lst,10)
    each = []
    for targ in targ_label:
        grouped = df[df.target==targ][['set','session','time']].groupby(['session','set']).mean()
        unstacked = grouped.unstack(level=-1)['time']
        where = unstacked.idxmin(axis=1).reset_index(drop=True) # where
        improve = (unstacked[0]-unstacked.min(axis=1))/unstacked[0] *100
        res = improve.to_frame(name='improvement')
        res['target'] = targ
        res['session'] = res.index.values
        res['where'] = where.values
        res.reset_index(drop=True, inplace=True)
        each.append(res)
    result = pd.concat(each)

    plt.figure(figsize=(4,3))
    for i in range(3):
        m = result[result.target==targ_label[i]].improvement.mean()
        s = result[result.target==targ_label[i]].improvement.sem()
        plt.errorbar(i,m,yerr=s,capsize=3,c='k')

    ax = sns.barplot(x="target", y="improvement", data=result, palette=colors,
          ci=None,edgecolor="black",errcolor="black",errwidth=1.5,capsize = 0.05,alpha=0.5)

    sns.stripplot(x="target", y="improvement", data=result, ax=ax,
                  palette=colors,edgecolor="black",linewidth=1,alpha=0.5,legend=False)
    plt.ylabel('Maximum improvement (%)')
    plt.xlabel(None)
    plt.ylim([-3,100])
    plt.title(f'{nf_all.subject}-{block}')
    if SAVEFIG:
        plt.savefig(f'{nf_all.subject}-{block}_maxlearning_rate.svg')
    plt.show()

    if show_stats:
        for targ in targ_label:
            d = result[result.target==targ].improvement
            # _,p = stats.ttest_1samp(d,0)
            print(targ, len(d))
            res = pg.ttest(d,0,alternative='greater')
            print(res[['p-val','cohen-d']])
    
        print(pg.anova(result,dv='improvement',between='target',detailed=True),'\n')
        print(pg.pairwise_tukey(result,dv='improvement',between='target'))
    

def plot_example_nbp(nf_session: Neurofeedback_Session):
    """
    Demonstration of early, mid, late trials.
    """    
    def find_intarget(data):
        """
        Count how many times the time series are within the target
        """
        ts = (data<=loc+0.042)&(data>=loc-0.042) # PF time series change to binary time series
        counts = np.array([len(list(group)) for _, group in groupby(ts)]) # Count different groups
        slices = np.arange(0,len(counts)//2*2,2) # The index for IN the target (or True in ts)
        if not ts[0]: slices += 1 # Adjust if the first index is False
        intarget = counts[slices]
    
        return counts, intarget
    
    loc = nf_session.find_location()
    loc = loc[2]
    M = nf_session.M
    NBP = slide_avg(M.profile('fraction'),5)
    df = M.times_table
    df = df[df.target=='High'].reset_index()
    h,l = loc+0.042,loc-0.042
    example = [4,32,75]
    
    plt.figure(figsize=(5,3))
    td = 0
    for i in range(3):
        trial = df.iloc[example[i]]
        start = trial.wait
        end = trial.reward-5
        data = NBP[start:end]
        td = max(td,end-start)
        plt.scatter(range(end-start),data-i,s=10,
                    c=data,cmap='rainbow',vmin=0.4,vmax=0.9)
        plt.plot(range(end-start),data-i,c='k',lw=0.5)
        plt.axhline(l-i,c='k')
        plt.axhline(h-i,c='k')
        counts, intarget = find_intarget(data)
        print(np.round(intarget/60*1000))
        print(len(intarget))
        plt.eventplot(np.cumsum(counts),lineoffsets=0.9-i,linelengths=0.2)
    
    plt.xticks(np.arange(td//60+2)*60,np.arange(td//60+2))
    plt.xlabel('Time (s)')
    plt.yticks([0.5,-0.5,-1.5],example)
    plt.ylabel('Trial number')
    plt.title(f'{M.session_name}-{M.block_name}_SingleTrial')
    plt.colorbar()
    if SAVEFIG:
        plt.savefig(f'{M.session_name}-{M.block_name}_SingleTrial_event.svg')
    plt.show()
    
    
def plot_aligned_lfp(
        nf_session,
        block,
        band,
        behavioral_metric):
    
    """
    Align NBP/power to Reward/Hold in a Main/Transfer block.
    """
    
    assert band != 'spectrogram' # This is only for aligning LFP power.
    
    if behavioral_metric == 'reward':
        t = Time_align(1)
    else:
        t = Time_align(0.5)
    
    plt.figure(figsize=(4,3))    
    # Special plotting for aligning NBP.
    if nf_session.has_control and band=='fraction':
        l = nf_session.find_location()
        for i in range(3):
            plt.axhline(l[i],c='k',ls='--',lw=1)
            
    # Deal with each target
    for targ in targ_label:
        targ_data = nf_session.find_aligned(block=block,
                                            band=band,
                                            target=targ,
                                            behavioral_metric=behavioral_metric,
                                            T=t)
        targ_avg = np.nanmean(targ_data, axis=0)
        targ_sem = np.nanstd(targ_data, axis=0)/np.sqrt(len(targ_data)-1)
        plt.plot(t.time, slide_avg(targ_avg,10), c=colors[targ], label=f'Target {targ}')
        plt.fill_between(t.time, slide_avg(targ_avg,10)-targ_sem,slide_avg(targ_avg,10)+targ_sem, color=colors[targ],alpha=0.4)

    plt.title(f'{nf_session.session}-{block}_aligned_{band}_{behavioral_metric}')
    plt.ylabel('Power'); plt.xlabel('Time (second)')
    if SAVEFIG:
        plt.savefig(f'{nf_session.session}-{block}_aligned_{band}_{behavioral_metric}.svg')
    plt.show()
    
    
def plot_PSD(nf_session: Neurofeedback_Session, block: str = 'Control'):
    """
    Power Spectral Density of a Control block
    """

    block_data = np.nanmean(nf_session.find_spectrogram(block), axis=0)
    
    # PSD for Control block
    plt.figure(figsize=(5,4))
    plt.plot(block_data)
    plt.xticks(np.arange(26), np.linspace(0,100,26,dtype=int), rotation=90)
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD ($V^2$)')
    if SAVEFIG:
        plt.savefig(f'{nf_session.session}-{block}_PSD.svg')
    plt.show()
    
    

def plot_algined_spectrogram(nf_session, block, color_extreme: int = None):
    """ 
    Align spectrogram to reward
    """
    
    nf_session = braz.braz20230404
    block = 'Main'
    
    title = nf_session.session + '-' + block
    t = Time_align(0.5)
    C = np.nanmean(nf_session.find_spectrogram('Control'), axis=0)
    
    
    k = 1
    plt.figure(figsize=(10,4.5))
    for targ in targ_label:
        NF = np.nanmean(
                nf_session.find_aligned(
                    block=block,
                    band='spectrogram',
                    target=targ,
                    behavioral_metric='reward',
                    T=t),
                axis=0)
        diff = (NF-C)/C *100
        
        plt.subplot(1,3,k)
        plt.imshow(np.flipud(diff.T),aspect='auto',
                   vmin=-color_extreme,vmax=color_extreme,
                   cmap='seismic',interpolation='gaussian')
        plt.yticks(np.linspace(0,26,6)-0.5,np.linspace(100,0,6,dtype=int))
        plt.xticks(np.linspace(0,60,5)-0.5,np.linspace(-0.5,0.5,5))
        plt.axvline(30,lw=1,ls='--',c='k')
        plt.axhline(26-6+1,lw=1,ls='--',c='k')
        plt.axhline(26-9-1,lw=1,ls='--',c='k')
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        # plt.xticks([]); plt.yticks([])
        plt.title(targ)
        plt.colorbar(label='% power difference from Control',orientation='horizontal')
        k+=1
    
    plt.subplots_adjust(wspace=0.3)
    plt.suptitle(title)
    if SAVEFIG:
        plt.savefig(f'{title}_NF_spectrogram.svg')
    plt.show()    


def plot_power_change_percentage(nf_session: Neurofeedback_Session,
                                 show_stats: bool = True):
    """
    Percent of power change compared to Control
    """
    C = nf_session.C
    bands = ['delta','theta','alpha','beta','gamma']
    spectrogram_C = nf_session.find_spectrogram('Control')
    total_df = []

    for k in range(len(bands)):
        ctrl_avg = spectrogram_C[:,slice(*C.index(bands[k]))].sum(1).mean(0) # The avg power in Control
            
        for block in ['Main','Transfer']:
            spect = nf_session.find_spectrogram(block)
            block_data = nf_session.get_block_data(block)
            df = block_data.times_table
            
            for targ in targ_label:
                align_pts = df[df.target==targ]['reward'].values
                psd = np.zeros((len(align_pts),26)) # 26 is the number of frequency indices from 0 to 100 Hz
                for j in range(len(align_pts)):
                    psd[j]= spect[align_pts[j]-12:align_pts[j]].mean(0) # 12 samples means 200 ms in 60 Hz sampling rate
                    
                power = ( psd[:,slice(*block_data.index(bands[k]))].sum(1)/ctrl_avg - 1 ) * 100 # Turned into % 
                
    
                total_df.append(pd.DataFrame({'Block':block,
                                              'Target':targ,
                                              'Power':power,
                                              'Band':bands[k]}))

    total_df = pd.concat(total_df)
    for block in ['Main','Transfer']:
        df = total_df[total_df.Block==block]
        sns.catplot(data=df,x='Target',y='Power',errorbar='se',palette=colors,kind='bar',col='Band',aspect=0.5)
        plt.suptitle(f'{nf_session.session}-{block}_PowerBar',y=1.05)
        if SAVEFIG:
            plt.savefig(f'{nf_session.session}-{block}_PowerBar.svg')
        plt.show()
        
    if show_stats:
        for block in ['Main','Transfer']:
        
            print(f'----{block}----')
        
            for band in bands:
                d = total_df[(total_df.Block==block)&(total_df.Band==band)]
        
                print(f'---{band}---')
                # for targ in targ_label:
                #     res = pg.ttest(d[d.Target==targ]['Power'],0)
                #     print(f"{targ} {res[['p-val','cohen-d']].values}")
        
                print('---ANOVA---')
                print(pg.anova(data=d, dv='Power', between='Target'))
                print()
                print(pg.pairwise_tukey(data=d, dv='Power', between='Target'))
    

def plot_reward_rate(nf_all: Neurofeedback_All, show_stats: bool = True):
    """
    Reward per minute
    """
    
    if nf_all is None: return 

    def count_intarget(nbp):
        intargets = [] # Count how many times the cursor was in the target region
        for loc in range(3):
            ts = (nbp<=top[loc])&(nbp>=bot[loc]) # NBP time series change to binary time series of whether it's in the target region or not
            counts = np.array([len(list(group)) for _, group in groupby(ts)]) # Count different groups
            slices = np.arange(0,len(counts)//2*2,2) # The index for IN the target (or True in ts)
            if not ts[0]: slices += 1 # Adjust if the first index is False
            intarget = counts[slices]
            intargets.append(np.sum(intarget>=13)) # Count how many are greater than 12 samples
        return intargets
    
    rc,rm,rt,rw = [],[],[],[] ## Reward rates for all blocks
    
    for session in nf_all.session_names:
        nf_session: Neurofeedback_Session = getattr(nf_all, session)
        C = nf_session.C
        M = nf_session.M
        T = nf_session.T if nf_session.has_transfer else None
        W = nf_session.W if nf_session.has_washout else None
        
        target = nf_session.find_location()
        top,bot = target + 0.042, target - 0.042 # Boundaries for targets
    
        # Control block
        nbp = nonan(C.profile('fraction'))
        intargets = count_intarget(nbp)
        rc.append(np.sum(intargets)/5) # Reward rate (in any target) in Control
        
        # Main block
        rm.append(len(M.targ_per_trial) / (M.state_time[-2] / 3600))
        
        # Transfer block
        if T is not None:
            rt.append(len(T.targ_per_trial) / (T.state_time[-2] / 3600))
        
        # Washout block
        if W is not None:
            nbp = nonan(W.profile('fraction'))
            intargets = count_intarget(nbp)
            rw.append(np.sum(intargets)/5) # Reward rate (in any target) in Control
    
    dfc = pd.DataFrame({'Rate':rc,'Block':'Control'})
    dfm = pd.DataFrame({'Rate':rm,'Block':'Main'})
    dft = pd.DataFrame({'Rate':rt,'Block':'Transfer'})
    dfw = pd.DataFrame({'Rate':rw,'Block':'Washout'})
    
    df = pd.concat([dfc,dfm,dft,dfw])
    plt.figure(figsize=(4,3))
    ax = sns.barplot(x="Block", y="Rate", data=df, palette=block_clr,
          edgecolor="black",errcolor="black",errwidth=1.5,capsize = 0.05,alpha=0.5)
    
    sns.stripplot(x="Block", y="Rate", data=df, ax=ax,
                  palette=block_clr,edgecolor="black",linewidth=1,alpha=0.5,legend=False)
    plt.ylabel('Reward per minute')
    if SAVEFIG:
        plt.savefig(f'{nf_all.subject}_ChanceLevel.svg')
    plt.show()
    
    if show_stats:
        print(pg.anova(data=df,dv='Rate',between='Block'))
        print('\n\n')
        print(pg.pairwise_tukey(data=df,dv='Rate',between='Block'))


def plot_correlation_NBP_and_power(nf_session: Neurofeedback_Session):
    """
    Correlation of NBP and beta or gamma
    """
    C = nf_session.C
    
    # Panel A
    power = C.lfp_power.reshape(C.lfp_power.shape[0],129,3,order='F').mean(2)
    beta = power[100:,6:9].sum(1)
    spearman = np.zeros((7,17))
    
    for start in range(7):
        for end in range(17):
            denom = power[100:,start:end+9].sum(1)
            r,_ = stats.spearmanr(beta/denom,beta)
            spearman[start,end] = r
    
    plt.figure(figsize=(2,4))
    plt.pcolormesh(spearman.T,cmap='binary',edgecolors='k',lw=1,vmin=0,vmax=1.0)
    plt.colorbar(label='Spearman rank corr coef')
    plt.xticks(np.arange(7)+0.5,4*np.arange(7).astype(int))
    plt.yticks(np.arange(17)+0.5,4*np.arange(9,26).astype(int))
    plt.xlabel('Starting frequency'); plt.ylabel('Ending frequency')
    if SAVEFIG:
        plt.savefig(f'{C.session_name}_Spearman.svg')
    plt.show()
    
    # Panel B-D
    ranges = [slice(1,15),slice(6,26),slice(5,11)]
    ranges_name = ['5-60','25-100','20-45']
    power = C.lfp_power.reshape(C.lfp_power.shape[0],129,3,order='F').mean(2)
    
    for i in range(3):
        beta = power[100:,6:9].sum(1)
        denom = power[100:,ranges[i]].sum(1)
        plt.figure(figsize=(3,2.5))
        plt.hist2d(beta/denom,np.log(beta),cmap='Blues',bins=30,vmin=1,vmax=300,norm='log')
        plt.xlim([0,1])
        # plt.xlim([0,100])
        plt.colorbar()
        plt.ylabel('Log beta power')
        plt.xlabel('NBP')
        plt.title(f'{stats.spearmanr(beta/denom,beta)[0]:.3f}')
        if SAVEFIG:
            plt.savefig(f'{C.session_name}-{C.block_name}_2Dhist_{ranges_name[i]}Hz.svg')
        plt.show()
    
    # Panel E
    power = C.lfp_power.reshape(C.lfp_power.shape[0],129,3,order='F').mean(2)
    gamma = power[100:,9:26].sum(1)
    denom = power[100:,6:26].sum(1)
    plt.figure(figsize=(3,2.5))
    plt.hist2d(gamma/denom,np.log(gamma),cmap='Reds',bins=30,vmin=1,vmax=300,norm='log')
    plt.xlim([0,1])
    # plt.xlim([0,100])
    plt.colorbar()
    plt.ylabel('Log gamma power')
    plt.xlabel('NBP')
    plt.title(f'{stats.spearmanr(gamma/denom,gamma)[0]:.3f}')
    if SAVEFIG:
        plt.savefig(f'{C.session_name}-{C.block_name}_2Dhist_gamma.svg')
    plt.show()
    

def plot_theta_power(nf_session: Neurofeedback_Session):
    """
    Show theta power in Main and Transfer in different targets over trials.
    """
    M = nf_session.M
    T = nf_session.T
    
    # Panel B
    spectrogram_C = nf_session.find_spectrogram('Control')
    avg_theta = spectrogram_C[:,1:3].sum(1).mean(0) 
    total_df = []
    for block in [M,T]:
        spectrogram_nf = nf_session.find_spectrogram(block.block_name)
        df = block.times_table
        
        for targ in targ_label:
            align_pts = df[df.target==targ].reward.values 
            psd = np.zeros((len(align_pts),26))
            for j in range(len(align_pts)):
                psd[j]= spectrogram_nf[align_pts[j]+12:align_pts[j]+30].mean(0)
            power = psd[:,1:3].sum(1)/avg_theta*100  # Normalized theta power for each trial
        
            total_df.append(pd.DataFrame({'block':block.block_name,
                                          'target':targ,
                                          'power':slide_avg(power,20),
                                          'index':np.arange(len(power))}))
    
    total_df = pd.concat(total_df)
    sns.relplot(data=total_df,x='index',y='power',hue='block',palette=['purple','g'],
                kind='line',col='target',height=3,aspect=0.8)
    plt.suptitle(nf_session.session,y=1.05)
    if SAVEFIG:
        plt.savefig(f'{nf_session.session}-RPE.svg')
    plt.show()


def plot_theta_power_change(nf_all: Neurofeedback_All, show_stats: bool = True):
    """
    Theta power change in % compared to Control blocks.
    Excluding two werid sessions in each subject. 
    """
    
    if nf_all is None: return 

    total_df = []
    
    use_session = [T.session_name for T in nf_all.Ts if T.session_name not in ['braz20230425','airp20231221']]
    
    for session in use_session:
        nf_session = getattr(nf_all, session)
        
        spectrogram_ref = nf_session.find_spectrogram('Control')[:,1:3].sum(1).mean(0)
        spectrogram_main = nf_session.find_spectrogram('Main')
        spectrogram_transfer = nf_session.find_spectrogram('Transfer')
        
        M_df = nf_session.M.times_table
        T_df = nf_session.T.times_table
        
        for targ in targ_label:
            align_pts = M_df[M_df.target==targ].reward.values
            psd = np.zeros((len(align_pts),26))
            for j in range(len(align_pts)):
                psd[j]= spectrogram_main[align_pts[j]+12:align_pts[j]+30].mean(0)
            Mp = psd[:,1:3].sum(1).mean(0)/spectrogram_ref  # Normalized beta for each trial
        
            align_pts = T_df[T_df.target==targ].reward.values
            psd = np.zeros((len(align_pts),26))
            for j in range(len(align_pts)):
                psd[j]= spectrogram_transfer[align_pts[j]+12:align_pts[j]+30].mean(0)
            Tp = psd[:,1:3].sum(1).mean(0)/spectrogram_ref  # Normalized beta for each trial
        
            total_df.append(pd.DataFrame({'session':nf_session.session,
                                          'target':targ,
                                          'ratio':[Tp/Mp]}))
    total_df = pd.concat(total_df).reset_index()
    
    plt.figure(figsize=(4,3))
    ax = sns.barplot(x="target", y="ratio", data=total_df,palette=colors,
                     alpha=0.5,errorbar='se')
    sns.stripplot(x="target", y="ratio", data=total_df,ax=ax,palette=colors,
                  edgecolor="black",linewidth=1,dodge=True,legend=False)
    if SAVEFIG:
        plt.savefig(f'{nf_all.subject}_RPE_ratio.svg')
    plt.show()
    
    if show_stats:
        for targ in targ_label:
            d = total_df[total_df.target==targ].ratio
            res = pg.ttest(d,1,alternative='greater')
            print(res[['p-val','cohen-d']])
            
        print(pg.anova(data=total_df,dv='ratio',between='target'))
        print()


def plot_theta_power_success_failed(nf_session: Neurofeedback_Session, show_stats: bool = True):
    """
    Theta power separated from successful and failed holds in Main and Transfer.
    """
    M = nf_session.M
    T = nf_session.T
    t = Time_align(1)
    
    for block in [M,T]:
        df = pd.DataFrame({'state':block.state[:-1],'time':block.state_time[:-1]})
        df['next_state'] = block.state[1:]
    
        # If the next state was lfp_target, meaning the hold-target failed.
        align_pts = df[(df.state==b'lfp_hold')&(df.next_state==b'lfp_target')].time.values
        data_failed = nf_session.find_aligned(block.block_name, 
                                              'theta', 
                                              None, 
                                              align_pts, 
                                              t)
        
        # If the next state was reward, then it succeeded.
        align_pts = df[(df.state==b'lfp_hold')&(df.next_state==b'reward')].time.values
        data_success = nf_session.find_aligned(block.block_name, 
                                              'theta', 
                                              None, 
                                              align_pts, 
                                              t)
    
        targ_avg_f = np.nanmean(data_failed, axis=0)
        targ_sem_f = np.nanstd(data_failed, axis=0)/np.sqrt(len(data_failed)-1)
        targ_avg_s = np.nanmean(data_success, axis=0)
        targ_sem_s = np.nanstd(data_success, axis=0)/np.sqrt(len(data_success)-1)
    
        plt.figure(figsize=(4,3))
        plt.plot(t.time, slide_avg(targ_avg_f,10),c='b',label='Failed')
        plt.fill_between(t.time, 
                         slide_avg(targ_avg_f,10)-targ_sem_f,
                         slide_avg(targ_avg_f,10)+targ_sem_f, 
                         color='b',alpha=0.4)
        plt.plot(t.time, slide_avg(targ_avg_s,10),c='r',label='Successful')
        plt.fill_between(t.time, 
                         slide_avg(targ_avg_s,10)-targ_sem_s,
                         slide_avg(targ_avg_s,10)+targ_sem_s, 
                         color='r',alpha=0.4)
    
        plt.title(f'{nf_session.session}-{block.block_name}, theta band')
        plt.legend(frameon=False,loc='upper left')
        plt.ylim([10,40])
        if SAVEFIG:
            plt.savefig(f'{nf_session.session}-{block.block_name}-aligned_theta_SuccessFailed.svg')
        plt.show()
        
        
        if show_stats:
             # Avg power from the starting of reward to 0.5 sec after reward, aka 0.2-0.7 sec from hold
            failed = data_failed[:,60+12:60+42].mean(1)
            success = data_success[:,60+12:60+42].mean(1)
            res = pg.ttest(failed,success)
            print(res[['p-val','cohen-d']])
            
            
def plot_washout_control_change(nf_all: Neurofeedback_All, show_stats: bool = True):
    """
    Washout and Control
    """
    
    if nf_all is None: return 

    # Make sure all chosen sessions have washout blocks
    nf_sessions = [getattr(nf_all, session) for session in nf_all.session_names]
    Cs = [nf_session.C for nf_session in nf_sessions if nf_session.has_washout]
    Ws = [nf_session.W for nf_session in nf_sessions if nf_session.has_washout]
    
    # Calculate % change in individual bands
    bands = ['delta','theta','alpha','beta','gamma']
    df_band = []
    for metric in bands:
        mean = [nonan(Ws[i].profile(metric)).mean() / nonan(Cs[i].profile(metric)).mean() for i in range(len(Cs))]
        df_band.append(pd.DataFrame({'change':mean,'metric':metric}))
    df_band = pd.concat(df_band)
    
    # Calculate % change in NBP
    mean = [nonan(Ws[i].profile('fraction')).mean() / nonan(Cs[i].profile('fraction')).mean() for i in range(len(Cs))]
    df_nbp = pd.DataFrame({'change':mean,'metric':'fraction'})
    
    # Fig for bands
    ax=sns.barplot(data=df_band,x='metric',y='change',edgecolor="black",linewidth=1)
    sns.stripplot(data=df_band,x='metric',y='change', ax=ax,edgecolor="black",linewidth=1)
    plt.ylabel('Power change (W/C) in folds')
    if SAVEFIG:
        plt.savefig(f'{nf_all.subject}-WashoutChange_power.svg')
    plt.show()
    
    # Fig for NBP
    plt.figure(figsize=(1.5,3))
    ax=sns.barplot(data=df_nbp,x='metric',y='change',palette=['w'],edgecolor="black",linewidth=1)
    sns.stripplot(data=df_nbp,x='metric',y='change', ax=ax,palette=['w'],edgecolor="black",linewidth=1)
    plt.ylabel('NBP change (W/C) in folds')
    if SAVEFIG:
        plt.savefig(f'{nf_all.subject}-WashoutChange_nbp.svg')
    plt.show()
    
    if show_stats:
        for metric in ['beta','gamma']:
            data = df_band[df_band.metric==metric]['change']
            res = pg.ttest(data,1,alternative='greater')
            print(f'n={len(data)}', res[['p-val','cohen-d']])
    
        data = df_nbp[df_nbp.metric=='fraction']['change']
        res = pg.ttest(data,1,alternative='greater')
        print(f'n={len(data)}', res[['p-val','cohen-d']])
        

def plot_variance(nf_all: Neurofeedback_All, show_stats: bool = True):
    """
    Variance of each block
    """
    
    if nf_all is None: return 

    v_c,v_m,v_t,v_w = [],[],[],[]
    for session in nf_all.session_names:
        nf_session: Neurofeedback_Session = getattr(nf_all, session)
        C = nf_session.C
        M = nf_session.M
        T = nf_session.T if nf_session.has_transfer else None
        W = nf_session.W if nf_session.has_washout else None
        
        
        v_c.append(np.nanstd(C.profile('fraction')))
        v_m.append(np.nanstd(M.profile('fraction')))
        if T is not None:
            v_t.append(np.nanstd(T.profile('fraction')))
        if W is not None:
            v_w.append(np.nanstd(W.profile('fraction')))
            
    block_names = ['Control','Main','Transfer','Washout']
    variances = [v_c,v_m,v_t,v_w]
    df = []
    for i in range(4):
        df.append(pd.DataFrame({'var':variances[i],'block':block_names[i]}))
    df = pd.concat(df)
    
    plt.figure(figsize=(4,3))
    sns.barplot(data=df,x='block',y='var',color='w',edgecolor='k',capsize=0.05)
    plt.ylabel('Variance of NBP in a block')
    plt.title(nf_all.subject)
    # plt.ylim([0,0.16])
    if SAVEFIG:
        plt.savefig(f'{nf_all.subject}-CrossBlock_Variance.svg')
    plt.show()
    
    if show_stats:
        print(pg.anova(data=df,dv='var',between='block'))
        print('\n\n')
        print(pg.pairwise_tukey(data=df,dv='var',between='block'))


def plot_compare_raw_zscore_nbp(nf_all: Neurofeedback_All):
    """
    Compare raw, z-scored, and normalized beta power
    """

    if nf_all is None: return 

    Cs = nf_all.Cs
    
    nbp_data = [nonan(C.profile('fraction')) for C in Cs]
    raw_data,zdata = [],[]
    session_name = [C.session_name for C in Cs]
    
    for C in Cs:
        power = C.lfp_power.reshape(C.lfp_power.shape[0],129,3,order='F').mean(2)
        beta = nonan(power[:,6:9].sum(1))   
        raw_data.append(beta)
        zdata.append(stats.zscore(beta))
    
    plt.figure(figsize=(8,5))
    plt.subplot(131)
    plt.boxplot(raw_data[::-1],vert=False,whis=(5,95),showfliers=False)
    plt.yticks(np.arange(len(Cs),0,-1),session_name)
    # plt.xlim([0,40])
    plt.xscale('log')
    plt.xlabel('Raw beta power')
    plt.subplot(132)
    plt.boxplot(zdata[::-1],vert=False,whis=(5,95),showfliers=False)
    plt.yticks([])
    # plt.xscale('symlog')
    # plt.xlim([-50,50])
    plt.xlabel('Z-scored beta power')
    plt.subplot(133)
    plt.boxplot(nbp_data[::-1],vert=False,whis=(5,95),showfliers=False)
    plt.yticks([])
    plt.xlim([0,1])
    plt.xlabel('Normalized beta power')
    if SAVEFIG:
        plt.savefig(f'{nf_all.subject}-{nf_all.block_name}_Boxplot.svg')
        plt.savefig(f'{nf_all.subject}-{nf_all.block_name}_Boxplot.png')
    plt.show()
    
    r_skew = [np.abs(stats.skew(d)) for d in raw_data]
    z_skew = [np.abs(stats.skew(d)) for d in zdata]
    nbp_skew = [np.abs(stats.skew(d)) for d in nbp_data]
    
    if nf_all.subject == 'braz':
        r_skew.pop(7)
        z_skew.pop(7)
        nbp_skew.pop(7)
        
    d_raw = pd.DataFrame({'Skewness':r_skew,'Type':'Raw'})
    d_zscore = pd.DataFrame({'Skewness':z_skew,'Type':'Z-scored'})
    d_nbp = pd.DataFrame({'Skewness':nbp_skew,'Type':'NBP'})
    df = pd.concat([d_raw,d_zscore,d_nbp])
    plt.figure(figsize=(4,3))
    ax = sns.barplot(x="Type", y="Skewness", data=df, 
          edgecolor="black",errcolor="black",errwidth=1.5,capsize = 0.05,alpha=0.5)
    sns.stripplot(x="Type", y="Skewness", data=df, ax=ax,
                  edgecolor="black",linewidth=1,alpha=0.5,legend=False)
    plt.show()
    
    print(len(df))
    print(pg.anova(data=df, dv='Skewness', between='Type'))
    print(pg.pairwise_tukey(data=df, dv='Skewness', between='Type'))


def __plot_correlation_RPE_next_trial_duration(nf_session: Neurofeedback_Session):
    """
    Potential correlation between RPE and next trial duratiaon
    """
    M = nf_session.M
    T = nf_session.T
    
    fig,ax = plt.subplots(2,1,sharex=True,figsize=(4,6))
    for i,block in enumerate([M,T]):
        df = block.times_table
        theta = nf_session.find_aligned(block.block_name, 'theta', None, 'reward', Time_align(1))
        theta = theta[:,60:60+30].sum(1)
        theta = np.insert(theta[:-1],0,0)
        df['theta'] = theta
        sns.scatterplot(df,x='time',y='theta',hue='target',ax=ax[i],hue_order=['High','Center','Low'])
    plt.show()


def plot_main_transfer_strategy_correlation(nf_session: Neurofeedback_Session):
    """
    The strategies in Main and Transfer blocks.
    """    
    def linear_model(data_df):
        """
        Linear model to be fitted
        """
        Y = data_df[[f'beta{ad}']].values
        X = data_df[[f'gamma{ad}']].values
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
        print(f'R2: {results.rsquared:.3f}, p-val: {results.pvalues[1]}')
        return results.params
        
    C = nf_session.C
    c_beta,c_gamma = C.profile('beta').mean(),C.profile('gamma').mean()
    
    ad = '_adj' # For adjusted band power
    temp = []    
    blocks = ['Main','Transfer']
    for block in blocks:
        block_data = nf_session.get_block_data(block)
        
    
        beta  = slide_avg(block_data.profile('beta'),5)
        gamma = slide_avg(block_data.profile('gamma'),5)
    
        b,g = [],[]
        for i in range(len(block_data.times_table)):
            align_pts = block_data.times_table.iloc[i]['reward']
            b.append(beta[align_pts-12:align_pts].mean())
            g.append(gamma[align_pts-12:align_pts].mean())
    
        temp_df = block_data.times_table[['target','time']]
        temp_df['block'] = block
        temp_df['beta'] = b
        temp_df['gamma'] = g
        temp_df['beta' + ad] = (temp_df['beta'] - c_beta) / c_beta * 100
        temp_df['gamma' + ad] = (temp_df['gamma'] - c_gamma) / c_gamma * 100
        temp.append(temp_df)
        
    df = pd.concat(temp)    
    for targ in ['High','Low']:
        sns.relplot(data=df[df.target==targ],
                    x=f'gamma{ad}',
                    y=f'beta{ad}',
                    col='block',kind='scatter',linewidth=0.5,ec='k',
                    height=3,aspect=0.9,palette=['r','g'],alpha=1)
        g = df[df.target==targ][[f'beta{ad}',f'gamma{ad}','block']].groupby('block')
        g_mean = g.mean()
        print(targ)
        for j in range(2):
            block = blocks[j]
            print(block)
            plt.subplot(1,2,j+1)
            intercept, slope = g.apply(linear_model)[block]
            print(f'Slope: {np.round(slope,2)}')
            x_min = g.min().iloc[j]['gamma_adj']
            x_max = g.max().iloc[j]['gamma_adj']
            plt.plot([x_min,x_max],[x_min*slope+intercept,x_max*slope+intercept])
            plt.scatter(g_mean[f'gamma{ad}'],g_mean[f'beta{ad}'],c=['r','g'],
                        marker='^',s=90,ec='k',linewidth=0.5)
            plt.axvline(0,c='k',ls='--')
            plt.axhline(0,c='k',ls='--')
            plt.xlabel('∆ gamma')
            plt.ylabel('∆ beta')
        if SAVEFIG:
            plt.savefig(f'{nf_session.session}-strategy_{targ}.svg')
        plt.show()
    
    
def plot_main_transfer_strategy_entire(nf_all: Neurofeedback_All):
    """
    The overall consistent strategy across all sessions.
    """
    
    if nf_all is None: return 

    def custom_fx(data_df, ad: str = '_adj', metric: str = 'correlation'):
        
        m_b = np.mean(data_df[f'beta{ad}'])
        m_g = np.mean(data_df[f'gamma{ad}'])
        
        match metric:
            case 'ratio':
                return m_b / m_g ## Take ratio of beta and gamma
            case 'distance':
                return np.sqrt(m_b**2 + m_g**2) ## Take distance
            case 'angle':
                return np.arctan2(m_b,m_g) ## Take angle
            case 'correlation':
                return np.corrcoef(data_df[f'beta{ad}'],data_df[f'gamma{ad}'])[0,1] ## Take correlation
            case _:
                raise ValueError('Unsupported metric.')
    
    # For all blocks that have Transfer
    temp = []
    ad = '_adj'
    blocks = ['Main','Transfer']
    for session in nf_all.session_names:
        nf_session = getattr(nf_all, session)
        if nf_session.has_transfer:
            C = nf_session.C
            c_beta,c_gamma = C.profile('beta').mean(),C.profile('gamma').mean()
            for block in blocks:
                block_data = nf_session.get_block_data(block)
                beta  = slide_avg(block_data.profile('beta'),5)
                gamma = slide_avg(block_data.profile('gamma'),5)
            
                b,g = [],[]
                for i in range(len(block_data.times_table)):
                    align_pts = block_data.times_table.iloc[i]['reward']
                    b.append(beta[align_pts-12:align_pts].mean())
                    g.append(gamma[align_pts-12:align_pts].mean())
            
                temp_df = block_data.times_table[['target','time']]
                temp_df['block'] = block
                temp_df['session'] = session
                temp_df['beta'] = b
                temp_df['gamma'] = g
                temp_df['beta' + ad] = (temp_df['beta'] - c_beta) / c_beta * 100
                temp_df['gamma' + ad] = (temp_df['gamma'] - c_gamma) / c_gamma * 100
                temp.append(temp_df)
            df = pd.concat(temp)    
    
    name = 'correlation'
    for targ in targ_label:
        sub_df = df[df.target==targ][[f'beta{ad}',f'gamma{ad}','session','block']] \
            .groupby(['session','block']) \
            .apply(custom_fx) \
            .to_frame(name=name).reset_index()
        plt.figure(figsize=(4,3))
        ax = sns.barplot(data=sub_df,y=name,x='block',
                         palette=['purple','g'],alpha=0.2)
        sns.stripplot(data=sub_df,y=name,x='block',
                      palette=['purple','g'],
                      edgecolor="black",linewidth=0.5,dodge=True,ax=ax)
        if SAVEFIG:
            plt.savefig(f'airp_braz_{name}_strategy_{targ}.svg')
        plt.show()
    
        main_data = sub_df[sub_df.block=='Main'][name].values
        transfer_data = sub_df[sub_df.block=='Transfer'][name].values
        test = pg.ttest(main_data,transfer_data,paired=True)[['p-val','cohen-d']]
        print(f'{nf_all.subject} {targ} sample number: Main-{len(main_data)}, Transfer-{len(transfer_data)}')
        print(test)
        

def R1_plot_trial_by_trial_strategy(nf_all: Neurofeedback_All, selected: bool = True):
    """
    Differential regulation over trials for individual sessions
    """
    
    if nf_all is None: return 

    # For each session
    t = Time_align(1)
    
    if selected: # Use selected sessions
        if nf_all.subject == 'airp':
            selection = [nf_all.airp20240320.session]
        else:
            selection = [nf_all.braz20230404.session]
    else: # Use all the sessions
        selection = [nf_all.session_names]
    
    for session in selection:
        nf_session = getattr(nf_all, session)
        spectrogram_C = nf_session.find_spectrogram('Control')
        fig, ax = plt.subplots(2, 2, figsize=(8,8), sharex=True)
        for j, block in enumerate(['Main','Transfer']):
            if block=='Transfer' and not nf_session.has_transfer:
                continue
            for i, band in enumerate(['beta','gamma']):
                ctrl_avg = spectrogram_C[:,slice(*nf_session.C.index(band))].sum(1).mean(0) # The avg power in Control
                for targ in targ_label:
                    targ_data = nf_session.find_aligned(block=block,
                                                        band=band,
                                                        target=targ,
                                                        behavioral_metric='reward',
                                                        T=t)
                    
                    data = slide_avg(nonan(targ_data[:,48:60].mean(1)), 20) / ctrl_avg * 100 - 100
                    ax[i,j].plot(data, c=colors[targ])
                ax[i,j].set_title(f'{nf_session.session}-{block}-{band}')
                ax[i,j].set_ylabel('Power change (%) compared to Control')
                ax[i,j].set_xlabel('Trial number')
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        if SAVEFIG:
            plt.savefig(f'{nf_all.subject}_trial-by-trial_strategy.svg')
        
        plt.show()
        
    # Session-average of differential regulation strategy
    # For main block only.
    for j, band in enumerate(['beta','gamma']):
        
        result = {'High': np.zeros((len(nf_all.session_names), 100)),
                  'Center': np.zeros((len(nf_all.session_names), 100)),
                  'Low': np.zeros((len(nf_all.session_names), 100))}
        
        for i, session in enumerate(nf_all.session_names):
            nf_session = getattr(nf_all, session)
            spectrogram_C = nf_session.find_spectrogram('Control')
        
            ctrl_avg = spectrogram_C[:,slice(*nf_session.C.index('beta'))].sum(1).mean(0) # The avg power in Control
            for targ in targ_label:
                targ_data = nf_session.find_aligned(block='Main',
                                                    band=band,
                                                    target=targ,
                                                    behavioral_metric='reward',
                                                    T=t)
                data = slide_avg(nonan(targ_data[:,48:60].mean(1)), 20) / ctrl_avg * 100 - 100
                result[targ][i,:min(100, len(data))] = data[:min(100, len(data))]
        
        
        plt.figure(figsize=(4,4))
        for targ in targ_label:
            combined = result[targ]
            avg = np.mean(combined,0)
            std = np.std(combined,0) / np.sqrt(len(combined)-1)
            plt.plot(range(100),avg,c=colors[targ],label=targ)
            plt.fill_between(range(100),
                             avg+std,avg-std,
                             alpha=0.2,color=colors[targ])
        plt.xlabel('Trial')
        plt.ylabel('Power change (%) compared to Control')
        plt.title(f'{nf_all.subject}-{band}')
        plt.show()


def plot_nbp_distribution_all_blocks(nf_session: Neurofeedback_Session):
    """
    Demonstrate NBP distribution of all blocks within a session.
    """
    C = nf_session.C
    M = nf_session.M
    T = nf_session.T
    W = nf_session.W
    
    data = [nonan(block.profile('fraction')) for block in [C,M,T,W]]
    profile = np.zeros((len(data),3))
    for i in range(len(data)):
        profile[i] = np.percentile(data[i],[95,50,5])
        
    overlap = 0.1
    n_points = 300
    xx = np.linspace(0,1,n_points)
    
    title = f'{nf_session}_nbp_stability_over_block'
    
    fig,ax = plt.subplots(figsize=(4,3))
    for i, d in enumerate(data):
        pdf = gaussian_kde(d)
        y = -i*(1.0-overlap)
        curve = pdf(xx)
        plt.fill_between(xx,np.ones(n_points)*y,curve+y, color='b', alpha=1)
        plt.plot(xx, curve+y, c='k')
    plt.xlabel('Normalized Beta Power')
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylim([y,None])
    plt.title(title)
    if SAVEFIG:
        plt.savefig(f'{title}.svg')
    plt.show()
    
    fig,ax = plt.subplots(figsize=(1,1))
    plt.eventplot(profile,lineoffsets=-1, colors='b', linewidths=5)
    plt.xlim([0,1])
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xticks([0,0.5,1])
    if SAVEFIG:
        plt.savefig(f'{title}_loc.svg')
    plt.show()

#%% 
# Figure 1: Experimental setup (No plotting needed)

# Figure 2: Normalized beta power is superior to raw or z-scored beta power
plot_compare_raw_zscore_nbp(airp) # Panel C
plot_compare_raw_zscore_nbp(braz) # Panel D

# Figure 3: Optimizing NBP using rank correlations between NBP and beta power
plot_correlation_NBP_and_power(braz.braz20230404)

# Figure 4: NBP and target locations
plot_nbp_pdf_target_location(braz.braz20230404) # Panel A
plot_NBP_SegmentControl(braz.braz20230404.C) # Panel B

# Figure 5: Successful volitional regulation and improvement of VTA NBP with training
plot_time_smoothed_trial_duration(airp, 'Main') # Panel A left
plot_time_smoothed_trial_duration(braz, 'Main') # Panel A right
plot_time_smoothed_trial_duration_combined(airp) # Panel A left inset
plot_time_smoothed_trial_duration_combined(braz) # Panel A left inset
plot_max_improvement(airp, block='Main') # Panel B left
plot_max_improvement(braz, block='Main') # Panel B right
plot_example_nbp(braz.braz20230404) # Panel C
plot_aligned_lfp(braz.braz20230404, 'Main', 'fraction', 'reward') # Panel D

# Figure 6: Rewards per minute in each block
plot_reward_rate(airp) # Panel A
plot_reward_rate(braz) # Panel B

# Figure 7: Differential contribution of frequency bands to volitional VTA regulation
# ================ PANEL NUMBER WRONG ================
plot_PSD(braz.braz20230404)
plot_algined_spectrogram(braz.braz20230404, 'Main', 500) # Panel A
plot_power_change_percentage(braz.braz20230404) # Panel B
plot_aligned_lfp(braz.braz20230404, 'Main', 'beta', 'reward') # Panel C
plot_aligned_lfp(braz.braz20230404, 'Main', 'gamma', 'reward') # Panel C
plot_aligned_lfp(braz.braz20230404, 'Main', 'beta', 'wait') # Panel C
plot_aligned_lfp(braz.braz20230404, 'Main', 'gamma', 'wait') # Panel C

# Figure 8: Stability of powers other than beta and gamma frequency bands during regulation
plot_power_change_percentage(braz.braz20230404) # Panel B and Figure 8

# Figure 9: Regulation transfers to occluded cursor condition and improves over time
plot_time_smoothed_trial_duration(airp, 'Transfer') # Panel A left
plot_time_smoothed_trial_duration(braz, 'Transfer') # Panel A right
plot_time_smoothed_trial_duration_combined(airp) # Panel A left inset
plot_time_smoothed_trial_duration_combined(braz) # Panel A left inset
plot_max_improvement(airp, block='Transfer') # Panel B left
plot_max_improvement(braz, block='Transfer') # Panel B right
plot_aligned_lfp(braz.braz20230404, 'Transfer', 'fraction', 'reward') # Panel C
plot_algined_spectrogram(braz.braz20230404, 'Transfer', 500) # Panel D
plot_power_change_percentage(braz.braz20230404) # Panel E
plot_aligned_lfp(braz.braz20230404, 'Transfer', 'beta', 'reward') # Panel F
plot_aligned_lfp(braz.braz20230404, 'Transfer', 'gamma', 'reward') # Panel F
plot_aligned_lfp(braz.braz20230404, 'Transfer', 'beta', 'wait') # Panel F
plot_aligned_lfp(braz.braz20230404, 'Transfer', 'gamma', 'wait') # Panel F
plot_main_transfer_strategy_correlation(braz.braz20230404) # Panel G
plot_main_transfer_strategy_entire(airp) # Panel H
plot_main_transfer_strategy_entire(braz) # Panel H

# Figure 10: Theta power in the Transfer blocks shows positive RPE after reward
plot_aligned_lfp(braz.braz20230404, 'Main', 'theta', 'reward') # Panel A
plot_aligned_lfp(braz.braz20230404, 'Transfer', 'theta', 'reward') # Panel A
plot_aligned_lfp(braz.braz20230404, 'Main', 'theta', 'wait') # Panel A
plot_aligned_lfp(braz.braz20230404, 'Transfer', 'theta', 'wait') # Panel A
plot_theta_power(braz.braz20230404) # Panel B
plot_theta_power_change(airp) # Panel C left
plot_theta_power_change(braz) # Panel C right
plot_theta_power_success_failed(braz.braz20230404) # Panel D


# Figure 11: Changes in beta and gamma power and NBP after neurofeedback training.
plot_washout_control_change(airp) # Panel A left and Panel B left
plot_washout_control_change(braz) # Panel A right and Panel B right

# Figure 12:  Distribution and variance of NBP in the four blocks
plot_nbp_distribution_all_blocks(braz.braz20230411)
plot_variance(airp) # Panel B left
plot_variance(braz) # Panel B right

# Unused 
__plot_correlation_RPE_next_trial_duration(braz.braz20230404)

# Analysis for revision 1
R1_plot_trial_by_trial_strategy(braz)

















#%% Spike and LFP

BMI_FOLDER = '/Users/hungyunlu/Library/CloudStorage/Box-Box/Hung-Yun Lu Research File/Projects/bmi_python'
NSX_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'blackrock')
NS_FOLDER = os.path.join(BMI_FOLDER, 'riglib', 'ripple', 'pyns', 'pyns')
os.chdir(NSX_FOLDER)
from brpylib import NsxFile
os.chdir(DATA_FOLDER)

prefix = os.path.join(DATA_FOLDER, 'braz20230404', 'braz20230404_06_te3743')
ns2file = NsxFile(prefix + '.ns2')
data = ns2file.getdata()['data'][3:6] # In this session, channel 3-5 were used.

#%% TCR

threshold =  - 4.5 * np.sqrt(np.mean(data[0,:5000]**2))
pseudo_spike = np.array(data[0,:100000] < threshold, dtype=int)
plt.eventplot(np.where(pseudo_spike>0)[0])
plt.show()

#%%
from scipy.signal import butter, filtfilt

def spiking_band_power(data, lowcut, highcut, fs, order=5):
    """
    Apply a band-pass filter to the input data.

    Parameters:
        data (array-like): The input signal to filter.
        lowcut (float): The lower cutoff frequency for the band-pass filter (Hz).
        highcut (float): The upper cutoff frequency for the band-pass filter (Hz).
        fs (float): The sampling frequency of the signal (Hz).
        order (int): The order of the filter. Default is 5.

    Returns:
        array-like: The filtered signal.
    """
    # Design the band-pass filter
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist  # Normalize the lower cutoff frequency
    high = highcut / nyquist  # Normalize the upper cutoff frequency
    b, a = butter(order, [low, high], btype='band', analog=False)

    # Apply the filter to the data
    filtered_data = filtfilt(b, a, data)
    return filtered_data

filt = spiking_band_power(data[0], 400 , 499, 1000)
plt.plot(filt[:10000])
plt.show()
















