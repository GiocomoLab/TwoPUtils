from random import randrange

import numpy as np
import scipy as sp

from sklearn.impute import KNNImputer as knn_imp

#import utilities as u
from . import utilities as u

def trial_matrix(arr_in, pos_in, tstart_inds, tstop_inds, bin_size=10, min_pos = 0,
                 max_pos=450, speed=None, speed_thr=2, perm=False,
                 mat_only=False, impute_nans = False, use_sum=False,
                only_spatial_binning = False):
    """

    :param arr: timepoints x anything array to be put into trials x positions format
    :param pos: position at each timepoint
    :param tstart_inds: indices of trial starts
    :param tstop_inds: indices of trial stops
    :param bin_size: spatial bin size in cm
    :param max_pos: maximum position on track
    :param speed: vector of speeds at each timepoint. If None, then no speed filtering is done
    :param speed_thr: speed threshold in cm/s. Timepoints of low speed are dropped
    :param perm: bool. whether to circularly permute timeseries before binning. used for permutation testing
    :param mat_only: bool. return just spatial binned data or also occupancy, bin edges, and bin bin_centers
    :return: if mat_only
                    trial_mat - position binned data
             else
                    trial_mat
                    occ_mat - trials x positions matrix of bin occupancy
                    bin_edges - position bin edges
                    bin_centers - bin centers
    """

    arr = np.copy(arr_in)
    pos = np.copy(pos_in)

    ntrials = tstart_inds.shape[0]
    if speed is not None:  # mask out speeds below speed threshold
        pos[speed < speed_thr] = -1000
        arr[speed < speed_thr, :] = np.nan

    # make position bins
    bin_edges = np.arange(min_pos, max_pos + bin_size, bin_size)
    bin_centers = bin_edges[:-1] + bin_size / 2
    bin_edges = bin_edges.tolist()

    # if arr is a vector, expand dimension
    if len(arr.shape) < 2:
        arr = arr[:, np.newaxis]
        # arr = np.expand_dims(arr, axis=1)
    
    if not only_spatial_binning:
        trial_mat = np.zeros([int(ntrials), len(bin_edges) - 1, arr.shape[1]])
        trial_mat[:] = np.nan
        occ_mat = np.zeros([int(ntrials), len(bin_edges) - 1])
        for trial in range(int(ntrials)):  # for each trial
            # get trial indices
            firstI, lastI = tstart_inds[trial], tstop_inds[trial]

            arr_t, pos_t = arr[firstI:lastI, :], pos[firstI:lastI]

            if perm:  # circularly permute if desired
                ## shift by a minumum of 1 s (15 samples at 15 Hz), maximum length of the trial
                pos_t = np.roll(pos_t, np.random.randint(15,high=pos_t.shape[0]))
                #arr_t = np.roll(arr_t, np.random.randint(80,high=arr_t.shape[0]),axis=0)

            # average within spatial bins
            for b, (edge1, edge2) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                if np.where((pos_t > edge1) & (pos_t <= edge2))[0].shape[0] > 0:
                    if use_sum:
                        trial_mat[trial, b] = np.nansum(arr_t[(pos_t > edge1) & (pos_t <= edge2), :], axis=0)
                    else:
                        trial_mat[trial, b] = np.nanmean(arr_t[(pos_t > edge1) & (pos_t <= edge2), :], axis=0)
                    # occ_mat[trial, b] = np.where((pos_t > edge1) & (pos_t <= edge2))[0].shape[0]
                    ## Counts the samples where the mouse was within the position bin and neural activity was not nan
                    occ_mat[trial, b] = (1-np.isnan(arr_t[(pos_t > edge1) & (pos_t <= edge2),0])).sum()
                else:
                    pass
    else:
        trial_mat = np.zeros([len(bin_edges) - 1, arr.shape[1]])
        trial_mat[:] = np.nan
        occ_mat = np.zeros([len(bin_edges) - 1, 1])
        
        if perm:  # circularly permute if desired
            ## shift by a minumum of 1 s (15 samples at 15 Hz), maximum length of the trial
            pos = np.roll(pos, np.random.randint(15,high=pos.shape[0]))
            #arr_t = np.roll(arr_t, np.random.randint(80,high=arr_t.shape[0]),axis=0)

        # average within spatial bins
        for b, (edge1, edge2) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if np.where((pos > edge1) & (pos <= edge2))[0].shape[0] > 0:
                if use_sum:
                    trial_mat[b,:] = np.nansum(arr[(pos > edge1) & (pos <= edge2), :], axis=0)
                else:
                    trial_mat[b, :] = np.nanmean(arr[(pos > edge1) & (pos <= edge2), :], axis=0)
                # occ_mat[trial, b] = np.where((pos_t > edge1) & (pos_t <= edge2))[0].shape[0]
                ## Counts the samples where the mouse was within the position bin and neural activity was not nan
                occ_mat[b] = (1-np.isnan(arr[(pos > edge1) & (pos <= edge2),0])).sum()
            else:
                pass

        
        
    if impute_nans:

        for trial in range(trial_mat.shape[0]):
            nan_inds = np.isnan(trial_mat[trial,:,0])
            _c = bin_centers[~nan_inds]
            for cell in range(trial_mat.shape[2]):
                _m = trial_mat[trial, ~nan_inds, cell]
                trial_mat[trial,:,cell] = np.interp(bin_centers, _c, _m)

    if mat_only:
        return np.squeeze(trial_mat)
    elif not mat_only and not only_spatial_binning:
        return np.squeeze(trial_mat), np.squeeze(occ_mat / (occ_mat.sum(axis=1)[:, np.newaxis] + 1E-3)), bin_edges, bin_centers
    elif not mat_only and only_spatial_binning:
        return np.squeeze(trial_mat), (occ_mat / (occ_mat.sum() + 1E-3)).T, bin_edges, bin_centers


def spatial_info(frmap,occupancy):
    '''calculate spatial information bits/spike
    inputs: frmap - [spatial bins, cells] spatially binned activity rate for each cell
            occupancy - [spatial bins,] fractional occupancy of each spatial bin
    outputs: SI - [cells,] spatial information for each cell '''
    
    ### vectorizing
    P_map = frmap - np.amin(frmap)+.001 # make sure there's no negative activity rates
    # print((P_map<0).sum())
    mean_rate = (P_map * occupancy[:,np.newaxis]).sum(axis=0,keepdims=True)

    P_map_norm = P_map/mean_rate #np.nanmean(P_map, axis=0,keepdims=True)
    SI = np.nansum((P_map_norm*occupancy[:,np.newaxis])*np.log2(P_map_norm), axis=0) # Skaggs and McNaughton spatial information

    return SI

def spatial_info_tank(ts,frmap,occ,speed=None):
    
    arr = np.copy(ts)
    arr[speed < 2, :] = np.nan
    arr = arr - np.nanmin(arr)+.001
    # f_bar = np.nanmean(arr,axis=0, keepdims=True)
    
    f_i = frmap - np.amin(frmap)+.001  # just make all values non-negative
    # f_bar = np.nanmean(f_i, axis=0, keepdims=True)
    f_bar = (f_i * occ[:,np.newaxis]).sum(axis=0,keepdims=True)
    
    SI = np.nansum((occ[:, np.newaxis] * f_i) *
                    np.log2(f_i / f_bar), axis=0)
    
    
    return SI


def place_cells_calc(C, position, tstart_inds,
                     teleport_inds, pthr = .05, speed=None, nperms = 100, 
                     output_shuffle = False, use_tank_method=False, **kwargs):
    '''Find cells that have significant spatial information info. Use bootstrapped estimate of each cell's
    spatial information to minimize effect of outlier trials
    inputs:C - [timepoints, neurons] activity rate/dFF over whole session
            position - [timepoints,] animal's position on the track at each timepoint
            trial_info - trial information dict from u.by_trial_info()
            tstart_inds - [ntrials,] array of trial start times (can filter by which trials you want to
                use for calculation)
            teleport_inds - [ntrials,] array of trial stop times
            pthr - (float) 1 - p-value for shuffle procedure
            speed - [timepoints,] or None; animal's speed at each timepoint. Used for filtering stationary
                timepoints. If None, no filtering is performed
            nperms - number of permutations. Shuffling is performed within a trial. 
    outputs:
            if output_shuffle:
                masks - array of masks for cells with significant spatial info in each morph
                SI - array of spatial information per cell per morph
                p - p value per cell
                perm_trial_mat - array of shuffles per cell (trials x pos x neurons x permutations)
            else:
                masks
                SI
                p

    '''

    # get by trial info
    C_trial_mat, occ_trial_mat, edges,centers = trial_matrix(C,position,tstart_inds,teleport_inds,speed = speed, **kwargs)

    occ = occ_trial_mat.sum(axis=0) + 1E-3
    occ /= occ.sum()
    if use_tank_method:
        SI = spatial_info_tank(C, np.nanmean(C_trial_mat,axis=0),occ, speed=speed)
    elif 'only_spatial_binning' in kwargs.keys():
        if kwargs['only_spatial_binning']:
            SI = spatial_info(C_trial_mat,occ)
        else:
            SI = spatial_info(np.nanmean(C_trial_mat,axis=0),occ)
    else:    
        SI = spatial_info(np.nanmean(C_trial_mat,axis=0),occ)
    # SI = spatinfo_per_morph(C_trial_mat,occ_trial_mat)

    SI_perms = np.zeros([nperms,C.shape[1]])
    if output_shuffle:
        perm_trial_mat = np.zeros(np.hstack([[i for i in C_trial_mat.shape],[nperms]]))

    for perm in range(nperms):
        if perm%100 == 0:
            print('perm',perm)
        C_trial_mat, occ_trial_mat, _,__ = trial_matrix(C,position,tstart_inds,teleport_inds,speed = speed,perm=True,**kwargs)
        occ = occ_trial_mat.sum(axis=0) + 1E-3
        occ /= occ.sum()
        if use_tank_method:
            _SI_perm = spatial_info_tank(C, np.nanmean(C_trial_mat,axis=0),occ, speed=speed)
        elif 'only_spatial_binning' in kwargs.keys():
            if kwargs['only_spatial_binning']:
                _SI_perm = spatial_info(C_trial_mat,occ)
            else:
                _SI_perm =  spatial_info(np.nanmean(C_trial_mat,axis=0),occ)
        else:
            _SI_perm =  spatial_info(np.nanmean(C_trial_mat,axis=0),occ)

        SI_perms[perm,:]=_SI_perm
        
        if output_shuffle:
            if len(perm_trial_mat.shape)==4:
                perm_trial_mat[:,:,:,perm] = C_trial_mat
            else:
                perm_trial_mat[:,:,perm] = C_trial_mat

    p = np.ones([C.shape[1],])
    for cell in range(C.shape[1]):
        p[cell] = (SI[cell] <= SI_perms[:,cell]).sum()/nperms
    masks = p<pthr

    if output_shuffle:
        return masks, SI, p, perm_trial_mat, SI_perms
    else:
        return masks, SI, p


def spatial_info_perm_test(SI,C,position,tstart,tstop,nperms = 10000,shuffled_SI=None,win_trial = True, **kwargs):
    '''run permutation test on spatial information calculations and return empirical p-values for each cell
    inputs: SI - [ncells,] array of 'true' spatial information for each cell
            C - [ntimepoints,cells] activity rate of each cell over whole session
            position - [ntimepoints,] array of animal positions at each timepoint
            tstart - [ntrials,] array of trial start indices to be considered in shuffle
            tstop - [ntrials,] array of trial stop indices to be considered in shuffle
            perms - (float/int); number of permutations to run
            shuffled_SI - [ncells,permutations] or None; if shuffled distribution already exists, just
                calculate p - values
            win_trial - bool; whether to do shuffling within each trial or across whole timeseries
    returns p - [ncells,] array of 1 - p-values from perm test
            shuffled_SI - [ncells, nperms] shuffled distribution
    '''
    if len(C.shape)<2: # if only considering one cell, expand dimensions
        C = C[:,np.newaxis]
    # print(tstart,tstop)
    if shuffled_SI is None:
        shuffled_SI = np.zeros([nperms,C.shape[1]])
        for perm in range(nperms): # for each permutation

            if win_trial: # within trial permuation
                C_tmat, occ_tmat, edes,centers = trial_matrix(C,position,tstart,tstop,perm=True, **kwargs)
            else:
                C_perm = np.roll(C,randrange(30,position.shape[0],30),axis=0) # perform permutation over whole time series
                # print(tstart,tstop)
                C_tmat, occ_tmat, edes,centers = trial_matrix(C,position,tstart,tstop,perm=False, **kwargs)

            fr, occ = np.squeeze(np.nanmean(C_tmat,axis=0)), occ_tmat.sum(axis=0) # average firing rate and occupancy
            occ/=occ.sum()

            si = spatial_info(fr,occ) # shuffled spatial information
            shuffled_SI[perm,:] = si


    # calculate p-values
    p = np.zeros([C.shape[1],])
    for cell in range(C.shape[1]):
        p[cell] = np.sum(SI[cell]>shuffled_SI[:,cell])/nperms

    return p, shuffled_SI

def placecell_sort(C_trial_mat,masks,cv_sort=True,sigma = 2):
    '''plot place place cells across morph values using a cross-validated population sorting
    inputs: C_morph_dict - output from u.trial_type_dict(C_trial_mat, morphs) where C is the [trials, positions,ncells]
                and morphs is [ntrials,] array of mean morph values
            masks -  place cell masks from place_cells_calc
            cv_sort - bool; calculate sorting from all trials (False) or a randomly selected half of trials (True)
            plot - bool; whether or not to actually generate matplotlib plots or just return sorted data
    outputs: f - figure handle
            ax - axis array
            PC_dict - dictionary of cross-val sorted population activity rate maps'''


    getSort = lambda fr : np.argsort(np.argmax(np.squeeze(np.nanmean(fr,axis=0)),axis=0))

    # sorts,norms = {},{}
    if cv_sort:
        # get sort for 0 morph from random half of trials
        ntrials = C_trial_mat.shape[0]

        arr = C_trial_mat[:,:,masks]
        arr = arr[::2,:,:] # odd trials
        sorts = getSort(arr)

        arr = np.copy(arr)
        arr[np.isnan(arr)]=0.
        norms = np.amax(np.nanmean(arr,axis=0),axis=0) # normalization from training data




        # get rate maps for other half of trials

        fr = np.squeeze(np.nanmean(C_trial_mat[1::2,:,:],axis=0))
        fr = fr[:,masks]
        # print(fr.shape,norms[m])
        fr = fr/norms
        if sigma>0:
            fr = sp.ndimage.filters.gaussian_filter1d(fr[:,sorts],sigma,axis=0)
        else:
            fr = fr[:,sorts]

    return fr.T, sorts