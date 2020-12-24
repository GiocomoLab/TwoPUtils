import numpy as np


def trial_matrix(arr, pos, tstart_inds, tstop_inds, bin_size=10,
                 max_pos=450, speed=None, speed_thr=2, perm=False,
                 mat_only=False):
    '''make a ntrials x position [x neurons] matrix[/tensor]---heavily used
    inputs: arr - timepoints x anything array to be put into trials x positions format
            pos - position at each timepoint
            tstart_inds - indices of trial starts
            tstop_inds - indices of trial stops
            bin_size - spatial bin size in cm
            max_pos - maximum position on track
            speed - vector of speeds at each timepoint. If None, then no speed filtering is done
            speed_thr - speed threshold in cm/s. Timepoints of low speed are dropped
            perm - bool. whether to circularly permute timeseries before binning. used for permutation testing
            mat_only - bool. return just spatial binned data or also occupancy, bin edges, and bin bin_centers

    outputs: if mat_only
                    trial_mat - position binned data
            else
                    trial_mat
                    occ_mat - trials x positions matrix of bin occupancy
                    bin_edges - position bin edges
                    bin_centers - bin centers '''

    ntrials = tstart_inds.shape[0]
    if speed is not None:  # mask out speeds below speed threshold
        pos[speed < speed_thr] = -1000
        arr[speed < speed_thr, :] = np.nan

    # make position bins
    bin_edges = np.arange(0, max_pos + bin_size, bin_size)
    bin_centers = bin_edges[:-1] + bin_size / 2
    bin_edges = bin_edges.tolist()

    # if arr is a vector, expand dimension
    if len(arr.shape) < 2:
        arr = np.expand_dims(arr, axis=1)

    trial_mat = np.zeros([int(ntrials), len(bin_edges) - 1, arr.shape[1]])
    trial_mat[:] = np.nan
    occ_mat = np.zeros([int(ntrials), len(bin_edges) - 1])
    for trial in range(int(ntrials)):  # for each trial
        # get trial indices
        firstI, lastI = tstart_inds[trial], tstop_inds[trial]

        arr_t, pos_t = arr[firstI:lastI, :], pos[firstI:lastI]
        if perm:  # circularly permute if desired
            pos_t = np.roll(pos_t, np.random.randint(pos_t.shape[0]))

        # average within spatial bins
        for b, (edge1, edge2) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if np.where((pos_t > edge1) & (pos_t <= edge2))[0].shape[0] > 0:
                trial_mat[trial, b] = np.nanmean(arr_t[(pos_t > edge1) & (pos_t <= edge2), :], axis=0)
                occ_mat[trial, b] = np.where((pos_t > edge1) & (pos_t <= edge2))[0].shape[0]
            else:
                pass

    if mat_only:
        return np.squeeze(trial_mat)
    else:
        return np.squeeze(trial_mat), np.squeeze(occ_mat / occ_mat.sum(axis=1)[:, np.newaxis]), bin_edges, bin_centers


def spatial_info(frmap,occupancy):
    '''calculate spatial information bits/spike
    inputs: frmap - [spatial bins, cells] spatially binned activity rate for each cell
            occupancy - [spatial bins,] fractional occupancy of each spatial bin
    outputs: SI - [cells,] spatial information for each cell '''

    ### vectorizing
    P_map = frmap - np.amin(frmap)+.001 # make sure there's no negative activity rates
    P_map = P_map/P_map.mean(axis=0)
    SI = ((P_map*occupancy[:,np.newaxis])*np.log2(P_map)).sum(axis=0) # Skaggs and McNaughton spatial information

    return SI


def place_cells_calc(C, position, trial_info, tstart_inds,
                teleport_inds,pthr = .05,speed=None,win_trial_perm=True,morphlist = [0,1],
                bootstrap = True,nperms = 100):
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
            win_trial_perm - (bool); whether to perform shuffling within a trial. If false,
                shuffling is performed with respect to the entire timeseries
            morphlist - (list); which mean morph values to use in separate place cell calculations
    outputs: masks - dictionary of masks for cells with significant spatial info in each morph
            FR - dictionary of firing rate maps per cell per morph
            SI - dictionary of spatial information per cell per morph

    '''

    # get by trial info
    C_trial_mat, occ_trial_mat, edges,centers = u.make_pos_bin_trial_matrices(C,position,tstart_inds,teleport_inds,speed = speed)
    morphs = trial_info['morphs'] # mean morph values

    def spatinfo_per_morph(_trial_mat,_occ_mat):
        _SI = {}
        for m in morphlist:
            _occ = _occ_mat[morphs==m,:].sum(axis=0)
            _occ/=_occ.sum()
            _SI[m] = spatial_info(np.nanmean(_trial_mat[morphs==m,:,:],axis=0),_occ)
        return _SI

    SI = spatinfo_per_morph(C_trial_mat,occ_trial_mat)

    SI_perms = {m:np.zeros((nperms,C.shape[1])) for m in morphlist}

    for perm in range(nperms):
        if perm%100 == 0:
            print('perm',perm)
        C_trial_mat, occ_trial_mat, _,__ = u.make_pos_bin_trial_matrices(C,position,tstart_inds,teleport_inds,speed = speed,perm=True)
        _SI_perm =  spatinfo_per_morph(C_trial_mat,occ_trial_mat)
        for m in _SI_perm.keys():
            SI_perms[m][perm,:]=_SI_perm[m]
    masks = {}
    pvals = {}
    for m in morphlist:
        masks[m]=[]
        p = np.ones([C.shape[1],])
        for cell in range(C.shape[1]):
            p[cell] = (SI[m][cell] <= SI_perms[m][:,cell]).sum()/nperms
        masks[m] = p<=pthr
        pvals[m] = p
    return masks, SI, pvals


def spatial_info_perm_test(SI,C,position,tstart,tstop,nperms = 10000,shuffled_SI=None,win_trial = True):
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
                C_tmat, occ_tmat, edes,centers = u.make_pos_bin_trial_matrices(C,position,tstart,tstop,perm=True)
            else:
                C_perm = np.roll(C,randrange(30,position.shape[0],30),axis=0) # perform permutation over whole time series
                # print(tstart,tstop)
                C_tmat, occ_tmat, edes,centers = u.make_pos_bin_trial_matrices(C,position,tstart,tstop,perm=False)

            fr, occ = np.squeeze(np.nanmean(C_tmat,axis=0)), occ_tmat.sum(axis=0) # average firing rate and occupancy
            occ/=occ.sum()

            si = spatial_info(fr,occ) # shuffled spatial information
            shuffled_SI[perm,:] = si


    # calculate p-values
    p = np.zeros([C.shape[1],])
    for cell in range(C.shape[1]):
        p[cell] = np.sum(SI[cell]>shuffled_SI[:,cell])/nperms

    return p, shuffled_SI
