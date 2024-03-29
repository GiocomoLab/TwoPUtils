import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import filters


def gaussian(mu, sigma, x):
    """
    radial basis function centered at 'mu' with width 'sigma', sampled at 'x'
    :param mu:
    :param sigma:
    :param x:
    :return:
    """

    return np.exp(-(mu - x) ** 2 / sigma ** 2)


class LOTrialO:
    """
    Iterator for train and test indices for  leave-one-trial-out cross-validation
    using all timepoints.
    usage: for train,test in LOTrialO(starts,stops,S.shape[0]):
                S_train,S_test = S[train,:], S[test,:]
                model.fit(S_train)
                model.test(S_test)

            starts - indices of trial starts
            stops - indices of trial stops
            N - size of timepoints dimension

            returns: train - boolean of training indices
                    test - boolean of test indices

    """

    def __init__(self, starts, stops, N):
        """

        :param starts:
        :param stops:
        :param N:
        """
        self.train_mask = np.zeros([N, ])
        self.test_mask = np.zeros([N, ])
        self.starts = starts
        self.stops = stops
        self.N = N

    def __iter__(self):
        """

        :return:
        """
        self.c = -1
        return self

    def get_masks(self):
        """

        :return:
        """
        self.train_mask *= 0
        self.test_mask *= 0
        for t, (start, stop) in enumerate(zip(self.starts, self.stops)):
            if t == self.c:
                self.test_mask[start:stop] += 1
            else:
                self.train_mask[start:stop] += 1
        return self.train_mask > 0, self.test_mask > 0

    def __next__(self):
        """

        :return:
        """
        self.c += 1
        if self.c >= self.starts.shape[0]:
            raise StopIteration
        train, test = self.get_masks()
        return train, test


def nansmooth(a, sig):
    """
    apply Gaussian smoothing to matrix A containing nans with kernel sig
    without propagating nans
    :param a:
    :param sig:
    :return:
    """

    # find nans
    nan_inds = np.isnan(a)
    a_nanless = np.copy(a)
    # make nans 0
    a_nanless[nan_inds] = 0

    # inversely weight nanned indices
    one = np.ones(a.shape)
    one[nan_inds] = .001
    a_nanless = filters.gaussian_filter(a_nanless, sig)
    one = filters.gaussian_filter(one, sig)
    return a_nanless / one


def dff(C, sig_baseline=10, win_baseline=300, sig_output=3, method='maximin'):
    """
    delta F / F using maximin method from Suite2P
    inputs: C - neuropil subtracted fluorescence (neurons x timepoints)
    outputs dFF -  neurons x timepoints

    :param C:
    :param sig_baseline:
    :param win_baseline:
    :param sig_output:
    :param method:
    :return:
    """

    C_ = np.copy(C)
    
    if method == 'maximin':  # windowed baseline estimation
        flow = filters.gaussian_filter(C_, [0, sig_baseline])
        flow = filters.minimum_filter1d(flow, win_baseline, axis=1)
        flow = filters.maximum_filter1d(flow, win_baseline, axis=1)
    else:
        flow = None
        raise NotImplementedError

    C_ -= flow  # substract baseline (dF)
    C_ /= flow  # divide by baseline (dF/F)
    return filters.gaussian_filter(C_, [0, sig_output])  # smooth result


def correct_trial_mask(rewards, starts, stops, N):
    """
    create mask for indices where rewards is greater than 0
    inputs: rewards - [trials,] list or array with number of rewards per trial
            starts - list of indices for trial starts
            stops - list of inidices for trial stops
            N - length of total timeseries (i.e. S.shape[0])
    outputs: pcnt - mask of indices for trials where the animal received a reward


    :param rewards:
    :param starts:
    :param stops:
    :param N:
    :return:
    """

    pcnt = np.zeros([N, ])  # initialize

    # loop through trials and make mask
    for i, (start, stop) in enumerate(zip(starts, stops)):
        pcnt[start:stop] = int(rewards[i] > 0)
    return pcnt


def lick_positions(licks, position):
    """
    creates vector of lick positions for making a lick raster
    inputs: licks - [timepoints,] or [timepoints,1] vector of number of licks at each timepoint
            positions - corresponding vector of positions
    outputs: lickpos - nans where licks==0 and position where licks>0

    :param licks:
    :param position:
    :return:
    """

    lickpos = np.zeros([licks.shape[0], ])
    lickpos[:] = np.nan
    lick_inds = np.where(licks > 0)[0]
    lickpos[lick_inds] = position[lick_inds]
    return lickpos


def smooth_raster(x, mat, ax=None, smooth=False, sig=2, vals=None, cmap='cool', tports=None):
    """
    plot mat ( ntrials x positions) as a smoothed histogram
    inputs: x - positions array (i.e. bin centers)
            mat - trials x positions array to be plotted
            ax - matplotlib axis object to use. if none, create a new figure and new axis
            smooth - bool. smooth raster or not
            sig - width of Gaussian smoothing
            vals - values used to color lines in histogram (e.g. morph value)
            cmap - colormap used appled to vals
            tports - if mouse is teleported between the end of the trial, plot position  of teleport as x
    outputs: ax - axis of plot object


    :param x:
    :param mat:
    :param ax:
    :param smooth:
    :param sig:
    :param vals:
    :param cmap:
    :param tports:
    :return:
    """

    if ax is None:
        f, ax = plt.subplots()

    cm = plt.cm.get_cmap(cmap)

    if smooth:
        mat = filters.gaussian_filter1d(mat, sig, axis=1)

    for ind, i in enumerate(np.arange(mat.shape[0] - 1, 0, -1)):
        if vals is not None:
            ax.fill_between(x, mat[ind, :] + i, y2=i, color=cm(np.float(vals[ind])), linewidth=.001)
        else:
            ax.fill_between(x, mat[ind, :] + i, y2=i, color='black', linewidth=.001)

        if tports is not None:
            ax.scatter(tports[ind], i + .5, color=cm(np.float(vals[ind])), marker='x', s=50)

    ax.set_yticks(np.arange(0, mat.shape[0], 10))
    ax.set_yticklabels(["%d" % i for i in np.arange(mat.shape[0], 0, -10).tolist()])

    if ax is None:
        return f, ax
    else:
        return ax
