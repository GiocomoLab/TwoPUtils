import sqlite3 as sql
import warnings

import numpy as np
import pandas as pd
import scipy as sp


def load_sqlite(sqlite_filename: str, fix_teleports=True):
    """

    :param sqlite_filename:
    :param fix_teleports:
    :return:
    """

    # connect to sqlite file
    sess_conn = sql.connect(sqlite_filename)

    # load all columns
    df = pd.read_sql('''SELECT * FROM data''',sess_conn)

    if fix_teleports:
        _fix_teleports(df)

    return df


def _fix_teleports(df: pd.DataFrame):
    """
        fix shoddy teleport signals in-place from older vr sessions
    :param df:
    :return:
    """
    print("Fixing teleports")
    try:
        pos = df['pos']._values
    except:
        pos = df['posz']._values
    pos[pos < -50] = -50
    teleport_inds = np.where(np.ediff1d(pos, to_end=0) <= -50)[0]
    tstart_inds = np.append([0], teleport_inds[:-1] + 1)
    assert teleport_inds.shape==tstart_inds.shape , "trial starts and teleports not the same shape"

    for ind in range(tstart_inds.shape[0]):  # for teleports
        while (pos[tstart_inds[ind]] < 0):  # while position is negative
            if tstart_inds[ind] < pos.shape[0] - 1:  # if you haven't exceeded the vector length
                tstart_inds[ind] = tstart_inds[ind] + 1  # go up one index
            else:  # otherwise you should be the last teleport and delete this index
                print("deleting last index from trial start")
                tstart_inds = np.delete(tstart_inds, ind)
                break

    tstart_inds_vec = np.zeros([df.shape[0], ])
    tstart_inds_vec[tstart_inds] = 1

    teleport_inds_vec = np.zeros([df.shape[0], ])
    teleport_inds_vec[teleport_inds] = 1
    assert tstart_inds_vec.sum() == teleport_inds_vec.sum(), "start and teleport ind2 to vec failed"
    df['teleport'] = teleport_inds_vec
    df['tstart'] = tstart_inds_vec
    # return df

def _fix_tstarts(df: pd.DataFrame):
    """

    :param df:
    :return:
    """

    try:
        pos = df['pos']._values
    except:
        pos = df['posz']._values

    teleport_inds = np.where(df['teleport']._values ==1)[0]
    tstart_inds = np.append([0], teleport_inds[:-1] + 1)

    for ind in range(tstart_inds.shape[0]):  # for teleports
        while (pos[tstart_inds[ind]] < 0):  # while position is negative
            if tstart_inds[ind] < pos.shape[0] - 1:  # if you haven't exceeded the vector length
                tstart_inds[ind] = tstart_inds[ind] + 1  # go up one index
            else:  # otherwise you should be the last teleport and delete this index
                print("deleting last index from trial start")
                tstart_inds = np.delete(tstart_inds, ind)
                break

    tstart_inds_vec = np.zeros([df.shape[0], ])
    tstart_inds_vec[tstart_inds] = 1


    df['tstart'] = tstart_inds_vec


def _ttl_check(ttl_times):
    """
    on Feb 6, 2019 noticed that AA's new National Instruments board
    created a floating ground on my TTL circuit. This caused a bunch of extra TTLs
    due to unexpected grounding of the signal. This is only a problem for a small number
    of sessions in Feb-Mar of 2019

    :param ttl_times:
    :return:
    """


    dt_ttl = np.diff(np.insert(ttl_times, 0, 0))  # insert zero at beginning and calculate delta ttl time
    tmp = np.zeros(dt_ttl.shape)
    tmp[dt_ttl < .01] = 1  # find ttls faster than 200 Hz (unrealistically fast - probably a ttl which bounced to ground)
    # ensured outside of this script that this finds the true start ttl on every scan
    mask = np.insert(np.diff(tmp), 0, 0)  # find first ttl in string that were too fast
    mask[mask < 0] = 0
    print('num aberrant ttls', tmp.sum())
    return mask==0 # original ttl's up to a 1 VR frame error (shouldn't be a meaningful issue for calcium but
                   # but it is an issue for voltage imaging

def vr_align_to_2P(vr_dataframe, scan_info, run_ttl_check=False):
    """
    place holder
    :param infofile:
    :param n_imaging_planes: 
    :param n_lines: 
    :return: 
    """




    fr = scan_info['frame_rate']  # frame rate
    lr = fr * scan_info['config']['lines']/scan_info['fov_repeats']  # line rate

    if 'frame' in scan_info.keys() and 'line' in scan_info.keys():
        frames = np.array([f * scan_info['fov_repeats'] for f in scan_info['frame']])
        if scan_info['fold_lines']>0:
            lines = np.array([l % scan_info['fold_lines'] for l in scan_info['line']])
        else:
            lines = np.array(scan_info['line'])
    else:
        frames = np.array([f * scan_info['fov_repeats'] for f in scan_info['frames']])

        # lines = np.array([l % scan_info['fold_lines'] for l in scan_info['lines']])
        if scan_info['fold_lines']>0:
            lines = np.array([l % scan_info['fold_lines'] for l in scan_info['lines']])
        else:
            lines = np.array(scan_info['lines'])
    if 'otwave' in scan_info.keys():
        frames = frames[::scan_info['otwave'].shape[0]]
        lines = lines[::scan_info['otwave'].shape[0]]
    else:
        print("no optotune wave parameters, assuming single plane")
    # try:
    #     frames = np.array([f*scan_info['fov_repeats'] for f in scan_info['frames']])
    #     lines = np.array([l%scan_info['fold_lines'] for l in scan_info['lines']])
    # except:
    #     frames = np.array([f * scan_info['fov_repeats'] for f in scan_info['frame']])
    #     lines = np.array([l % scan_info['fold_lines'] for l in scan_info['line']])
    ttl_times = frames / fr + lines / lr

    if run_ttl_check:
        mask = _ttl_check(ttl_times)
        ttl_times = ttl_times[mask]
        frames = frames[mask]
        # lines = lines[mask]


    numVRFrames = frames.shape[0]


    # create empty pandas dataframe to store calcium aligned data
    ca_df = pd.DataFrame(columns=vr_dataframe.columns, index=np.arange(scan_info['max_idx']))
    ca_time = np.arange(0, 1 / fr * scan_info['max_idx'], 1 / fr)  # time on this even grid
    ca_time[ca_time>ttl_times[-1]]=ttl_times[-1]
    print(ttl_times.shape,ca_time.shape)
    print(ttl_times[-1],ca_time[-1])
    if (ca_time.shape[0] - ca_df.shape[0]) == 1:  # occasionally a 1 frame correction due to
        # scan stopping mid frame
        warnings.warn('one frame correction')
        ca_time = ca_time[:-1]

    ca_df.loc[:, 'time'] = ca_time
    mask = ca_time >= ttl_times[0]  # mask for when ttls have started on imaging clock
    # (i.e. imaging started and stabilized, ~10s)

    # take VR frames for which there are valid TTLs
    vr_dataframe = vr_dataframe.iloc[-numVRFrames:]

    #find columns that exist in sqlite file from iterable
    column_filter = lambda columns: [col for col in vr_dataframe.columns if col in columns]
    # linear interpolation of position and catmull rom spline "time" parameter
    lin_interp_cols = column_filter(('pos','posx','posy','t'))

    f_mean = sp.interpolate.interp1d(ttl_times, vr_dataframe[lin_interp_cols]._values, axis=0, kind='slinear')
    ca_df.loc[mask, lin_interp_cols] = f_mean(ca_time[mask])
    ca_df.loc[~mask, 'pos'] = -500.

    # nearest frame interpolation
    near_interp_cols = column_filter(('morph', 'towerJitter', 'wallJitter',
                                      'bckgndJitter','trialnum','cmd','scanning','dreamland', 'LR'))

    f_nearest = sp.interpolate.interp1d(ttl_times, vr_dataframe[near_interp_cols]._values, axis=0, kind='nearest')
    ca_df.loc[mask, near_interp_cols] = f_nearest(ca_time[mask])
    ca_df.fillna(method='ffill', inplace=True)
    ca_df.loc[~mask, near_interp_cols] = -1.

    # integrate, interpolate and then take difference, to make sure data is not lost
    cumsum_interp_cols = column_filter(('dz', 'lick', 'reward', 'tstart', 'teleport', 'rzone'))
    f_cumsum = sp.interpolate.interp1d(ttl_times, np.cumsum(vr_dataframe[cumsum_interp_cols]._values, axis=0), axis=0,
                                       kind='slinear')
    ca_cumsum = np.round(np.insert(f_cumsum(ca_time[mask]), 0, [0]*len(cumsum_interp_cols), axis=0))
    if ca_cumsum[-1, -2] < ca_cumsum[-1, -3]:
        ca_cumsum[-1, -2] += 1

    ca_df.loc[mask, cumsum_interp_cols] = np.diff(ca_cumsum, axis=0)
    ca_df.loc[~mask, cumsum_interp_cols] = 0.

    # fill na here
    ca_df.loc[np.isnan(ca_df['teleport']._values), 'teleport'] = 0
    ca_df.loc[np.isnan(ca_df['tstart']._values), 'tstart'] = 0
    # if first tstart gets clipped
    if ca_df['teleport'].sum(axis=0) != ca_df['tstart'].sum(axis=0):
        warnings.warn("Number of teleports and trial starts don't match")
        if ca_df['teleport'].sum(axis=0) - ca_df['tstart'].sum(axis=0) == 1:
            warnings.warn(("One more teleport and than trial start, Assuming the first trial start got clipped during "))
            ca_df['tstart'].iloc[0]=1

        if ca_df['teleport'].sum(axis=0) - ca_df['tstart'].sum(axis=0) == -1:
            warnings.warn(('One more trial start than teleport, assuming the final teleport got chopped'))
            ca_df['teleport'].iloc[-1]=1
    # smooth instantaneous speed

    cum_dz = sp.ndimage.filters.gaussian_filter1d(np.cumsum(ca_df['dz']._values), 5)
    ca_df['dz'] = np.ediff1d(cum_dz, to_end=0)

    # ca_df['speed'].interpolate(method='linear', inplace=True)
    ca_df['speed'] = np.array(np.divide(ca_df['dz'], np.ediff1d(ca_df['time'], to_begin=1. / fr)))
    ca_df['speed'].iloc[0] = 0

    # calculate and smooth lick rate
    ca_df['lick rate'] = np.array(np.divide(ca_df['lick'], np.ediff1d(ca_df['time'], to_begin=1. / fr)))
    ca_df['lick rate'] = sp.ndimage.filters.gaussian_filter1d(ca_df['lick rate']._values, 5)

    # replace nans with 0s
    ca_df.fillna(value=0, inplace=True)
    return ca_df
