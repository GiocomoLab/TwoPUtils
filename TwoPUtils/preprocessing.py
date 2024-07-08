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
    Fix both trial starts and teleports so that trial starts and teleports 
    are marked with "1.0" at exactly one sample per trial.

    Needed because in some sqlite files from Unity, tstart goes to 1 on the start sample 
    and then back to 0 for the remainder of the trial (same for teleport);
    but for other data sets, tstart and teleport go to 1 and stay at 1 after the first sample, and when
    this is integrated downstream, it produces values above 1 which we don't want.

    :param df: sqlite VR data as a pandas DataFrame
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
    assert teleport_inds.shape==tstart_inds.shape , "trial starts and teleports not the same shape, %d starts %d teleports" % (tstart_inds.shape[0], teleport_inds.shape[0])

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
    print(dt_ttl)
    tmp = np.zeros(dt_ttl.shape)
    tmp[dt_ttl < .01] = 1  # find ttls faster than 200 Hz (unrealistically fast - probably a ttl which bounced to ground)
    # ensured outside of this script that this finds the true start ttl on every scan
    mask = np.insert(np.diff(tmp), 0, 0)  # find first ttl in string that were too fast
    
    mask[mask < 0] = 0
    print('num aberrant ttls', tmp.sum())
    return mask==0 # original ttl's up to a 1 VR frame error (shouldn't be a meaningful issue for calcium but
                   # but it is an issue for voltage imaging

def vr_align_to_2P(vr_dataframe, scan_info, run_ttl_check=False, n_planes = 1):
    """
    Align VR data to 2P scanning data, for NLW rig
    :param vr_dataframe: VR SQLite data as a pandas dataframe
    :param scan_info: scanning metadata from Scanbox .mat file
    :param run_ttl_check: whether to check for aberrant TTLs from poor grounding
    :param n_planes: number of imaged planes
    :return: dataframe with one row per imaging frame, containing aligned/interpolated VR data
    """

    # TTLs coming from Unity to the scanbox computer are stored as frame and line
    # indices in the scanbox .mat file, loaded here as scan_info['frames'] and scan_info['lines'].
    # In MATLAB, this would be info.frames and info.lines after loading the .mat file.
    # For instance, if the second Unity TTL arrived at imaging frame 3, line 112 (e.g. out of 512),
    # then scan_info['frame_rate'][1]=3 and scan_info['lines'][1]=112. 

    # Use the frame rate and line rate to estimate timestamps at which each of these TTLs arrived,
    # relative to the start of the imaging session.
    fr = scan_info['frame_rate'] # frame rate
    lr = fr * scan_info['config']['lines']/scan_info['fov_repeats']  # line rate

    if 'frame' in scan_info.keys() and 'line' in scan_info.keys():
        frames = scan_info['frame'].astype(np.int)
        frame_diff = np.ediff1d(frames, to_begin=0)
        try:
            mods = np.argwhere(frame_diff < -100)[0]
            for i, mod in enumerate(mods.tolist()):
                frames[mod:] += (i + 1) * 65535
        except:
            pass
        frames = frames * scan_info['fov_repeats']

        # frames = np.array([f * scan_info['fov_repeats'] for f in scan_info['frame']])
        if scan_info['fold_lines']>0:
            lines = np.array([l % scan_info['fold_lines'] for l in scan_info['line']])
        else:
            lines = np.array(scan_info['line'])
    else:
        # frames = np.array([f * scan_info['fov_repeats'] for f in scan_info['frames']])
        frames = scan_info['frames'].astype(np.int)
        frame_diff = np.ediff1d(frames, to_begin=0)
        try:
            mods = np.argwhere(frame_diff < -100)[0]
            for i, mod in enumerate(mods.tolist()):
                frames[mod:] += (i + 1) * 65535
        except:
            pass
        frames = frames * scan_info['fov_repeats']
        # lines = np.array([l % scan_info['fold_lines'] for l in scan_info['lines']])
        if scan_info['fold_lines']>0:
            lines = np.array([l % scan_info['fold_lines'] for l in scan_info['lines']])
        else:
            lines = np.array(scan_info['lines'])
    

    print(f"frame rate {fr}")
    # Estimate the TTL timestamps
    ttl_times = frames / fr + lines / lr
    # print(ttl_times[-100:])

    if run_ttl_check:
        mask = _ttl_check(ttl_times)
        print('bad ttls', mask.sum())
        ttl_times = ttl_times[mask]
        frames = frames[mask]
        # lines = lines[mask]


    numVRFrames = frames.shape[0]
    # print('numVRFrames', numVRFrames)

    # create empty pandas dataframe to store calcium aligned data
    ca_df = pd.DataFrame(columns=vr_dataframe.columns, index=np.arange(int(scan_info['max_idx']/n_planes)))

    ## create an evenly spaced timeseries for the aligned frames
    # ca_time = np.arange(0, 1 / fr * scan_info['max_idx'], 1 / fr)
    ca_time = np.arange(0,1/fr * scan_info['max_idx'], n_planes/fr)
    ca_time[ca_time>ttl_times[-1]]=ttl_times[-1]

    print(f"{ttl_times.shape} ttl times,{ca_time.shape} ca2+ frame times")
    print(f"last time: VR {ttl_times[-1]}, ca2+ {ca_time[-1]}")
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

    # Below, use interpolation to "downsample" the behavior data to match the times of the imaging data

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

    # Note that for licks, taking a cumulative count per imaging frame corresponds to the 
    # number of VR frames where the capacative sensor remained at 1, which should not be 
    # interpreted literally as a number of complete licks per imaging frame, but as an
    # approximation of the rate.
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
            warnings.warn(("One more teleport and than trial start, Assuming the first trial start got clipped"))
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

    # calculate and smooth lick rate -- note this uses the cumulative lick count per imaging frame,
    # which may produce unnaturally high rates given the small time bin of each imaging frame.
    # For a more conservative estimate of lick rate when we do spatial binning downstream, we will set
    # lick count per imaging frame to 1 if ca_df['lick']>=1. 
    ca_df['lick rate'] = np.array(np.divide(ca_df['lick'], np.ediff1d(ca_df['time'], to_begin=1. / fr)))
    ca_df['lick rate'] = sp.ndimage.filters.gaussian_filter1d(ca_df['lick rate']._values, 5)

    # replace nans with 0s
    ca_df.fillna(value=0, inplace=True)
    return ca_df


def vr_align_to_2P_thor(vr_dataframe, 
                        thor_metadata, 
                        ttl_times, 
                        run_ttl_check=False,
                        n_planes = 1):

    """
    Align VR data to 2P scanning data, for Thorlabs rig. See vr_align_to_2P for more documentation.
    thor_metadata and ttl_times must already be extracted using scanner_tools.thorlabs_utils.

    :param vr_dataframe: VR SQLite data as a pandas dataframe
    :param thor_metadata: scanning metadata from ThorImage .xml file, 
        extracted with scanner_tools.thorlabs_utils.ThorHaussIO
    :param ttl_times: dictionary with ttl time stamps for 'scan' and 'unity', 
        extracted with scanner_tools.thorlabs_utils.extract_thor_sync_ttls.
        Unlike on the NLW rig, TTL timestamps come from processing the analog TTL signal.
    :param run_ttl_check: whether to check for aberrant TTLs from poor grounding
    :param n_planes: number of imaged planes
    :return: dataframe with one row per imaging frame, containing aligned/interpolated VR data
    """

    print("This function has undergone limited testing; please inspect the output!")
    
    fr = thor_metadata.frame_rate # frame rate
    if thor_metadata.zplanes is not None:
        n_planes = thor_metadata.zplanes
    
    n_frames = ttl_times['scan'].shape[0]
    
    print(f"frame rate {fr}")
    unity_ttl_times = np.copy(ttl_times['unity'])
    # print(ttl_times[-100:])
    if run_ttl_check:
        mask = tpu.preprocessing._ttl_check(unity_ttl_times)
        print('bad ttls', (~mask).sum()) #mask.sum(): these are not bad ttls, they are times to keep?
        unity_ttl_times = unity_ttl_times[mask]

    numVRFrames = unity_ttl_times.shape[0]

    # print('numVRFrames', numVRFrames)
    # print('numScanFrames', n_frames)

    # create empty pandas dataframe to store calcium aligned data
    ca_df = pd.DataFrame(columns=vr_dataframe.columns, index=np.arange(int(n_frames/n_planes)))

    ## Use the "FrameOut" TTL times from the scanner as the timeseries for the aligned frames
    ca_time = ttl_times['scan']

    ca_time[ca_time>unity_ttl_times[-1]]=unity_ttl_times[-1]
    print(f"{unity_ttl_times.shape} ttl times,{ca_time.shape} ca2+ frame times")
    print(f"last time: VR {unity_ttl_times[-1]}, ca2+ {ca_time[-1]}")
    if (ca_time.shape[0] - ca_df.shape[0]) == 1:  # occasionally a 1 frame correction due to
        # scan stopping mid frame
        warnings.warn('one frame correction')
        ca_time = ca_time[:-1]
        
    ca_df.loc[:, 'time'] = ca_time
    mask = ca_time >= unity_ttl_times[0]  # mask for when ttls have started on imaging clock
    # (i.e. imaging started and stabilized, ~10s)
    
    # take VR frames for which there are valid TTLs
    vr_dataframe = vr_dataframe.iloc[-numVRFrames:]

    #find columns that exist in sqlite file from iterable
    column_filter = lambda columns: [col for col in vr_dataframe.columns if col in columns]

    # Below, use interpolation to "downsample" the behavior data to match the times of the imaging data

    # linear interpolation of position and catmull rom spline "time" parameter
    lin_interp_cols = column_filter(('pos','posx','posy','t'))

    f_mean = sp.interpolate.interp1d(unity_ttl_times, vr_dataframe[lin_interp_cols]._values, axis=0, kind='slinear')
    ca_df.loc[mask, lin_interp_cols] = f_mean(ca_time[mask])
    # set positions before Unity TTLs started to -500
    ca_df.loc[~mask, 'pos'] = -500.

    # nearest frame interpolation
    near_interp_cols = column_filter(('morph', 'towerJitter', 'wallJitter',
                                      'bckgndJitter','trialnum','cmd','scanning','dreamland', 'LR'))

    f_nearest = sp.interpolate.interp1d(unity_ttl_times, vr_dataframe[near_interp_cols]._values, axis=0, kind='nearest')
    ca_df.loc[mask, near_interp_cols] = f_nearest(ca_time[mask])
    ca_df.fillna(method='ffill', inplace=True)
    ca_df.loc[~mask, near_interp_cols] = -1.
    
    # integrate, interpolate and then take difference, to make sure data is not lost

    # Note that for licks, taking a cumulative count per imaging frame corresponds to the 
    # number of VR frames where the capacative sensor remained at 1, which should not be 
    # interpreted literally as a number of complete licks per imaging frame, but as an
    # approximation of the rate.
    cumsum_interp_cols = column_filter(('dz', 'lick', 'reward', 'tstart', 'teleport', 'rzone'))
    f_cumsum = sp.interpolate.interp1d(unity_ttl_times, np.cumsum(vr_dataframe[cumsum_interp_cols]._values, axis=0), axis=0,
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

    # calculate and smooth lick rate -- note this uses the cumulative lick count per imaging frame,
    # which may produce unnaturally high rates given the small time bin of each imaging frame.
    # For a more conservative estimate of lick rate when we do spatial binning downstream, we will set
    # lick count per imaging frame to 1 if ca_df['lick']>=1. 
    ca_df['lick rate'] = np.array(np.divide(ca_df['lick'], np.ediff1d(ca_df['time'], to_begin=1. / fr)))
    ca_df['lick rate'] = sp.ndimage.filters.gaussian_filter1d(ca_df['lick rate']._values, 5)

    # replace nans with 0s
    ca_df.fillna(value=0, inplace=True)
    
    # subtract off the first scanning TTL time so that everything starts at time 0
    ca_df['time'] = ca_df['time']-ca_df['time'].iloc[0]
    
    # check against nframes from thor metadata
    if thor_metadata.nframes > ca_df.shape[0]:
        print("Aligned dataframe has %d fewer frames than Thor metadata" % (
            thor_metadata.nframes - ca_df.shape[0]))
        print("Adding %d empty rows to data frame" % (thor_metadata.nframes > ca_df.shape[0]))
        ca_df.loc[len(ca_df)] = pd.Series(dtype='float64')
    elif thor_metadata.nframes < ca_df.shape[0]:
        print("Aligned dataframe has %d more frames than Thor metadata" % (
            ca_df.shape[0] - thor_metadata.nframes))

    return ca_df