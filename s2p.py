import os
import numpy as np


def default_ops():
    ops = {
        # file paths
        'look_one_level_down': False,  # whether to look in all subfolders when searching for tiffs
        'fast_disk': os.path.join("E:", "s2ptmp"),
        # used to store temporary binary file, defaults to save_path0 (set to a string NOT a list)
        'delete_bin': False,  # whether to delete binary file after processing
        'mesoscan': False,  # for reading in scanimage mesoscope files
        'h5py': [],  # take h5py as input (deactivates data_path)
        'h5py_key': 'data',  # key in h5py where data array is stored
        'save_path0': [],  # stores results, defaults to first item in data_path
        'subfolders': [],
        'data_path': [],
        # main settings
        'nplanes': 1,  # each tiff has these many planes in sequence
        'nchannels': 1,  # each tiff has these many channels per plane
        'functional_chan': 1,  # this channel is used to extract functional ROIs (1-based)
        'tau': 1.,  # this is the main parameter for deconvolution
        'fs': 15.4609,  # sampling rate (PER PLANE - e.g. if you have 12 planes then this should be around 2.5)
        'force_sktiff': False,  # whether or not to use scikit-image for tiff reading
        # output settings
        'preclassify': 0,  # apply classifier before signal extraction with probability 0.5 (turn off with value 0)
        'save_mat': False,  # whether to save output as matlab files
        'combined': True,  # combine multiple planes into a single result /single canvas for GUI
        'aspect': 1.0,  # um/pixels in X / um/pixels in Y (for correct aspect ratio in GUI)
        # bidirectional phase offset
        'do_bidiphase': True,
        'bidiphase': 0,
        # registration settings
        'do_registration': 1,  # whether to register data (2 forces re-registration)
        'keep_movie_raw': False,
        'nimg_init': 300,  # subsampled frames for finding reference image
        'batch_size': 500,  # number of frames per batch
        'maxregshift': 0.1,  # max allowed registration shift, as a fraction of frame max(width and height)
        'align_by_chan': 1,  # when multi-channel, you can align by non-functional channel (1-based)
        'reg_tif': False,  # whether to save registered tiffs
        'reg_tif_chan2': False,  # whether to save channel 2 registered tiffs
        'subpixel': 10,  # precision of subpixel registration (1/subpixel steps)
        'smooth_sigma': 1.15,  # ~1 good for 2P recordings, recommend >5 for 1P recordings
        'th_badframes': 1.0,
        # this parameter determines which frames to exclude when determining cropping - set it smaller to exclude more frames
        'pad_fft': False,
        # non rigid registration settings
        'nonrigid': True,  # whether to use nonrigid registration
        'block_size': [128, 128],  # block size to register (** keep this a multiple of 2 **)
        'snr_thresh': 1.2,
        # if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing
        'maxregshiftNR': 5,  # maximum pixel shift allowed for nonrigid, relative to rigid
        # 1P settings
        '1Preg': False,  # whether to perform high-pass filtering and tapering
        'spatial_hp': 50,  # window for spatial high-pass filtering before registration
        'pre_smooth': 2,  # whether to smooth before high-pass filtering before registration
        'spatial_taper': 50,
        # how much to ignore on edges (important for vignetted windows, for FFT padding do not set BELOW 3*ops['smooth_sigma'])
        # cell detection settings
        'roidetect': True,  # whether or not to run ROI extraction
        'sparsemode': True,
        'spatial_scale': 0,  # 0: multi-scale; 1: 6 pixels, 2: 12 pixels, 3: 24 pixels, 4: 48 pixels
        'connected': True,  # whether or not to keep ROIs fully connected (set to 0 for dendrites)
        'nbinned': 5000,  # max number of binned frames for cell detection
        'max_iterations': 20,  # maximum number of iterations to do cell detection
        'threshold_scaling': 1.5,  # adjust the automatically determined threshold by this scalar multiplier
        'max_overlap': 0.75,  # cells with more overlap than this get removed during triage, before refinement
        'high_pass': 100,  # running mean subtraction with window of size 'high_pass' (use low values for 1P)
        # ROI extraction parameters
        'inner_neuropil_radius': 2,  # number of pixels to keep between ROI and neuropil donut
        'min_neuropil_pixels': 350,  # minimum number of pixels in the neuropil
        'allow_overlap': True,  # pixels that are overlapping are thrown out (False) or added to both ROIs (True)
        # channel 2 detection settings (stat[n]['chan2'], stat[n]['not_chan2'])
        'chan2_thres': 0.65,  # minimum for detection of brightness on channel 2
        # deconvolution settings
        'baseline': 'maximin',  # baselining mode (can also choose 'prctile')
        'win_baseline': 60.,  # window for maximin
        'sig_baseline': 10.,  # smoothing constant for gaussian filter
        'prctile_baseline': 8.,  # optional (whether to use a percentile baseline)
        'neucoeff': .7,  # neuropil coefficient
        'xrange': np.array([0, 0]),
        'yrange': np.array([0, 0]),
        'input_format': "sbx",
        'sbx_ndeadcols': 0}
    return ops


def set_ops(d={}):
    ops = default_ops()
    for k, v in d.items():
        ops[k] = v
    return ops


def default_db():
    db = {
        'h5py': [],  # a single h5 file path
        'h5py_key': 'data',
        'fast_disk': os.path.join("E:", "s2ptmp"),
        # string which specifies where the binary file will be stored (should be an SSD)
    }
    return db


def set_db(h5fname, d={}):
    db = default_db()
    db['h5py'] = h5fname
    for k, v in d:
        db[k] = v
    return db