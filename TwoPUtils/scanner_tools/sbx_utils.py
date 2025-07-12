import os

import h5py
import numpy as np
import scipy.io as spio

# for cutting deadband induced by bdirectional recording
import sbxreader
import tifffile

def loadmat(filename, sbx_version=2):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    info = _check_keys(data)['info']
    # Defining number of channels/size factor
    if sbx_version==3:
        if info['chan']['nchan'] == 1:
            info['nChan'] = 1
            factor = 2
        elif info['chan']['nchan'] == 2:
            info['nChan'] = 2
            factor = 1
        else:
            raise UserWarning("wrong 'channels' argument")
    else:
        if info['channels'] == 1:
            info['nChan'] = 2
            factor = 1
        elif info['channels'] == 2:
            info['nChan'] = 1
            factor = 2
        elif info['channels'] == 3:
            info['nChan'] = 1
            factor = 2
        else:
            raise UserWarning("wrong 'channels' argument")

    if info['scanmode'] == 0:
        info['recordsPerBuffer'] *= 2

    if 'fold_lines' in info.keys():
        if info['fold_lines']>0:
            info['fov_repeats'] = int(info['config']['lines']/info['fold_lines'])
        else:
            info['fov_repeats']=1
    else:
        info['fold_lines']=0
        info['fov_repeats']=1
    # Determine number of frames in whole file

    info['orig_max_idx'] = int(
        os.path.getsize(filename[:-4] + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1) 


    info['max_idx'] = int(
        os.path.getsize(filename[:-4] + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1) * int(info['fov_repeats'])

    
    info['frame_rate'] = info['resfreq'] / info['config']['lines'] * (2 - info['scanmode'])*info['fov_repeats']
    if 'otwave' in info.keys():
        info['n_planes']=info['otwave'].shape[0]

    return info
# def loadmat(filename):
#     '''
#     this function should be called instead of direct spio.loadmat
#     as it cures the problem of not properly recovering python dictionaries
#     from mat files. It calls the function check keys to cure all entries
#     which are still mat-objects
#     '''
#     data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
#     info = _check_keys(data)['info']
#     # Defining number of channels/size factor
#     if info['channels'] == 1:
#         info['nChan'] = 2
#         factor = 1
#     elif info['channels'] == 2:
#         info['nChan'] = 1
#         factor = 2
#     elif info['channels'] == 3:
#         info['nChan'] = 1
#         factor = 2
#     else:
#         raise UserWarning("wrong 'channels' argument")

#     if info['scanmode'] == 0:
#         info['recordsPerBuffer'] *= 2

#     if 'fold_lines' in info.keys():
#         if info['fold_lines']>0:
#             info['fov_repeats'] = int(info['config']['lines']/info['fold_lines'])
#         else:
#             info['fov_repeats']=1
#     else:
#         info['fold_lines']=0
#         info['fov_repeats']=1
#     # Determine number of frames in whole file

#     info['orig_max_idx'] = int(
#         os.path.getsize(filename[:-4] + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1) 


#     info['max_idx'] = int(
#         os.path.getsize(filename[:-4] + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1) * int(info['fov_repeats'])

    
#     info['frame_rate'] = info['resfreq'] / info['config']['lines'] * (2 - info['scanmode'])*info['fov_repeats']
#     if 'otwave' in info.keys():
#         info['n_planes']=info['otwave'].shape[0]

#     return info


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''

    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''

    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def sbxread(filename, k=0, N=None, **kwargs):
    '''
    Input: filename should be full path excluding .sbx, starting index, batch size
    By default Loads whole file at once, make sure you have enough ram available to do this
    '''
    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    # Load info
    info = loadmat(filename + '.mat', **kwargs)  # ['info']
    # print info.keys()

    # Paramters
    # k = 0; #First frame
    max_idx = info['max_idx']
    if N is None:
        N = max_idx  # Last frame
    else:
        N = min([N, max_idx - k])

    nSamples = info['sz'][1] * info['recordsPerBuffer'] / info['fov_repeats']* 2 * info['nChan']

    # Open File
    fo = open(filename + '.sbx')

    # print(int(k) * int(nSamples))
    fo.seek(int(k) * int(nSamples), 0)
    x = np.fromfile(fo, dtype='uint16', count=int(nSamples / 2 * N))
    x = np.int16((np.int32(65535) - x).astype(np.int32) / np.int32(2))
    x = x.reshape((info['nChan'], info['sz'][1], int(info['recordsPerBuffer']/info['fov_repeats']), int(N)), order='F')

    return x
# def sbxread(filename, k=0, N=None):
#     '''
#     Input: filename should be full path excluding .sbx, starting index, batch size
#     By default Loads whole file at once, make sure you have enough ram available to do this
#     '''
#     # Check if contains .sbx and if so just truncate
#     if '.sbx' in filename:
#         filename = filename[:-4]

#     # Load info
#     info = loadmat(filename + '.mat')  # ['info']
#     # print info.keys()

#     # Paramters
#     # k = 0; #First frame
#     max_idx = info['max_idx']
#     if N is None:
#         N = max_idx  # Last frame
#     else:
#         N = min([N, max_idx - k])

#     nSamples = info['sz'][1] * info['recordsPerBuffer'] / info['fov_repeats']* 2 * info['nChan']
#     # print(nSamples, N)

#     # Open File
#     fo = open(filename + '.sbx')

#     # print(int(k) * int(nSamples))
#     fo.seek(int(k) * int(nSamples), 0)
#     x = np.fromfile(fo, dtype='uint16', count=int(nSamples / 2 * N))
#     x = np.int16((np.int32(65535) - x).astype(np.int32) / np.int32(2))
#     x = x.reshape((info['nChan'], info['sz'][1], int(info['recordsPerBuffer']/info['fov_repeats']), int(N)), order='F')

#     return x

def sbx2h5(filename, channel_i=-1, batch_size=1000, dataset="data", output_name=None, max_idx=None,
           force_2chan=False,**kwargs):
    info = loadmat(filename + '.mat', **kwargs)  # ['info']
    if force_2chan:
        nchan = 2
    else:
        nchan = info['nChan']
        
    k = 0 # starting frame to read
    k_write = 0 # starting frame to write
    
    if output_name is None:
        h5fname = filename + '.h5'
    else:
        h5fname = output_name
        
    if 'sbx_version' in kwargs.keys():
        if kwargs['sbx_version'] !=3:
            info['pockmux'] = 0
    else:
        info['pockmux'] = 0
        
    if max_idx is None:
        max_idx = info['max_idx']


    base, last = os.path.split(h5fname)
    os.makedirs(base, exist_ok=True)
    with h5py.File(h5fname, 'w') as f:

        if channel_i == -1:
            if info['pockmux']==1:
                dset = f.create_dataset(dataset, (int(max_idx/2) * nchan, int(info['sz'][0]/info['fov_repeats']), info['sz'][1]))
                write_batch_size = int(batch_size/2)
            else:
                dset = f.create_dataset(dataset, (int(max_idx) * nchan, int(info['sz'][0]/info['fov_repeats']), info['sz'][1]))
                write_batch_size = batch_size
                
            print('dset size', dset)
            while (k <= max_idx):  # info['max_idx']:
                data = sbxread(filename, k, batch_size, **kwargs)
                data = np.transpose(data[:, :, :, :], axes=(0, 3, 2, 1))

                print(k, min((k + batch_size, max_idx)))
                
                # iterate through channels
                for chan in range(info['nChan']): # keep this loop as true info['nChan'] to avoid indexing error in data
                    if info['pockmux']==1: # if using mux mode in Scanbox 3
                        if chan==0:
                            # for PMT0 we take every other frame starting with frame 0 
                            this_batch = np.squeeze(data[chan, ::2, :, :])
                        
                        elif chan==1:
                            # for PMT1 we take every other frame starting with frame 1
                             # check size of remaining data
                            this_batch = np.squeeze(data[chan, 1::2, :, :])
                            
                            
                        # check size of remaining data
                        size_to_alloc = int((min((k + batch_size, max_idx))-k)/2)

                        size_to_read = this_batch.shape[0]

                        # print('read', size_to_read, 'alloc', size_to_alloc, 'this_batch', this_batch.shape)
                        
                        ## Below is a catch if the scan has an odd info['max_idx'] and therefore a 
                        ## different number of frames on each channel
                        if size_to_read > size_to_alloc:
                            # if one more sample to write that the allocation size, trim it
                            this_batch = this_batch[:-1, :, :]
                            print('trimming', this_batch.shape)
                        elif size_to_alloc > size_to_read:
                            # if one fewer sample, pad with a frame of zeros
                            this_batch = np.stack(this_batch, np.zeros((1,this_batch.shape[1],this_batch.shape[2])),axis=0)
                            print('padding', this_batch.shape)

                        dset[k_write * nchan + chan:min(
                                (nchan * (k_write + write_batch_size) + chan, nchan * max_idx)):nchan, :,
                            :] = this_batch
      
                    else: # else if not using mux or using an earlier Scanbox version
                        dset[k * nchan + chan:min(
                            (nchan * (k_write + write_batch_size) + chan, nchan * max_idx)):nchan, :,
                        :] = np.squeeze(data[chan, :, :, :])

                f.flush()
                k += batch_size
                k_write += write_batch_size
            
        else:
            dset = f.create_dataset(dataset, (int(max_idx), int(info['sz'][0]/info['fov_repeats']), info['sz'][1]))
            while k <= max_idx:  # info['max_idx']:
                # print(k)
                data = sbxread(filename, k, batch_size, **kwargs)
                data = np.transpose(data[channel_i, :, :, :], axes=(2, 1, 0))
                print(k, min((k + batch_size, info['max_idx'])))
                dset[k:min((k + batch_size, info['max_idx'])), :, :] = data
                f.flush()
                k += batch_size
           

    return h5fname

# def array2h5(arr, h5fname, dataset="data"):
#     with h5py.File(h5fname, 'w') as f:
#         dset = f.create_dataset(dataset, data=arr)


# def sbx2h5(filename, channel_i=-1, batch_size=1000, dataset="data", output_name=None, max_idx=None,
#            force_2chan=False):
#     info = loadmat(filename + '.mat')  # ['info']
#     if force_2chan:
#         nchan = 2
#     else:
#         nchan = info['nChan']
#     k = 0
#     if output_name is None:
#         h5fname = filename + '.h5'
#     else:
#         h5fname = output_name

#     if max_idx is None:
#         max_idx = info['max_idx']

#     base, last = os.path.split(h5fname)
#     os.makedirs(base, exist_ok=True)
#     with h5py.File(h5fname, 'w') as f:

#         if channel_i == -1:
#             dset = f.create_dataset(dataset, (int(max_idx) * nchan, int(info['sz'][0]/info['fov_repeats']), info['sz'][1]))
#             while k <= max_idx:  # info['max_idx']:
#                 # print(k)
#                 data = sbxread(filename, k, batch_size)
#                 data = np.transpose(data[:, :, :, :], axes=(0, 3, 2, 1))

#                 print(k, min((k + batch_size, info['max_idx'])))
#                 # channel 0
#                 for chan in range(info['nChan']): # keep this loop as true info['nChan'] to avoid indexing error in data
#                     dset[k * nchan + chan:min(
#                         (nchan * (k + batch_size) + chan, nchan * info['max_idx'])):nchan, :,
#                     :] = np.squeeze(data[chan, :, :, :])

#                 f.flush()
#                 k += batch_size
#         else:
#             dset = f.create_dataset(dataset, (int(max_idx), int(info['sz'][0]/info['fov_repeats']), info['sz'][1]))
#             while k <= max_idx:  # info['max_idx']:
#                 # print(k)
#                 data = sbxread(filename, k, batch_size)
#                 data = np.transpose(data[channel_i, :, :, :], axes=(2, 1, 0))
#                 print(k, min((k + batch_size, info['max_idx'])))
#                 dset[k:min((k + batch_size, info['max_idx'])), :, :] = data
#                 f.flush()
#                 k += batch_size

#     return h5fname

def find_deadbands(filename, multiplane = True):
    # this function is to find the deadband due to bidirectional recording
    #  Didn't git to cut the dead row due to lack of undertand of the sbx data structure, should potential fix this though
    f = sbxreader.sbx_memmap (filename + '.sbx')
    if f.metadata["scanning_mode"] == 'bidirectional':
        ndeadcols_l = f. ndeadcols+20
        ndeadcols_r = 30
    else:
        ndeadcols_l = 76+20 # For Can's data, due to wrong recording, some of the animals were unidirectional recording
                            # put the same numder here for the sake of cutting the FOV to the same size
        ndeadcols_r = 30

    if multiplane == True:
       #colprofile = np.array(np.mean(tmpsbx[0][0][0], axis=1))
       ndeadrows = 50#100  #np.argmax(np.diff(colprofile)) + 1
    else:
       ndeadrows = 0

    return ndeadcols_l, ndeadcols_r,ndeadrows

    

def sbx2h5_cutdb(filename, channel_i=-1, batch_size=1000, dataset="data", output_name=None, max_idx=None,
           force_2chan=False, output_dir=None, **kwargs):
    info = loadmat(filename + '.mat', **kwargs)  # ['info']
    if force_2chan:
        nchan = 2
    else:
        nchan = info['nChan']
    k = 0
    if output_name is None:
        h5fname = filename + '.h5'
    else:
        h5fname = output_name

    if max_idx is None:
        max_idx = info['max_idx']

    # Handle output directory
    if output_dir is not None:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Get just the filename without path
        base_name = os.path.basename(h5fname)
        # Create full path with output directory
        h5fname = os.path.join(output_dir, base_name)
    else:
        # Use original behavior if no output_dir specified
        base, last = os.path.split(h5fname)
        os.makedirs(base, exist_ok=True)

    with h5py.File(h5fname, 'w') as f:
        ndeadcols_l, ndeadcols_r,ndeadrows = find_deadbands(filename)
        if channel_i == -1:
            dset = f.create_dataset(dataset, (int(max_idx) * nchan, int(info['sz'][0]/info['fov_repeats']-ndeadrows), info['sz'][1]-ndeadcols_r-ndeadcols_l))
            while k <= max_idx:  # info['max_idx']:
                # print(k)
            
                data = sbxread(filename, k, batch_size,**kwargs)
                data = np.transpose(data[:, ndeadcols_l:-ndeadcols_r, ndeadrows:, :], axes=(0, 3, 2, 1))

                print(k, min((k + batch_size, info['max_idx'])))
                # channel 0
                for chan in range(info['nChan']): # keep this loop as true info['nChan'] to avoid indexing error in data
                    # print(k * nchan + chan:min((nchan * (k + batch_size) + chan,nchan * info['max_idx'])))
                    dset[k * nchan + chan:min(
                        (nchan * (k + batch_size) + chan, nchan * info['max_idx'])):nchan, :,
                    :] = np.squeeze(data[chan, :, :, :])

                f.flush()
                k += batch_size
                #tifffile.imwrite('test_tif.tif',dset)
        else:
            dset = f.create_dataset(dataset, (int(max_idx), int(info['sz'][0]/info['fov_repeats']-ndeadrows), info['sz'][1]-ndeadcols_r-ndeadcols_l))
            while k <= max_idx:  # info['max_idx']:
                # print(k)
                #ndeadcols = find_deadbands(filename)

                data = sbxread(filename, k, batch_size,**kwargs)
                data = np.transpose(data[channel_i, ndeadcols_l:-ndeadcols_r, ndeadrows:, :], axes=(2, 1, 0))
                print(k, min((k + batch_size, info['max_idx'])))
                dset[k:min((k + batch_size, info['max_idx'])), :, :] = data
                f.flush()
                k += batch_size
                #tifffile.imwrite('test_tif.tif',dset)
 
    return h5fname

def sbx2tiff(filename, channel_i=-1, batch_size=900, dataset="data", output_name=None, max_idx=None,
           force_2chan=False, nplanes=1):
    info = loadmat(filename + '.mat')  # ['info']
    if force_2chan:
        nchan = 2
    else:
        nchan = info['nChan']
    k = 0
    if output_name is None:
        tifname = filename + '.tiff'
    else:
        tifname = output_name

    if max_idx is None:
        max_idx = info['max_idx']

    if nplanes is None:
        if len(info['otwave'])>0:
            nplanes = info['otwave'].shape[0]
        else:
            nplanes = 1

    if batch_size % nplanes != 0:
        print('batch size have to be divisible by plane number!')
        quit()

    base, last = os.path.split(tifname)
    os.makedirs(base, exist_ok=True)
    print(nchan)
    print(nplanes)
    # Could potentially use the 'sbxreader' to make everything easier, but need to figure out which frame would bediscard when 
    # the number of frame is not divisible by the number of planes

    ndeadcols,ndeadrows = find_deadbands(filename)
    for chan in range(info['nChan']):
        #print(chan)
        for p in range (nplanes):
            k=0
            #print(p)
            with h5py.File(tifname, 'w') as f:
                dset = f.create_dataset(dataset,(int((max_idx-p-1)/ nplanes)+1, int(info['sz'][0]/info['fov_repeats']-ndeadrows), info['sz'][1]-ndeadcols))
        #if channel_i == -1:
                while k <= max_idx:  # info['max_idx']:

                    data = sbxread(filename, k, batch_size)
                    data = np.transpose(data[:, ndeadcols:, ndeadrows:, :], axes=(0, 3, 2, 1))
                    #print(data.shape)
                    print(min(int((k + batch_size)/nplanes),int((max_idx-p-1)/ nplanes)+1))
                    print(np.squeeze(data[chan,range(p,data.shape[1],nplanes),:,:]).shape)
                    dset[int(k/nplanes):min(int((k + batch_size)/nplanes),int((max_idx-p-1)/ nplanes)+1), :, :] = np.squeeze(data[chan,range(p,data.shape[1],nplanes),:,:])


                    # channel 0
                    # for chan in range(info['nChan']): # keep this loop as true info['nChan'] to avoid indexing error in data
                    #     dset[k * nchan + chan:min(
                    #         (nchan * (k + batch_size) + chan, nchan * info['max_idx'])):nchan, :,
                    #     :] = np.squeeze(data[chan, :, :, :])

                    f.flush()
                    k += batch_size
                tifffile.imwrite('test_tif_'+str(chan)+'_'+str(p)+'_'+'tif',dset)
                print('1 tiff file saved')
        # else:
        #     dset = f.create_dataset(dataset, (int(max_idx), int(info['sz'][0]/info['fov_repeats']), info['sz'][1]-ndeadcols))
        #     while k <= max_idx:  # info['max_idx']:
        #         # print(k)
        #         #ndeadcols = find_deadbands(filename)

        #         data = sbxread(filename, k, batch_size)
        #         data = np.transpose(data[channel_i, ndeadcols:, :, :], axes=(2, 1, 0))
        #         print(k, min((k + batch_size, info['max_idx'])))
        #         dset[k:min((k + batch_size, info['max_idx'])), :, :] = data
        #         #f.flush()
        #         k += batch_size
        #         #tifffile.imwrite('test_tif.tif',dset)
 
    return tifname


def default_ops():
    ops = {
        # file paths
        'look_one_level_down': False,  # whether to look in all subfolders when searching for tiffs
        'fast_disk': [],
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
        'threshold_scaling': 1.,  # adjust the automatically determined threshold by this scalar multiplier
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


def set_ops(d=None):
    ops = default_ops()
    if d is not None:
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


def set_db(h5fname, d=None):
    db = default_db()
    db['h5py'] = h5fname
    if d is not None:
        for k, v in d:
            db[k] = v
    return db
