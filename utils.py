import os

import h5py
import numpy as np
import scipy.io as spio


def trial_matrix(obj):
    pass

def loadsbxmat(filename): # replace with s2p_preprocessing definition to deal with FOV repeats
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    info = _check_keys(data)['info']
    # Defining number of channels/size factor
    if info['channels'] == 1:
        info['nChan'] = 2
        factor = 1
    elif info['channels'] == 2:
        info['nChan'] = 1
        factor = 2
    elif info['channels'] == 3:
        info['nChan'] = 1
        factor = 2

    if info['scanmode'] == 0:
        info['recordsPerBuffer'] *= 2

    # Determine number of frames in whole file
    info['max_idx'] = int(
        os.path.getsize(filename[:-4] + '.sbx') / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 / (
                    2 - info['scanmode']) - 1)
    info['frame_rate'] = info['resfreq'] / info['config']['lines'] * (2 - info['scanmode'])

    return info


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


def sbxread(filename, k=0, N=None):
    '''
    Input: filename should be full path excluding .sbx, starting index, batch size
    By default Loads whole file at once, make sure you have enough ram available to do this
    '''
    # Check if contains .sbx and if so just truncate
    if '.sbx' in filename:
        filename = filename[:-4]

    # Load info
    info = loadmat(filename + '.mat')  # ['info']
    # print info.keys()

    # Paramters
    # k = 0; #First frame
    max_idx = info['max_idx']
    if N is None:
        N = max_idx;  # Last frame
    else:
        N = min([N, max_idx - k])

    nSamples = info['sz'][1] * info['recordsPerBuffer'] * 2 * info['nChan']
    print(nSamples, N)

    # Open File
    fo = open(filename + '.sbx')

    print(int(k) * int(nSamples))
    fo.seek(int(k) * int(nSamples), 0)
    x = np.fromfile(fo, dtype='uint16', count=int(nSamples / 2 * N))
    x = np.int16((np.int32(65535) - x).astype(np.int32) / np.int32(2))
    x = x.reshape((info['nChan'], info['sz'][1], info['recordsPerBuffer'], int(N)), order='F')

    return x


def array2h5(arr, h5fname, dataset="data"):
    with h5py.File(h5fname, 'w') as f:
        dset = f.create_dataset(dataset, data=arr)


def sbx2h5(filename, channel_i=0, batch_size=1000, dataset="data", output_name=None, max_idx=None,
           force_2chan=False):
    info = loadmat(filename + '.mat')  # ['info']
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

    base, last = os.path.split(h5fname)
    os.makedirs(base, exist_ok=True)
    with h5py.File(h5fname, 'w') as f:

        if channel_i == -1:
            dset = f.create_dataset(dataset, (int(max_idx) * nchan, info['sz'][0], info['sz'][1]))
            while k <= max_idx:  # info['max_idx']:
                print(k)
                data = sbxread(filename, k, batch_size)
                data = np.transpose(data[:, :, :, :], axes=(0, 3, 2, 1))

                print(k, min((k + batch_size, info['max_idx'])))
                # channel 0
                for chan in range(info['nChan']): # keep this loop as true info['nChan'] to avoid indexing error in data
                    dset[k * nchan + chan:min(
                        (nchan * (k + batch_size) + chan, nchan * info['max_idx'])):nchan, :,
                    :] = np.squeeze(data[chan, :, :, :])

                f.flush()
                k += batch_size
        else:
            dset = f.create_dataset(dataset, (int(max_idx), info['sz'][0], info['sz'][1]))
            while k <= max_idx:  # info['max_idx']:
                print(k)
                data = sbxread(filename, k, batch_size)
                data = np.transpose(data[channel_i, :, :, :], axes=(2, 1, 0))
                print(k, min((k + batch_size, info['max_idx'])))
                dset[k:min((k + batch_size, info['max_idx'])), :, :] = data
                f.flush()
                k += batch_size

    return h5fname

