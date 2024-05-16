import numpy as np
import scipy as sp
import suite2p as s2p
import TwoPUtils as tpu
import os

def find_analog_square_wave(analog, max_voltage=1):
    """ 
    Finding leading (rising) edge of the sync pulse square wave
    after converting the analog signal to a binary
    """
    # convert to binary
    binary = np.zeros(analog.shape)
    
    if max_voltage is None:
        max_voltage = np.max(analog)
    # set values less than 10% of max voltage to 0
    binary[analog<0.1*max_voltage] = 0
    # set values greater than 90% of max to 1
    # even sampled at 30kHz, it should only take a single sample to rise above threshold
    binary[analog>0.9*max_voltage] = 1
    # pad the beginning with a zero, use diff to find the rising edge above the 90% threshold
    pulse_inds = np.where(np.ediff1d(binary,to_begin=0)==1)[0]
    
    # return indices of square wave leading edges
    return pulse_inds


def extract_thor_sync_ttls(thor_metadata):
    """
    Extracts timestamps for both imaging frames and Unity VR frames
    from ThorSync data
    """
    
    sync_data, sync_dt, sync_fs = thor_metadata.read_sync()
    
    print('Identified TTLs: \n', sync_data[0].keys()) 
    
    if "FrameOut" not in sync_data[0].keys():
        print("Missing digital scanning TTL 'FrameOut'")
        print("Using thor metadata estimated scan timing instead (may be less accurate)")
        scanning_ttls = np.copy(thor_metadata.timing)
        
    if "UnitySync" not in sync_data[0].keys():
        raise NotImplementedError("Missing analog Unity TTL 'UnitySync'; has it been renamed?")
        
    samples = sync_data[0]['UnitySync'].shape[0]
    total_time = samples*(sync_dt[0]['UnitySync']*1e-3)
    print(samples, ' samples, total time ', total_time)
    
    # time vector for ThorSync data in seconds
    unity_time_vec = np.arange(0, total_time, sync_dt[0]['UnitySync']*1e-3)
    
    # Find imaging times from TTLs
    scan_ttls = find_analog_square_wave(sync_data[0]['FrameOut'])
    scan_ttl_times = unity_time_vec[scan_ttls]
    print('%d scan ttls vs. %d timestamps in scan metadata' % (scan_ttl_times.shape[0], thor_metadata.timing.shape[0]))
    
    # Find Unity frame times from TTLs
    unity_ttls = find_analog_square_wave(sync_data[0]['UnitySync'])
    unity_ttl_times = unity_time_vec[unity_ttls]
    
    unity_fs = unity_ttls.shape/(unity_time_vec[-1])
    print('Unity sampling rate:',unity_fs)
    
    ttl_times = dict()
    ttl_times['scan'] = scan_ttl_times
    ttl_times['unity'] = unity_ttl_times
    
    return ttl_times