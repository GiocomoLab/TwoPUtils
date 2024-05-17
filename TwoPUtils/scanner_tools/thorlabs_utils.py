import numpy as np
import scipy as sp
import suite2p as s2p
import TwoPUtils as tpu
import os

import sys
import abc
import glob
import time

import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import tables


def find_analog_square_wave(analog, max_voltage=1):
    """ 
    Finding leading (rising) edges of the sync pulse square wave
    after converting the analog signal to a binary
    
    :param analog: analog square wave signal
    :param max_voltage: maximum voltage of the analog signal
    :return: pulse_inds - indices of the rising edges above threshold
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
    
    :param thor_metadata: a metadata class for the scan and sync generated
        by HaussIO (see below)
    :return: ttl_times, a dictionary of ttl timestamps for 'scan' and 'unity'
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



"""
Below here is mostly a copy of the haussmeister modules 
for importing and exporting 2p imaging datasets from ThorLabs.

(c) 2015 C. Schmidt-Hieber
GPLv3
original: https://github.com/neurodroid/haussmeister/blob/master/haussmeister/haussio.py

modified lightly by Mari Sosa 2024
"""

# import bottleneck as bn

# import sima
# # import tifffile
# try:
#     from skimage.external import tifffile
# except ImportError:
#     from sima.misc import tifffile
# import tifffile as tifffile_new
# try:
#     import libtiff
# except (ImportError, NameError, ValueError):
#     sys.stdout.write("Couldn't import libtiff\n")

# try:
#     from . import movies
# except (SystemError, ValueError):
#     import movies

# default filenames for raw files
THOR_RAW_FN = "Image_0001_0001.raw"
PRAIRIE_RAW_FN = "CYCLE_000001_RAWDATA_"
XZ_BIN = "/usr/local/bin/xz"

class HaussIO(object):
    """
    Base class for objects representing 2p imaging data.

    Attributes
    ----------
    xsize : float
        Width of frames in specimen dimensions (e.g. um)
    ysize : float
        Height of frames in specimen dimensions (e.g. um)
    xpx : int
        Width of frames in pixels
    ypx : int
        Height of frames in pixels
    flyback : int
        Number of flyback frames
    zplanes: int
        Number of scanned planes
    timing : numpy.ndarray
        Time points of frame acquisitions
    frame_rate : float
        Acquisition rate in frames per seconds
    frame_rate : float
        Acquisition interval in seconds
    movie_fn : str
        File path (full path) for exported movie
    scale_png : str
        File path (full path) for png showing scale bar
    sima_dir : str
        Full path to directory for sima exports
    basefile : str
        File name trunk for individual tiffs (without path)
        (e.g. ``ChanA_0001_``)
    filetrunk : str
        Full path and file name trunk for individual tiffs
        (e.g. ``/home/cs/data/ChanA_0001_``)
    ffmpeg_fn : str
        File name filter (full path) used as input to ffmpeg
        (e.g. ``/home/cs/data/ChanA_0001_%04d.tif``)
    filenames : list of str
        List of file paths (full paths) of individual tiffs
    width_idx : str
        Width of index string in file names
    maxtime : float
        Limit data to maxtime
    """
    def __init__(self, dirname, chan='A', xml_path=None, sync_path=None,
                 width_idx=4, maxtime=None):

        self.raw_array = None
        self.mptifs = None
        self.maxtime = maxtime

        self.dirname = os.path.abspath(dirname)
        self.chan = chan
        self.width_idx = width_idx

        self._get_filenames(xml_path, sync_path)
        if self.sync_path is None:
            self.sync_episodes = None
            self.sync_xml = None

        sys.stdout.write("Reading experiment settings for {0}... ".format(
            self.dirname))
        sys.stdout.flush()
        if self.xml_name is not None:
            self.xml_root = ET.parse(self.xml_name).getroot()
        self._get_dimensions()
        self._get_timing()
        self._get_sync()
        self.dt = np.mean(np.diff(self.timing))
        self.frame_rate = 1.0/self.dt # frame rate to get dt comes directly from the xml metadata
        self.nframes = len(self.timing)
        sys.stdout.write("done\n")

        if self.maxtime is not None:
            self.iend = np.where(self.timing >= self.maxtime)[0][0]
            self.filenames = self.filenames[:self.iend]
        else:
            self.iend = None

        if self.mptifs is not None:
            self.nframes = np.sum([
                len(mptif.pages)-self.pagesoffset for mptif in self.mptifs])
            if self.nframes == 0:
                self.nframes = np.sum([
                    len(mptif.IFD) for mptif in self.mptifs])
        elif xml_path is None:
            if self.rawfile is None or not os.path.exists(self.rawfile):
                try:
                    assert(len(self.filenames) <= self.nframes)
                except AssertionError as err:
                    print(len(self.filenames), self.nframes)
                    raise err
        else:
            if self.rawfile is None or not os.path.exists(self.rawfile):
                if len(self.filenames) != self.nframes:
                    self.nframes = len(self.filenames)

    @abc.abstractmethod
    def _get_dimensions(self):
        return

    @abc.abstractmethod
    def _get_timing(self):
        return

    @abc.abstractmethod
    def _get_sync(self):
        return

    @abc.abstractmethod
    def read_sync(self):
        return

    @abc.abstractmethod
    def read_raw(self):
        return

    # def raw2tiff(self, mp=False):
    #     arr = self.read_raw()
    #     if not mp:
    #         for ni, img in enumerate(arr):
    #             sys.stdout.write(
    #                 "\r{0:6.2%}".format(float(ni)/arr.shape[0]))
    #             sys.stdout.flush()
    #             tifffile.imsave(os.path.join(
    #                 self.dirname_comp,
    #                 self.basefile + self.format_index(ni+1)) + ".tif", img)
    #             sys.stdout.write("\n")
    #     else:
    #         tifffile.imsave(os.path.join(
    #             self.dirname_comp, self.basefile + "mp.tif"), arr)

    # def tiff2raw(self, path=None, compress=True):
    #     if path is None:
    #         path_f = self.dirname_comp
    #     else:
    #         path_f = path
    #     rawfn = os.path.join(path_f, THOR_RAW_FN)
    #     assert(not os.path.exists(rawfn))
    #     compressfn = rawfn + ".xz"
    #     assert(not os.path.exists(compressfn))

    #     if not os.path.exists(rawfn):
    #         sys.stdout.write("Reading files...")
    #         sys.stdout.flush()
    #         t0 = time.time()
    #         arr = self.asarray_uint16()
    #         assert(len(arr.shape) == 3)
    #         sys.stdout.write(" done in {0:.2f}s\n".format(time.time()-t0))
    #         compress_np(arr, path_f, THOR_RAW_FN, compress=compress)

    def _get_filenames(self, xml_path, sync_path):
        self.dirname_comp = self.dirname.replace("?", "n")
        self.movie_fn = self.dirname_comp + ".mp4"
        self.scale_png = self.dirname_comp + "_scale.png"
        self.sima_dir = self.dirname_comp + ".sima"
        self.rawfile = None
        self.sync_path = sync_path

    def get_normframe(self):
        """
        Return a representative frame that will be used to normalize
        the brightness in movies

        Returns
        -------
        arr : numpy.ndarray
            Frame converted to numpy.ndarray
        """
        if self.mptifs is None and (self.rawfile is None or not os.path.exists(self.rawfile)):
            if "?" in self.dirname:
                normdir = self.dirnames[int(np.round(len(self.dirnames)/2.0))]
                normtrunk = self.filetrunk.replace(
                    self.dirname, normdir)
                nframes = len(
                    glob.glob(os.path.join(normdir, self.basefile + "*.tif")))
                normframe = normtrunk + self.format_index(int(nframes/2)) + ".tif"
            else:
                normframe = self.filetrunk + self.format_index(
                    int(len(self.filenames)/2)) + ".tif"
            sample = Image.open(normframe)
            arr = np.asarray(sample, dtype=np.float)
        else:
            arr = self.read_raw()[int(self.nframes/2)]

        return arr

    ##### tosuite2p not currently used, keeping it here in case it's useful
    def tosuite2p(self, ops):
        """
        Writes out files in suite2p format (not currently used)

        Parameters
        ----------
        ops : dict
            suite2p options dictionary
        """
        nplanes = ops['nplanes']
        nchannels = ops['nchannels']
        ops1 = []
        # open all binary files for writing
        reg_file = []
        if nchannels>1:
            reg_file_chan2 = []
        for j in range(0,nplanes):
            ops['save_path'] = os.path.join(ops['save_path0'], 'suite2p', 'plane%d'%j)
            if ('fast_disk' not in ops) or len(ops['fast_disk']) == 0:
                ops['fast_disk'] = ops['save_path0']
            ops['fast_disk'] = os.path.join(ops['fast_disk'], 'suite2p', 'plane%d'%j)
            ops['ops_path'] = os.path.join(ops['save_path'],'ops.npy')
            ops['reg_file'] = os.path.join(ops['fast_disk'], 'data.bin')
            if nchannels>1:
                ops['reg_file_chan2'] = os.path.join(ops['fast_disk'], 'data_chan2.bin')
            if not os.path.isdir(ops['fast_disk']):
                os.makedirs(ops['fast_disk'])
            if not os.path.isdir(ops['save_path']):
                os.makedirs(ops['save_path'])
            ops1.append(ops.copy())
            reg_file.append(open(ops['reg_file'], 'wb'))
            if nchannels>1:
                reg_file_chan2.append(open(ops['reg_file_chan2'], 'wb'))

        rawdata = self.read_raw()
        assert(rawdata.dtype == np.uint16)
        nbatch = int(np.round(nplanes*nchannels*np.ceil(ops['batch_size']/(nplanes*nchannels))))
        nframes_all = rawdata.shape[0]
        # loop over all tiffs
        i0 = 0
        if nplanes > 1:
            rawdata.ndim == 4
        while True:
            irange = np.arange(i0, min(i0+nbatch, nframes_all), 1)
            if irange.size==0:
                break
            if nplanes > 1:
                im = rawdata[irange, :, : ,:]
            else:
                im = rawdata[irange, :, :]
            if i0==0:
                if nplanes > 1:
                    for j in range(0, nplanes):
                        ops1[j]['meanImg'] = np.zeros((im.shape[2],im.shape[3]),np.float32)
                        if nchannels>1:
                            ops1[j]['meanImg_chan2'] = np.zeros((im.shape[2],im.shape[3]),np.float32)
                else:
                    ops1[0]['meanImg'] = np.zeros((im.shape[1],im.shape[2]),np.float32)
                    ops1[0]['meanImg_chan2'] = np.zeros((im.shape[1],im.shape[2]),np.float32)
                if nplanes > 1:
                    for j in range(0,nplanes):
                        ops1[j]['nframes'] = 0
                else:
                    ops1[0]['nframes'] = 0
            nframes = im.shape[0]
            for j in range(0,nplanes):
                if nplanes > 1:
                    im2write = im[:,j,:,:]
                else:
                    im2write = im[np.arange(j, nframes, nplanes*nchannels),:,:]
                reg_file[j].write(bytearray(im2write.astype('int16')))
                ops1[j]['meanImg'] = ops1[j]['meanImg'] + im2write.astype(np.float32).sum(axis=0)
                if nchannels>1:
                    if nplanes > 1:
                        im2write = im[:,j,:,:]
                    else:
                        im2write = im[np.arange(j+1, nframes, nplanes*nchannels),:,:]
                    reg_file_chan2[j].write(bytearray(im2write.astype('int16')))
                    ops1[j]['meanImg_chan2'] += im2write.astype(np.float32).sum(axis=0)
                ops1[j]['nframes'] += im2write.shape[0]
            i0 += nframes
        # write ops files
        do_registration = ops['do_registration']
        for ops in ops1:
            ops['Ly'] = im2write.shape[1]
            ops['Lx'] = im2write.shape[2]
            if not do_registration:
                ops['yrange'] = np.array([0,ops['Ly']])
                ops['xrange'] = np.array([0,ops['Lx']])
            ops['meanImg'] /= ops['nframes']
            if nchannels>1:
                ops['meanImg_chan2'] /= ops['nframes']
            np.save(ops['ops_path'], ops)
        # close all binary files and write ops files
        for j in range(0,nplanes):
            reg_file[j].close()
            if nchannels>1:
                reg_file_chan2[j].close()
        for j in range(nplanes):
            print(ops1[j]['nframes'])
        return ops1
    
    ######

    def asarray(self):
        return np.array(self.tosima().sequences[0]).squeeze()

    def asarray_uint16(self):
        arr = np.array([
            np.array(Image.open(fn, 'r'), dtype=np.uint16)
            for fn in self.filenames])
        try:
            assert(arr.dtype == np.uint16)
            assert(arr.shape[0] == self.nframes)
        except AssertionError as err:
            print(arr.dtype)
            print(arr.shape)
            print(self.nframes, self.xpx, self.ypx)
            raise err

        return arr
   

    def get_scale_bar(self, prop=1/8.0):
        """
        Returns lengths in specimen dimensions (e.g. um) and in pixels
        of a scale bar that fills the given fraction of the width of the
        image

        Parameters
        ----------
        prop : float, optional
            Length of scale bar expressed as fraction of image width

        Returns
        -------
        scale_length_int : int
            Scale bar length in specimen dimensions (e.g. um)
        scale_length_px : int
            Scale bar length in pixels
        """

        # Reasonable scale bar length (in specimen dimensions, e.g. um)
        # given the width of the image:
        scale_length_float = self.xsize * prop

        # Find closest integer that looks pretty as a scale bar label:
        nzeros = int(np.log10(scale_length_float))
        closest_int = np.round(scale_length_float/10**nzeros)

        if closest_int <= 5:
            scale_length_int = closest_int * 10**nzeros
        else:
            # Closer to 5 or closer to 10?
            if 10-closest_int < closest_int-5:
                scale_length_int = 1 * 10**(nzeros+1)
            else:
                scale_length_int = 5 * 10**(nzeros)

        scale_length_px = scale_length_int * self.xpx/self.xsize

        return scale_length_int, scale_length_px


    def plot_scale_bar(self, ax):
        """
        Add scale bar to a matplotlib axis

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            matplotlib axes on which to plot the scale bar, as
            e.g. returned by 'fig.add_suplot(1,1,1)'
        """
        sb_int, sb_px = self.get_scale_bar()
        scale_text = u"{0:0d} $\mu$m".format(int(sb_int))
        ax.plot([self.xpx/10.0,
                 self.xpx/10.0+sb_px],
                [self.ypx/20.0,
                 self.ypx/20.0],
                lw=self.xpx/125.0, color='w')
        ax.text(self.xpx/10.0+sb_px/2.0,
                self.ypx/15.0,
                scale_text,
                va='top', ha='center', color='w')

    def format_index(self, n, width_idx=None):
        """
        Return formatted index string

        Parameters
        ----------
        n : int or str
            Index as int, or "?" (returns series of "?"), or "%" (returns
            old-school formatter)
        width_idx : int, optional
            Override default width of index string. Default: None

        Returns
        -------
        format : string
            Formatted index string, or series of "?", or old-school formatter
        """
        if width_idx is None:
            width_idx = self.width_idx

        if isinstance(n, str):
            if n == "?":
                ret = "?"
                for nq in range(width_idx-1):
                    ret += "?"
                return ret
            elif n == "%":
                return "%0{0:01d}d".format(width_idx)
        else:
            return "{0:0{width}d}".format(n, width=width_idx)


class ThorHaussIO(HaussIO):
    """
    Object representing 2p imaging data acquired with ThorImageLS
    """
    def __init__(self, dirname, chan='A', xml_path=None, sync_path=None,
                 width_idx=4, maxtime=None):
        self.pagesoffset = 1
        super(ThorHaussIO, self).__init__(
            dirname, chan, xml_path, sync_path,
            width_idx, maxtime)

    def _get_filenames(self, xml_path, sync_path):
        super(ThorHaussIO, self)._get_filenames(xml_path, sync_path)
        self.basefile = "Chan" + self.chan + "_0001_0001_0001_" # to-do: update this, not our naming scheme
        self.filetrunk = os.path.join(self.dirname, self.basefile)
        if "?" in self.filetrunk:
            self.dirnames = sorted(glob.glob(self.dirname))
            self.ffmpeg_fn = "'" + self.filetrunk + self.format_index(
                "?") + ".tif'"
        else:
            self.dirnames = [self.dirname]
            self.ffmpeg_fn = self.filetrunk + self.format_index("%") + ".tif"

        if xml_path is None:
            self.xml_name = self.dirname + "/Experiment.xml"
        else:
            self.xml_name = xml_path
        if "?" in self.xml_name:
            self.xml_name = sorted(glob.glob(self.xml_name))[0]
        if "?" in self.dirname:
            self.filenames = []
            for dirname in self.dirnames:
                filenames_orig = sorted(glob.glob(os.path.join(
                    dirname, self.basefile + "*.tif")))
                nf = len(self.filenames)
                self.filenames += [os.path.join(
                    self.dirname_comp, self.basefile +
                    self.format_index(nf+nfno) + ".tif")
                    for nfno, fno in enumerate(filenames_orig)]
            rawfiles = sorted(glob.glob(os.path.join(
                dirname, "*.raw*")))
            self.rawfile = rawfiles[0]
        else:
            self.filenames = sorted(glob.glob(self.filetrunk + "*.tif"))
            rawfiles = sorted(glob.glob(os.path.join(
                self.dirname, "*.raw*")))
            if 'Image' in rawfiles:    
                self.rawfile = os.path.join(self.dirname_comp, rawfiles[0])
                if os.path.exists(self.rawfile + ".xz"):
                    self.rawfile = self.rawfile + ".xz"
            else:
                self.rawfile = None

        

    def _get_dimensions(self):
        self.xsize, self.ysize = None, None
        self.xpx, self.ypx, self.flyback, self.plane, self.zplanes = None, None, 0, 0, None
        self.zenable = None
        for child in self.xml_root:
            if child.tag == "LSM":
                self.xpx = int(child.attrib['pixelX'])
                self.ypx = int(child.attrib['pixelY'])
                if int(child.attrib['averageMode']) == 1:
                    self.naverage = int(child.attrib['averageNum'])
                else:
                    self.naverage = None
                if 'widthUM' in child.attrib:
                    self.xsize = float(child.attrib['widthUM'])
                    self.ysize = float(child.attrib['heightUM'])
            elif child.tag == "Sample":
                for grandchild in child:
                    if grandchild.tag == "Wells":
                        for ggrandchild in grandchild:
                            if self.xsize is None:
                                self.xsize = float(
                                    ggrandchild.attrib['subOffsetXMM'])*1e3
                            if self.ysize is None:
                                self.ysize = float(
                                    ggrandchild.attrib['subOffsetYMM'])*1e3
            if child.tag == "Streaming":
                    self.flyback = int(child.attrib['flybackFrames'])
                    self.zenable = int(child.attrib['zFastEnable'])
            if child.tag == "ZStage":
                    self.plane= int(child.attrib['steps'])
        if self.plane != None:
            if self.zenable > 0:
                self.zplanes = self.flyback + self.plane
                print('The number of zplanes are:', self.zplanes)
        else:
            self.zplanes = 1

    def _get_timing(self):
        """
        Gets imaging frame timestamps from the timing.txt file,
        if it exists (it does not exist for us?), otherwise
        generates a vector of assumed timestamps using the sampling rate
        """
        
        if "?" in self.dirname:
            dirname_wildcard = self.dirname[
                :len(os.path.dirname(self.xml_name))] + "/timing.txt"
            timings = sorted(glob.glob(dirname_wildcard))
            self.timing = np.loadtxt(timings[0])
            for timing in timings[1:]:
                self.timing = np.concatenate([
                    self.timing, np.loadtxt(timing)+self.timing[-1]])
        else:
            timingfn = os.path.dirname(self.xml_name) + "/timing.txt"
            if os.path.exists(timingfn):
                self.timing = np.loadtxt(timingfn)
            else:
                for child in self.xml_root:
                    if child.tag == "LSM":
                        framerate = float(child.attrib['frameRate'])
                    if child.tag == "Streaming":
                        nframes = int(child.attrib['frames'])
                dt = 1.0/framerate
                self.timing = np.arange((nframes), dtype=float) * dt
                if self.zplanes != None:
                    if self.zplanes > 1:
                        nframes = int(nframes/self.zplanes)
                        print('The number of frames is: ', nframes)

    def _get_sync(self):
        """
        Find ThorSync data
        """
        if self.sync_path is None:
            return

        self.sync_paths = sorted(glob.glob(self.sync_path))
        self.sync_episodes = [sorted(glob.glob(sync_path + "/Episode*.h5"))
                              for sync_path in self.sync_paths]
        self.sync_xml = [sync_path + "/ThorRealTimeDataSettings.xml"
                         for sync_path in self.sync_paths]

    def _find_dt(self, name, nsync=0):
        """
        Find delta time between samples, in millseconds
        """
        self.sync_root = ET.parse(self.sync_xml[nsync]).getroot()
        for child in self.sync_root:
            if child.tag == "DaqDevices":
                for cchild in child:
                    if cchild.tag == "AcquireBoard":
                        for ccchild in cchild:
                            if ccchild.tag == "DataChannel":
                                if ccchild.attrib['alias'] == name:
                                    board = cchild
        for cboard in board:
            if cboard.tag == "SampleRate":
                if cboard.attrib['enable'] == "1":
                    return 1.0/float(cboard.attrib['rate']) * 1e3

                
    def _find_fs(self, name, nsync=0):
        """
        Find sampling rate of NIDAQ IO channels in Hz
        """
        self.sync_root = ET.parse(self.sync_xml[nsync]).getroot()
        for child in self.sync_root:
            if child.tag == "DaqDevices":
                for cchild in child:
                    if cchild.tag == "AcquireBoard":
                        for ccchild in cchild:
                            if ccchild.tag == "DataChannel":
                                if ccchild.attrib['alias'] == name:
                                    board = cchild
        for cboard in board:
            if cboard.tag == "SampleRate":
                if cboard.attrib['enable'] == "1":
                    return float(cboard.attrib['rate'])

    def read_sync(self):
        """
        Read ThorSync data and extract numpy arrays from
        analog and digital IO signals
        """
        if self.sync_path is None:
            return None
        sync_data = [] # digital or analog voltages
        sync_dt = [] # time between samples in ms
        sync_fs = [] # sampling rate in Hz from DAQ card
        for epi_files in self.sync_episodes:
            for episode in epi_files:
                sync_data.append({})
                sync_dt.append({})
                sync_fs.append({})
                print(episode)
                h5 = tables.open_file(episode)
                for el in h5.root.DI:
                    sync_data[-1][el.name] = np.squeeze(el)
                    sync_dt[-1][el.name] = self._find_dt(
                        el.name, len(sync_dt)-1)
                    sync_fs[-1][el.name] = self._find_fs(
                                el.name, len(sync_fs)-1)

                for el in h5.root.CI:
                    # this is the frame counter
                    sync_data[-1][el.name] = np.squeeze(el)
                    sync_dt[-1][el.name] = self._find_dt(
                        el.name, len(sync_dt)-1)
                    sync_fs[-1][el.name] = self._find_fs(
                                el.name, len(sync_fs)-1)
                for el in h5.root.AI:
                    sync_data[-1][el.name] = np.squeeze(el)
                    sync_dt[-1][el.name] = self._find_dt(
                        el.name, len(sync_dt)-1)
                    sync_fs[-1][el.name] = self._find_fs(
                                el.name, len(sync_fs)-1)

                h5.close()       

        return sync_data, sync_dt, sync_fs

    def read_raw(self):
        if self.raw_array is None:
            if os.path.exists(
                    os.path.join(
                        self.dirname_comp, THOR_RAW_FN)):
                shapefn = os.path.join(
                    self.dirname_comp, THOR_RAW_FN[:-3] + "shape.npy")
                if os.path.exists(shapefn):
                    shape = np.load(shapefn)
                else:
                    if (self.zplanes != None):
                        if (self.zplanes>1):
                            self.nframes = int(self.nframes/self.zplanes)
                            shape = (self.nframes, self.zplanes, self.xpx, self.ypx)
                    else:
                        shape = (self.nframes, self.xpx, self.ypx)
                self.raw_array = raw2np(self.rawfile, shape)[:self.iend]
            else:
                shapes = [
                    np.load(
                        os.path.join(rawdir, THOR_RAW_FN[:-3] + "shape.npy"))
                    for rawdir in sorted(glob.glob(self.dirname))]
                self.raw_array = np.concatenate([
                    raw2np(os.path.join(rawdir, THOR_RAW_FN), shape)
                    for shape, rawdir in zip(
                            shapes, sorted(glob.glob(self.dirname)))])

        return self.raw_array


def load_haussio(dirname, ftype=None):
    if ftype is None:
        # Try to get ftype from files in directory
        basename = os.path.basename(dirname)
        if os.path.exists(os.path.join(dirname, basename+".env")):
            ftype = "prairie"
        elif os.path.exists(os.path.join(dirname, "Experiment.xml")):
            ftype = "thor"
        else:
            # find all tiffs:
            tiffs = glob.glob(os.path.join(dirname, "*.tif"))
            # attempt to open first tiff:
            if len(tiffs):
                sampletiff = tifffile.TiffFile(tiffs[0])
                sampleifd = sampletiff.info()
                if 'Exposure' in sampleifd and 'Gain' in sampleifd:
                    ftype = "doric"

    if ftype is None:
        raise RuntimeError("File autodetection only for ThorLabs and Prairie files")

    if ftype == "thor":
        return ThorHaussIO(dirname, 'A')
 


