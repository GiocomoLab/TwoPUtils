import os
import warnings
from abc import ABC
from glob import glob

import dill
import numpy as np
import scipy as sp
import pandas as pd
import math

import social_int_analyses
from social_int_analyses import sleap_utils as slp

from .scanner_tools import sbx_utils
from .scanner_tools import thorlabs_utils
from . import preprocessing as pp
from . import spatial_analyses


def save_session(obj, output_basedir):
    """
    Save an instance of a Session or SessionInfo
    :param obj: Session or SessionInfo instance
    :param output_basedir: path to directory where pickled sessions are saved
    :return:
    """

    # make sure obj is the right type
    assert isinstance(obj, SessionInfo) | isinstance(obj,
                                                     Session), "obj must be a Session or SessionInfo class instance"

    # check that output_basedir exists
    assert os.path.exists(output_basedir), "output_basedir does not exist"
    # check that mouse, date, scene, and session exist and are not None
    for attr in ("mouse", "date", "scene", "session"):
        assert hasattr(obj, attr), "object missing %s attribute" % attr
        if getattr(obj, attr) is None:
            raise NameError("Will not save session if %s is NoneType" % attr)

    # save pickled instance of class
    pkldir = os.path.join(output_basedir, obj.mouse, obj.date)
    pklfile = os.path.join(pkldir, "%s_%d.pkl" % (obj.scene, obj.session))
    os.makedirs(pkldir, exist_ok=True)
    with open(pklfile, 'wb') as f:
        dill.dump(obj, f)


class SessionInfo:
    """Base class for any 2P session
    """

    def __init__(self, **kwargs):
        """
        session information

        basedir_2P = None  # base directory to find 2P data - see self._check_minimal_keys()
        basedir_VR = None  # base directory to find VR Data - see self._check_minimal_keys()
        mouse = None  # mouse name (string)
        date = None  # date of session (dd_mm_yyyy)
        scene = None  # name of unity scene
        session = None  # int, session number
        scan_number = None  # Neurolabware only
        scanner = None  # ['NLW','ThorLabs','Bruker']
        VR_only = None  # bool, whether the session is only vr data
        scan_file = None  # string, path to neural data
        scanheader_file = None  # string, path to header file for scan
        vr_filename = None  # string, path to vr sqlite file
        s2p_path = None  # string, suite2p path
        n_planes = 1 # int, number of imaging planes
        prompt_for_keys = True # bool, whether or not to run through prompts for minimal keys
        fixed_wheel = False #bool, if wheel is fixed will not look for teleports
        tunnel_data = False # bool, if true, will look for tunnel SLEAP data 


        """
        # session information
        # initialize attributes
        self.basedir_2P = None  # base directory to find 2P data - see self._check_minimal_keys()
        self.basedir_VR = None  # base directory to find VR Data - see self._check_minimal_keys()
        self.mouse = None  # mouse name (string)
        self.date = None  # date of session (dd_mm_yyyy)
        self.scene = None  # name of unity scene
        self.session = None  # int, session number
        self.fix_wheel = False # social interaction project fixed wheel 
        self.scan_number = None  # Neurolabware only
        self.scanner = None  # ['NLW','ThorLabs','Bruker']
        self.VR_only = None  # bool, whether the session is only vr data
        self.scan_file = None  # string, path to neural data
        self.scanheader_file = None  # string, path to header file for scan
        self.vr_filename = None  # string, path to vr sqlite file
        self.s2p_path = None  # string, suite2p path
        self.n_planes = 1  # int, number of imaging planes
        self.n_channels = 1  # int, number of functional channels
        self.prompt_for_keys = False  # bool, whether or not to run through prompts for minimal keys
        self.verbose = False
        self.fixed_wheel = False
        self.tunnel_data = False

        self.__dict__.update(kwargs)  # update keys based on inputs
        # if want to receive prompts for minimal keys
        if self.prompt_for_keys:
            self._check_minimal_keys()
        else:
            # check that provided keys are the right type
            if self.verbose:
                warnings.warn("skipping checking keys, remaining initialization not guaranteed to work")

        # check for VR data
        if self.vr_filename is None:
            self._check_for_VR_data()

        if not self.VR_only:
            # check for raw 2P data
            self._check_for_2P_data()

            # check for suite 2P data
            self._check_for_suite2P_data()

            # check for other sessions that 2P data is aligned to
            # self._check_for_coaligned_suite2p_sessions()

        # print available fields
        # self.print_session_info()

    def _check_minimal_keys(self):
        """
        checks to make sure initialization of class has proper attributes to prevent other functions from failing
        :return:
        """

        print(
            "Expected directory tree for VR Data"
            "basedir_VR\\mouse\\date_folder\\scene\\scene_sessionnumber.sql")
        print(
            "Expected directory tree for 2P Data "
            "basedir_2P\\mouse\\date_folder\\scene\\scene_sessionnumber_scannumber.sbx\mat")
        print(
            "Bruker compatibility to be added; Thorlabs B scope compatibility in beta"
        )
        if self.basedir_VR is None:
            print("What is the base directory for your VR data?")
            self.basedir = input()

        if self.basedir_2P is None and not self.VR_only:
            print("What is the base directory for you 2P data?")
            self.basedir_2P = input()

        if self.mouse is None:
            print("Mouse ID?")
            self.mouse = input()

        if self.date is None:
            print("Date of experiment (dd_mm_yyy)?")
            self.date = input()

        if self.scene is None:
            print("Name of Unity scene?")
            self.scene = input()

        if self.session is None:
            print("Session number")
            _session_number = input()
            self.session = int(_session_number)

        if self.VR_only is None:

            def _recursive_input_check():
                print("VR only? (1=yes,0=no)")
                _vr_only = input()
                if _vr_only.lower in ('true', 1, 't', 'y', 'yes', 'totes', 'yep', 'roger'):
                    print("Setting VR_only to True, skipping VR alignment to 2P")
                    self.VR_only = True
                elif _vr_only.lower in ('false', 0, 'f', 'n', 'no', 'nope', 'negative', 'no way'):
                    self.VR_only = False
                else:
                    print("Didn't understand input")
                    _recursive_input_check()

            _recursive_input_check()

        if not self.VR_only:
            if self.scanner is None:
                while self.scanner not in ("NLW", "Thorlabs", "Bruker", "skip"):
                    print("Which microscope? [NLW,ThorLabs,Bruker,skip]")
                    scanner = input()
                    self.scanner = scanner

            if self.scanner == "NLW":
                print("Scan Number?")
                scannumber = input()
                self.scan_number = int(scannumber)

    def print_session_info(self):
        """
        print session information
        :return:
        """
        # for each attribute
        for attr in dir(self):
            # if not reserved field name and not function
            if not attr.startswith('__') and not callable(getattr(self, attr)):
                print(attr, ":", getattr(self, attr))

    def _check_for_VR_data(self):
        """
        ensure VR data file exists
        :return:
        """

        # look for VR Data
        if self.vr_filename is None:
            try:
                self.vr_filename = os.path.join(self.basedir_VR, self.mouse,
                                                self.date, "%s_%d.sqlite" % (self.scene, self.session))
            except:
                pass
        else:
            if not os.path.exists(self.vr_filename):
                if self.verbose:
                    warnings.warn("VR File does not exist!", self.vr_filename)

    def _check_for_2P_data(self):
        """
        ensure 2P data exists
        :return:
        """

        # set paths
        if self.scanner == "NLW":

            if self.scanheader_file is None:
                # find paths to sbx file and mat file
                self.scanheader_file = os.path.join(self.basedir, self.mouse, self.date,
                                                    self.scene,
                                                    "%s_%03d_%03d.mat" % (self.scene, self.session, self.scan_number))
            if not os.path.exists(self.scanheader_file):
                if self.verbose:
                    warnings.warn("Could not find sbxmat file at %s" % self.scanheader_file)


            if self.scan_file is None:
                self.scan_file = os.path.join(self.basedir, self.mouse, self.date,
                                              self.scene,
                                              "%s_%03d_%03d.sbx" % (self.scene, self.session, self.scan_number))
            if not os.path.exists(self.scan_file):
                if self.verbose:
                    warnings.warn("Could not find sbx file at %s" % self.scan_file)


        elif self.scanner == "ThorLabs":
            if self.scanheader_file is None:
                # find paths to the ThorImage xml file
                self.scanheader_file = os.path.join(self.basedir, "Experiment.xml")
            if not os.path.exists(self.scanheader_file):
                if self.verbose:
                    warnings.warn("Could not find scan header file at %s" % self.scanheader_file)
            
            if self.scan_file is None:       
                self.scan_file = glob(os.path.join(self.basedir, 'Image_scan*.tif'))[0]

            if not os.path.exists(self.scan_file):
                if self.verbose:
                    warnings.warn("Could not find scan file at %s" % self.scan_file)

        elif self.scanner == "Bruker":
            raise NotImplementedError

    def _check_for_suite2P_data(self):
        """
        make sure suite2p path exists
        :return:
        """
        # look for suite2p results
        if self.s2p_path is None:
            base, _ = os.path.splitext(self.scanheader_file)
            if self.n_channels==1:
                self.s2p_path = os.path.join(base, 'suite2p')
                if not os.path.exists(self.s2p_path):
                    if self.verbose:
                        warnings.warn("Could not find suite2p path at %s" % self.s2p_path)
                else:
                    self.n_planes = len(glob(os.path.join(self.s2p_path, 'plane*')))
            else:
                self.s2p_path = os.path.join("Z:/giocomo/candong/social_interaction/calcium_imaging", self.mouse, self.date)
                # base, _ = os.path.splitext(self.scanheader_file)
                # self.s2p_path = [os.path.join(base,'suite2p')]
                
                # self.s2p_path.extend([os.path.join(base, f'chan{i+1}/suite2p') for i in range(1,self.n_channels)])

    def _check_for_coaligned_suite2p_sessions(self):
        # look for file that points to which session share ROIs
        warnings.warn("Looking for coaligned suite2p sessions is not implemented yet")
        # raise NotImplementedError


class Session(SessionInfo, ABC):
    """
    Extension of SessionInfo class that contains behavioral data and neural data
    """

    def __init__(self,  **kwargs):
        """

        :param load_pickled_sess: bool, look for and load previous instance of session in pickle_dir
        :param kwargs:
        basedir_2P = None  # base directory to find 2P data - see self._check_minimal_keys()
        basedir_VR = None  # base directory to find VR Data - see self._check_minimal_keys()
        mouse = None  # mouse name (string)
        date = None  # date of session (dd_mm_yyyy)
        scene = None  # name of unity scene
        session = None  # int, session number
        scan_number = None  # Neurolabware only
        scanner = None  # ['NLW','ThorLabs','Bruker']
        VR_only = None  # bool, whether the session is only vr data
        scan_file = None  # string, path to neural data
        scanheader_file = None  # string, path to header file for scan
        vr_filename = None  # string, path to vr sqlite file
        s2p_path = None  # string, suite2p path
        n_planes = 1 # int, number of imaging planes
        prompt_for_keys = True # bool, whether or not to run through prompts for minimal keys


        """

        # tunnel data
        self.tunnel_df = None
        
        # vr data
        self.vr_data = None
        self.trial_start_inds = None
        self.teleport_inds = None

        # neural data
        self.scan_info = None
        self.n_planes = None
        self.timeseries = {}
        self.trial_matrices = {}
        self.iscell = []
        self.plane_per_cell = np.empty((0,),dtype=float)
        self.s2p_ops = []
        self.s2p_stats = []

        # self.__dict__.update(kwargs)  # update keys based on inputs - might not need this line/called through super
        # inheritance
        super(Session, self).__init__(**kwargs)
        
        
        #     print("Multi-channel processing for %d channels" % self.n_channels)
        #     self.timeseries = {f'channel_{i}': None for i in range(self.n_channels)}
            
        #     self.trial_matrices

    @classmethod
    def from_file(cls, filename, **kwargs):
        '''
        initialize class from previous instance

        :param filename:
        :return:
        '''
        with open(filename, 'rb') as file:
            return cls(prev_sess=dill.load(file), **kwargs)
            
    def load_scan_info(self, sbx_version=2):
        if self.scanner == "NLW":
            self.scan_info = sbx_utils.loadmat(self.scanheader_file, sbx_version=sbx_version)

    def align_VR_to_2P(self, overwrite=True, run_ttl_check = False):
        
                
        if self.vr_data is None or overwrite:
            # load sqlite file as pandas array

            if "wheel" in self.vr_filename:
                df = pp.load_sqlite(self.vr_filename,fix_teleports=False)
            else:
                df = pp.load_sqlite(self.vr_filename,fix_teleports=True)
            
            if not self.VR_only:
                # feed pandas array and scene name to alignment function
                if self.scanner == "NLW":
                    self.vr_data = pp.vr_align_to_2P(df, self.scan_info, run_ttl_check = run_ttl_check, n_planes=self.n_planes)

                    # ES add multi-chan functionality 
                    if self.n_channels > 1:
                        self.chan0_vr, self.chan1_vr = pp.vr_align_to_2P(df, self.scan_info, run_ttl_check = run_ttl_check, n_planes=self.n_planes, mux = True)
                        print(self.chan0_vr.shape, self.chan1_vr.shape)

                        
                elif self.scanner == "ThorLabs":
                   
                    thor_metadata = thorlabs_utils.ThorHaussIO(self.basedir, chan='A', xml_path=None, 
                          sync_path= self.basedir + '_sync')
                    print('Only looking for one channel so far...')
                    ##
                    ttl_times = thorlabs_utils.extract_thor_sync_ttls(thor_metadata)
                    self.vr_data = pp.vr_align_to_2P_thor(df, 
                        thor_metadata, 
                        ttl_times, 
                        run_ttl_check=False,
                        n_planes = 1)
                else:
                    warnings.warn("VR alignment only implemented for Neurolabware and Thorlabs")
                    raise NotImplementedError
            else:
                self.vr_data = df

            self.trial_start_inds = self.vr_data.index[self.vr_data.tstart == 1]
            self.teleport_inds = self.vr_data.index[self.vr_data.teleport == 1]
        else:
            print("VR data already set or overwrite=False")


    def align_SLEAP_to_2P(self, overwrite=True, run_ttl_check = False):
        sleap_dir = "C:/Users/esay/data/social_interaction/SLEAPData"
        sleap_file = os.path.join(sleap_dir, self.mouse, self.date, (self.scene +'.h5') )
        
        if self.tunnel_df is None or overwrite:
            if 'Env1' in self.vr_filename:
                return
            elif "empty" in self.vr_filename:
                return
            else:
                df = slp.add_tunnel_sess(sleap_file, self)
                self.tunnel_data = pp.tunnel_align_to_2P(df, self.scan_info, run_ttl_check = run_ttl_check, n_planes=self.n_planes)
 


    def load_suite2p_data(self, which_ts=('F', 'Fneu', 'spks', 'F_chan2', 'Fneu_chan2'), custom_iscell=None,
                          frames=None, use_iscell=True, mux = False):

        if self.n_planes > 1:
            print(f"Multiplane processing for {self.n_planes} planes")
            plane = "combined"
        else:
            plane = "plane0"



        print(self.s2p_path)
        if self.n_channels>1:
            self.s2p_ops={f'channel_{i}': np.load(os.path.join(self.s2p_path[i], plane, 'ops.npy'), allow_pickle=True).all() for i in range(self.n_channels)}
        else:
            self.s2p_ops = np.load(os.path.join(self.s2p_path, plane, 'ops.npy'), allow_pickle=True).all()

        if frames is None:
            if self.n_channels>1:
                frames = slice(0, self.s2p_ops['channel_1']['nframes'])
                if mux:
                    chan0_frames = slice(0, 2*(self.s2p_ops['channel_0']['nframes']), 2)
                    chan1_frames = slice(1, 2*self.s2p_ops['channel_1']['nframes'],2)
            else:
                frames = slice(0, self.s2p_ops['nframes'])

        # Get iscell
        if custom_iscell in (None, False):
            if self.n_channels>1:
                self.iscell = {}
                for chan in range(self.n_channels):
                    default_iscell_path = os.path.join(self.s2p_path[chan], plane, 'iscell.npy')
                    if os.path.exists(default_iscell_path):
                        self.iscell[f'channel_{chan}'] = np.load(default_iscell_path)
                    else:
                        print("No iscell file found, using None")
                        self.iscell = None
            else:    
                default_iscell_path = os.path.join(self.s2p_path, plane, 'iscell.npy')
                if os.path.exists(default_iscell_path):
                    self.iscell = np.load(default_iscell_path)
                else:
                    print("No iscell file found, using None")
                    self.iscell = None
        else:
            if self.n_channels>1:
                raise NotImplementedError("Custom iscell not implemented for multi-channel data")
            
            custom_iscell = os.path.normpath(custom_iscell)
            if custom_iscell.count(os.path.sep) < 1:
                custom_iscell = os.path.join(self.s2p_path, plane, custom_iscell)

            if os.path.splitext(custom_iscell)[1] == '.npy':
                self.iscell = np.load(custom_iscell)
            elif os.path.splitext(custom_iscell)[1] == '.csv':
                self.iscell = pd.read_csv(custom_iscell)
            else:
                raise ValueError("custom_iscell must be a .npy or .csv file")        
        
        
        if self.n_channels>1:
            self.s2p_stats = {}
            for chan in range(self.n_channels):
                try:
                    self.s2p_stats[f'channel_{chan}'] = np.load(os.path.join(self.s2p_path[chan], plane, 'stats.npy'), allow_pickle=True)
                except:
                    self.s2p_stats[f'channel_{chan}'] = np.load(os.path.join(self.s2p_path[chan], plane, 'stat.npy'), allow_pickle=True)
                
                if use_iscell:
                    self.s2p_stats[f'channel_{chan}'] = self.s2p_stats[f'channel_{chan}'][self.iscell[f'channel_{chan}'][:, 0] > 0]
        else:
            try:
                self.s2p_stats = np.load(os.path.join(self.s2p_path, plane, 'stats.npy'), allow_pickle=True)#[self.iscell[:,0]>0]
            except:
                self.s2p_stats = np.load(os.path.join(self.s2p_path, plane, 'stat.npy'), allow_pickle=True)#[self.iscell[:,0]>0]

            if use_iscell:
                self.s2p_stats = self.s2p_stats[self.iscell[:,0]>0]
        
        # If multi-plane, iterate through planes to:
        # 1) Get plane indices per cell
        # 2) Concatenate timeseries per cell per plane across rows 
       
                
        if self.n_channels>1:
            for chan in range(self.n_channels):
                if self.n_planes > 1:
                    plane_dirs = glob(os.path.join(self.s2p_path[chan], 'plane*'))
                    ts_per_plane = {}
                    plane_start_ind = 0

                    for i,plane_dir in enumerate(plane_dirs):
                        ts_per_plane[i] = {}

                        # 1) Get plane indices per cell

                        iscell_i = np.load(os.path.join(plane_dir,"iscell.npy"),allow_pickle=True)
                        print(f"this plane is shape {iscell_i.shape}")
                        # if we curated on combined, iscell per plane will not be updated!
                        # therefore we use a chunk of the combined iscell                
                        iscell_i = self.iscell[f'channel_{chan}']
                        plane_start_ind = plane_start_ind + iscell_i.shape[0]                    

                        n_cells = np.sum((iscell_i[:, 0] > 0)*1)
                        print(f"{n_cells} cells in plane {i}")                    
                        self.plane_per_cell = np.append(self.plane_per_cell,np.ones(n_cells,)*i)

                        # 2) For each timeseries type, load plane timeseries to concatenate
                        ts_to_pull = {}
                        for ts in which_ts:
                            ts_path = os.path.join(plane_dir, "%s.npy" % ts)
                            ts_per_plane[i].update({ts : np.empty((0,0),dtype=float)})
                            if os.path.exists(ts_path):
                                ts_to_pull[ts] = ts_path

                        for ts_name in ts_to_pull.keys():
                            load_ts = np.load(ts_to_pull[ts_name],allow_pickle=True)
                            if frames is not None:
                                load_ts = load_ts[:,frames]

                            assert load_ts.shape[1] == self.vr_data.shape[0],\
                                "%s must be the same length as vr_data" % ts_name
                            # Keep curated cells
                            if use_iscell:
                                ts_per_plane[i][ts_name] = load_ts[iscell_i[:, 0] > 0, :]
                            else:
                                pass

                    print("Concatenating timeseries across planes...")
                    for ts in ts_to_pull.keys():
                        self.timeseries[f'channel_{chan}_{ts}'] = np.concatenate([ts_per_plane[p][ts] for p in range(self.n_planes)], 
                                                                                 axis=0)
                else:
                    self.plane_per_cell = np.zeros(self.iscell[f'channel_{chan}'].shape[0],)

                    ts_to_pull = {}
                    for ts in which_ts:
                        ts_path = os.path.join(self.s2p_path[chan], plane, "%s.npy" % ts)
                        if os.path.exists(ts_path):
                            ts_to_pull[f'channel_{chan}_{ts}'] = ts_path
                            
                     #TODO: add support for multi-channel timeseries
                    if mux: 
                        self.add_timeseries_from_file(frames = frames, **ts_to_pull, mux=True)
                    else:
                        self.add_timeseries_from_file(frames = frames, **ts_to_pull)
                    for ts_name in ts_to_pull.keys():
                        if self.n_channels > 1:
                            
                            assert self.timeseries[ts_name].shape[1] == math.floor((self.vr_data.shape[0])/2),\
                                "%s must be the same length as vr_data" % ts_name
                        else:
                            assert self.timeseries[ts_name].shape[1] == self.vr_data.shape[0],\
                                "%s must be the same length as vr_data" % ts_name
                        if use_iscell:
                            self.timeseries[ts_name] = self.timeseries[ts_name][self.iscell[f'channel_{chan}'][:, 0] > 0, :]
                        else:
                            pass
        else:    
            if self.n_planes > 1:
                plane_dirs = glob(os.path.join(self.s2p_path, 'plane*'))
                ts_per_plane = {}
                plane_start_ind = 0
                
                for i,plane_dir in enumerate(plane_dirs):
                    ts_per_plane[i] = {}
                    
                    # 1) Get plane indices per cell
                    
                    iscell_i = np.load(os.path.join(plane_dir,"iscell.npy"),allow_pickle=True)
                    print(f"this plane is shape {iscell_i.shape}")
                    # if we curated on combined, iscell per plane will not be updated!
                    # therefore we use a chunk of the combined iscell                
                    iscell_i = self.iscell[plane_start_ind:plane_start_ind+iscell_i.shape[0]]
                    plane_start_ind = plane_start_ind + iscell_i.shape[0]                    
                    
                    n_cells = np.sum((iscell_i[:, 0] > 0)*1)
                    print(f"{n_cells} cells in plane {i}")                    
                    self.plane_per_cell = np.append(self.plane_per_cell,np.ones(n_cells,)*i)

                    # 2) For each timeseries type, load plane timeseries to concatenate
                    ts_to_pull = {}
                    for ts in which_ts:
                        ts_path = os.path.join(plane_dir, "%s.npy" % ts)
                        ts_per_plane[i].update({ts : np.empty((0,0),dtype=float)})
                        if os.path.exists(ts_path):
                            ts_to_pull[ts] = ts_path

                    for ts_name in ts_to_pull.keys():
                        load_ts = np.load(ts_to_pull[ts_name],allow_pickle=True)
                        if frames is not None:
                            load_ts = load_ts[:,frames]
                            
                        assert load_ts.shape[1] == self.vr_data.shape[0],\
                            "%s must be the same length as vr_data" % ts_name
                        # Keep curated cells
                        if use_iscell:
                            ts_per_plane[i][ts_name] = load_ts[iscell_i[:, 0] > 0, :]
                        else:
                            pass
                
                print("Concatenating timeseries across planes...")
                for ts in ts_to_pull.keys():
                    self.timeseries[ts] = np.concatenate([ts_per_plane[p][ts] for p in range(self.n_planes)], axis=0)

            # If single plane:
            else:
                self.plane_per_cell = np.zeros(self.iscell.shape[0],)
                
                ts_to_pull = {}
                for ts in which_ts:
                    ts_path = os.path.join(self.s2p_path, plane, "%s.npy" % ts)
                    if os.path.exists(ts_path):
                        ts_to_pull[ts] = ts_path
                self.add_timeseries_from_file(frames = frames, **ts_to_pull)
                for ts_name in ts_to_pull.keys():
                    assert self.timeseries[ts_name].shape[1] == self.vr_data.shape[0],\
                        "%s must be the same length as vr_data" % ts_name
                    if use_iscell:
                        self.timeseries[ts_name] = self.timeseries[ts_name][self.iscell[:, 0] > 0, :]
                    else:
                        pass

    def load_suite2p_data_multi_session(self, which_ts=('F', 'Fneu', 'spks', 'F_chan2', 'Fneu_chan2'), custom_iscell=None, frames=None, use_iscell=True, multi_sess=True):
            # define path is a quick fit because we haven't decide on the data dir staruture for each day, once settled, can fix this
            if self.n_planes > 1:
                print(f"Multiplane processing for {self.n_planes} planes")
                plane = "combined"
            else:
                plane = "plane0"

            if multi_sess == False:
                print(self.s2p_path)
                self.s2p_ops = np.load(os.path.join(self.s2p_path, plane, 'ops.npy'), allow_pickle=True).all()
            else:
                date_index = self.s2p_path.find(self.date) + len(self.date)
                define_path = os.path.join(self.s2p_path[:date_index],'combined','suite2p')
                print(define_path)
                self.s2p_ops = np.load(os.path.join(define_path, plane, 'ops.npy'), allow_pickle=True).all()


                ## ella hardcode hacky fix 

                data_path = ['/home/candong/oak/candong/social_interaction_data/calcium_imaging/social-0914-4/03_10_2024/Env1_to_Env2_fixreward',
                             '/home/candong/oak/candong/social_interaction_data/calcium_imaging/social-0914-4/03_10_2024/social_restrict_fam',
                             '/home/candong/oak/candong/social_interaction_data/calcium_imaging/social-0914-4/03_10_2024/social_restrict_nov',
                             '/home/candong/oak/candong/social_interaction_data/calcium_imaging/social-0914-4/03_10_2024/social_unrestrict_fam',
                             '/home/candong/oak/candong/social_interaction_data/calcium_imaging/social-0914-4/03_10_2024/social_unrestrict_nov',
                             '/home/candong/oak/candong/social_interaction_data/calcium_imaging/social-0914-4/03_10_2024/social_wheel_restrict']
                
                self.s2p_ops['data_path'] = data_path

            if len(self.s2p_ops['data_path']) >1:
                print("Multiple sessions found in suite2p ops, current is ", self.scene)
            else:
                print("Only one session found in suite2p ops, current is ", self.scene)
    

            if frames is None:
                frames = slice(0, self.s2p_ops['nframes'])

            # Get iscell
            if custom_iscell in (None, False):
                default_iscell_path = os.path.join(self.s2p_path, plane, 'iscell.npy')
                if os.path.exists(default_iscell_path):
                    self.iscell = np.load(default_iscell_path)
                else:
                    default_iscell_path = os.path.join(define_path, plane, 'iscell.npy')
                    if os.path.exists(default_iscell_path):
                        self.iscell = np.load(default_iscell_path)
                    else:
                        print("No iscell file found, using None")
                        self.iscell = None
            else:
                custom_iscell = os.path.normpath(custom_iscell)
                if custom_iscell.count(os.path.sep) < 1:
                    custom_iscell = os.path.join(self.s2p_path, plane, custom_iscell)

                if os.path.splitext(custom_iscell)[1] == '.npy':
                    self.iscell = np.load(custom_iscell)
                elif os.path.splitext(custom_iscell)[1] == '.csv':
                    self.iscell = pd.read_csv(custom_iscell)
                else:
                    raise ValueError("custom_iscell must be a .npy or .csv file")        
            
            if define_path is None:
                try:
                    self.s2p_stats = np.load(os.path.join(self.s2p_path, plane, 'stats.npy'), allow_pickle=True)#[self.iscell[:,0]>0]
                except:
                    self.s2p_stats = np.load(os.path.join(self.s2p_path, plane, 'stat.npy'), allow_pickle=True)#[self.iscell[:,0]>0]
            else:
                try:
                    self.s2p_stats = np.load(os.path.join(define_path, plane, 'stats.npy'), allow_pickle=True)#[self.iscell[:,0]>0]
                except:
                    self.s2p_stats = np.load(os.path.join(define_path, plane, 'stat.npy'), allow_pickle=True)#[self.iscell[:,0]>0]

            if use_iscell:
                self.s2p_stats = self.s2p_stats[self.iscell[:,0]>0]
            
            # If multi-plane, iterate through planes to:
            # 1) Get plane indices per cell
            # 2) Concatenate timeseries per cell per plane across rows 
            
            if self.n_planes > 1:
                if define_path is None:
                    plane_dirs = glob(os.path.join(self.s2p_path, 'plane*'))
                else:
                    plane_dirs = glob(os.path.join(define_path, 'plane*'))
                print(plane_dirs)

                ts_per_plane = {}
                plane_start_ind = 0
                
                for i,plane_dir in enumerate(plane_dirs):
                    ts_per_plane[i] = {}
                    
                    # 1) Get plane indices per cell
                    print(plane_dir)
                    iscell_i = np.load(os.path.join(plane_dir,"iscell.npy"),allow_pickle=True)
                    print(f"this plane is shape {iscell_i.shape}")
                    # if we curated on combined, iscell per plane will not be updated!
                    # therefore we use a chunk of the combined iscell                
                    iscell_i = self.iscell[plane_start_ind:plane_start_ind+iscell_i.shape[0]]
                    plane_start_ind = plane_start_ind + iscell_i.shape[0]                    
                    
                    n_cells = np.sum((iscell_i[:, 0] > 0)*1)
                    print(f"{n_cells} cells in plane {i}")                    
                    self.plane_per_cell = np.append(self.plane_per_cell,np.ones(n_cells,)*i)

                    # 2) For each timeseries type, load plane timeseries to concatenate
                    ts_to_pull = {}
                    for ts in which_ts:
                        ts_path = os.path.join(plane_dir, "%s.npy" % ts)
                        ts_per_plane[i].update({ts : np.empty((0,0),dtype=float)})
                        if os.path.exists(ts_path):
                            ts_to_pull[ts] = ts_path

                    for ts_name in ts_to_pull.keys():
                        load_ts = np.load(ts_to_pull[ts_name],allow_pickle=True)
                        print("the shape of the ts is ", load_ts.shape)
                        
                        if frames is not None:
                            if multi_sess == False:
                                load_ts = load_ts[:,frames]
                            else:
                                #print(print(self.s2p_ops['nframes_per_folder']) )
                                #print(self.s2p_ops['data_path'])
                                #print(self.scene)
                                session_id = self.find_scene_index()
                                if session_id != -1:
                                    if session_id == 0:
                                        start_frame = 0
                                        end_frame = self.s2p_ops['nframes_per_folder'][0]
                                    else:
                                        start_frame = np.sum(self.s2p_ops['nframes_per_folder'][:session_id])
                                        end_frame = start_frame + self.s2p_ops['nframes_per_folder'][session_id]
                                        frames = end_frame - start_frame+1

                                else:
                                    print("Session not found in data_path")
                                    raise ValueError("Session not found in data_path")
                                #print(start_frame, end_frame,frames)
                                load_ts = load_ts[:,start_frame:end_frame]
                                print(load_ts.shape)
                            
                        assert load_ts.shape[1] == self.vr_data.shape[0],\
                            "%s must be the same length as vr_data" % ts_name
                        # Keep curated cells
                        if use_iscell:
                            ts_per_plane[i][ts_name] = load_ts[iscell_i[:, 0] > 0, :]
                        else:
                            pass
                
                print("Concatenating timeseries across planes...")
                for ts in ts_to_pull.keys():
                    self.timeseries[ts] = np.concatenate([ts_per_plane[p][ts] for p in range(self.n_planes)], axis=0)

            # If single plane: haven't fix the multisess for signle plane, but should be very hard
            else:
                self.plane_per_cell = np.zeros(self.iscell.shape[0],)
                
                ts_to_pull = {}
                for ts in which_ts:
                    ts_path = os.path.join(self.s2p_path, plane, "%s.npy" % ts)
                    if os.path.exists(ts_path):
                        ts_to_pull[ts] = ts_path
                self.add_timeseries_from_file(frames = frames, **ts_to_pull)
                for ts_name in ts_to_pull.keys():
                    assert self.timeseries[ts_name].shape[1] == self.vr_data.shape[0],\
                        "%s must be the same length as vr_data" % ts_name
                    if use_iscell:
                        self.timeseries[ts_name] = self.timeseries[ts_name][self.iscell[:, 0] > 0, :]
                    else:
                        pass

    def find_scene_index(self): # This doesn't need to be here
        try:
            return next(i for i, path in enumerate(self.s2p_ops["data_path"]) 
                    if self.scene in path)
        except StopIteration:
            return -1      
                        
    def add_timeseries(self, frames = None, **kwargs):
        for k, v in kwargs.items():
            if self.vr_data is not None:
                if len(v.shape) < 2:
                    v = np.array(v)[np.newaxis, :]

                if frames is not None:
                    v = v[:,frames]

                if self.n_planes>1:
                    # print(v.shape, self.vr_data.shape)
                    assert v.shape[1]-self.vr_data.shape[0]<2, "multiplane data more than 1 frame different from vr_data"

                    v = v[:,:self.vr_data.shape[0]]
                    
            if self.vr_data.shape[0] - v.shape[1] == 1:
                print("adding zero row")
                zeros = np.zeros((v.shape[0], 1))
                v = np.concatenate((v, zeros), axis = 1)
                print(v.shape[1])
                
            assert v.shape[1] == self.vr_data.shape[0], \
                        "%s must be the same length as vr_data, %s %d, vr %d " % (k, k, v.shape[1], self.vr_data.shape[0])

            self.timeseries[k] = v

    def add_timeseries_mux(self, frames = None, **kwargs):
        for k, v in kwargs.items():
            if self.vr_data is not None:
                if len(v.shape) < 2:
                    v = np.array(v)[np.newaxis, :]

                if frames is not None:
                    v = v[:,frames]

                if self.n_planes>1:
                    # print(v.shape, self.vr_data.shape)
                    assert v.shape[1]-self.vr_data.shape[0]<2, "multiplane data more than 1 frame different from vr_data"

                    v = v[:,:self.vr_data.shape[0]]
                # if "channel_0" in k:
                #     vr_shape = self.chan0_vr.shape[0]
                # elif "channel_1" in k:
                #     vr_shape = self.chan1_vr.shape[0]
                # else:
                #     vr_shape = self.vr_data.shape[0]
                # check that v is same length as vr_data
                if self.n_channels > 1:
                    '''
                    ES added multi-chan functionality, do not throw error if VR 2x each chan

                    '''
                    
                    # NEED TO ADD, if V is odd then add one nan frame to end (VR can be odd but 2p will always be even) 
                    assert v.shape[1] == self.chan1_vr.shape[0], \
                        "%s must be the same length as vr_data, %s %d, vr %d " % (k, k, v.shape[1], self.chan1_vr.shape[0])
                else:
                    assert v.shape[1] == self.vr_data.shape[0], \
                        "%s must be the same length as vr_data, %s %d, vr %d " % (k, k, v.shape[1], self.vr_data.shape[0])

            self.timeseries[k] = v

    def add_timeseries_from_file(self,frames = None, mux = False, **kwargs):
        if mux:
            self.add_timeseries_mux(frames = frames, **{key: np.load(path) for key, path in kwargs.items()})
        else:
            self.add_timeseries(frames = frames, **{key: np.load(path) for key, path in kwargs.items()})

        
    def add_pos_binned_trial_matrix(self, ts_name, pos_key='t', **trial_matrix_kwargs):
        """
        add an attribute from an existing timeseries attribute
        :param ts_name:
        :return:
        """

        def _check_and_add_key(key):
            assert key in self.timeseries.keys(), "%s is not an existing timeseries" % key
            self.trial_matrices[key] = spatial_analyses.trial_matrix(self.timeseries[key].T,self.vr_data[pos_key]._values, self.trial_start_inds,
                                                                     self.teleport_inds, **trial_matrix_kwargs)

        if isinstance(ts_name, list) or isinstance(ts_name, tuple):
            for _ts in ts_name:
                _check_and_add_key(_ts)
        else:
            _check_and_add_key(ts_name)
            
    def add_trial_matrix_from_array(self,ts_name,array):
        """
        add a trial matrix that has been pre-computed outside the session class
        :param ts_name: timeseries name to use as a key for the matrix
        :array: pre-computed matrix
        :return:
        """
        self.trial_matrices[ts_name] = array
    

    def rm_timeseries(self,ts_name):
        if not isinstance(ts_name,list) or not isinstance(ts_name,tuple):
            ts_name = [ts_name]
        _ = [self.timeseries.pop(_ts,None) for _ts in ts_name]


    def rm_pos_binned_trial_matrix(self,keys):
        if not isinstance(keys,list) or not isinstance(keys,tuple):
            keys = [keys]
        _ = [self.trial_matrices.pop(_k,None) for _k in keys]

