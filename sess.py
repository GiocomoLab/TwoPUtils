import os
import warnings
from abc import ABC
from glob import glob

import dill
import numpy as np

from . import preprocessing as pp
from . import utils


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



        """
        # session information
        # initialize attributes
        self.basedir_2P = None  # base directory to find 2P data - see self._check_minimal_keys()
        self.basedir_VR = None  # base directory to find VR Data - see self._check_minimal_keys()
        self.mouse = None  # mouse name (string)
        self.date = None  # date of session (dd_mm_yyyy)
        self.scene = None  # name of unity scene
        self.session = None  # int, session number
        self.scan_number = None  # Neurolabware only
        self.scanner = None  # ['NLW','ThorLabs','Bruker']
        self.VR_only = None  # bool, whether the session is only vr data
        self.scan_file = None  # string, path to neural data
        self.scanheader_file = None  # string, path to header file for scan
        self.vr_filename = None  # string, path to vr sqlite file
        self.s2p_path = None  # string, suite2p path
        self.n_planes = 1  # int, number of imaging planes
        self.prompt_for_keys = True  # bool, whether or not to run through prompts for minimal keys

        self.__dict__.update(kwargs)  # update keys based on inputs
        # if want to receive prompts for minimal keys
        if self.prompt_for_keys:
            self._check_minimal_keys()
        else:
            # check that provided keys are the right type
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
            self._check_for_aligned_suite2p_sessions()

        # print available fields
        self.print_session_info()

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
            "Thorlabs B scope and Bruker compatibility to be added"
        )
        if self.basedir_VR is None:
            print("What is the base directory for your VR data?")
            self.basedir = input()

        if self.basedir_2P is None:
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
            self.vr_filename = os.path.join(self.basedir, self.mouse,
                                            self.date, "%s_%d.sqlite" % (self.scene, self.session))

        if not os.path.exists(self.vr_filename):
            warnings.warn("VR File %s does not exist!" % self.vr_filename)

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
                warnings.warn("Could not find sbxmat file at %s" % self.scanheader_file)
                self.scanheader_file = None

            if self.scan_file is None:
                self.scan_file = os.path.join(self.basedir, self.mouse, self.date,
                                              self.scene,
                                              "%s_%03d_%03d.sbx" % (self.scene, self.session, self.scan_number))
            if not os.path.exists(self.scan_file):
                warnings.warn("Could not find sbx file at %s" % self.scan_file)
                self.scan_file = None

        elif self.scanner == "ThorLabs":
            raise NotImplementedError

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
            self.s2p_path = os.path.join(base, 'suite2p')
            if not os.path.exists(self.s2p_path):
                warnings.warn("Could not find suite2p path at %s" % self.s2p_path)
            else:
                self.n_planes = len(glob(os.path.join(self.s2p_path, 'plane*')))

    def _check_for_aligned_suite2p_sessions(self):
        # look for file that points to which session share ROIs
        warnings.warn("Looking for coaligned suite2p sessions is not implemented yet")
        # raise NotImplementedError


class Session(SessionInfo, ABC):
    """
    Extension of SessionInfo class that contains behavioral data and neural data
    """

    def __init__(self, load_pickled_sess=False, **kwargs):
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


        # vr data
        self.vr_data = None

        # neural data
        self.scan_info = None
        self.n_planes = None
        # green pmt
        self.F_pmt0 = None
        self.Fneu_pmt0 = None
        self.DFF_pmt0 = None
        self.S_pmt0 = None
        # red pmt
        self.F_pmt1 = None
        self.Fneu_pmt1 = None
        self.DFF_pmt1 = None
        self.S_pmt1 = None

        #self.__dict__.update(kwargs)  # update keys based on inputs - might not need this line/called through super inheritance
        super(Session, self).__init__(**kwargs)

        # check for pickled instance of Session class
        if load_pickled_sess:
            self.load_pickled_session()

    def load_pickled_session(self, pklfile=None):
        if pklfile is None:
            pklfile = self._check_for_pickled_session()

        with open(pklfile, 'rb') as f:
            pkl_sess = dill.load(f)

            for attr in dir(pkl_sess):
                if not attr.startswith('__') and not callable(getattr(pkl_sess, attr)):
                    setattr(self, attr, getattr(pkl_sess, attr))

    def _check_for_pickled_session(self):
        if hasattr(self, 'pickle_dir'):
            pklfile = os.path.join(self.pickle_dir, self.mouse,
                                   self.date,
                                   "%s_%d.pkl" % (self.scene, self.session))

        else:
            print("SessionInfo class instance has no attribute 'pickle_dir'. \n",
                  "Checking current working directory for pickled session")
            pklfile = os.path.join(os.getcwd(), self.mouse,
                                   self.date,
                                   "%s_%d.pkl" % (self.scene, self.session))

        assert os.path.exists(pklfile), "%s does not exist" % pklfile
        return pklfile

    def align_VR_to_2P(self, overwrite=False):

        if self.vr_data is None or overwrite:
            # load sqlite file as pandas array
            df = pp.load_sqlite(self.vr_filename)
            if not self.VR_only:
                # feed pandas array and scene name to alignment function
                if self.scanner == "NLW":
                    self.vr_data = pp.vr_align_to_2P(df, self.scan_info)
                else:
                    warnings.warn("VR alignment only implemented for Neurolabware at the moment")
                    raise NotImplementedError
            else:
                self.vr_data = df
        else:
            print("VR data already set or overwrite=False")

    def load_suite2p_data(self, n_chan=1, F=True, Fneu=True, S=True, custom_iscell=None):

        pass

        # setting suite2p path to be default

    def add_timeseries(self, **kwargs):
        for k, v in kwargs.items():
            if self.vr_data is not None:
                # check that v is same length as vr_data
                assert v.shape[0] == self.vr_data.shape[0], "%s must be the same length as vr_data" % k

            setattr(self, k, v)

    def add_timeseries_from_file(self, **kwargs):

        self.add_timeseries({attr: np.load(path) for attr, path in kwargs.items()})

    def add_pos_binned_trial_matrix(self, attr_name):
        """
        add an attribute from an existing timeseries attribute
        :param attr_name:
        :return:
        """

        assert hasattr(self, attr_name), "%s is not an attribute" % attr_name
        assert getattr(self, attr_name).is_timeseries, "%s must be a timeseries" % attr_name
        trial_matrix_attr_name = attr_name + "_trial_matrix"

        setattr(self, trial_matrix_attr_name, utils.trial_matrix())  ### add arguments to utils.trial matrix
