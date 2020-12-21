import os
import pickle
import warnings
from abc import ABC

from . import preprocessing as pp


class SessionInfo:
    """Base class for any 2P session"""

    def __init__(self, **kwargs):
        """

        {'basedir': str, base directory to look for data
        'mouse': str,
        'date': dd_mm_yyyy str,
        'scene': str,
        'session': int,
        'scan_number': int,
        'scanner': ["NLW","ThorLabs","Bruker"],
        'VR_only': bool}

        :type mouse: object
        """
        # session information
        self.basedir_2P = None  # base directory to find 2P data - see self._check_minimal_keys()
        self.basedir_VR = None  # base directory to find VR Data - see self._check_minimal_keys()
        self.mouse = None  # mouse name (string)
        self.date = None  # date of session (dd_mm_
        self.scene = None
        self.session = None
        self.scan_number = None  # Neurolabware only
        self.scanner = None
        self.VR_only = None
        self.scan_file = None
        self.scanheader_file = None
        self.vr_filename = None
        self.s2p_path = None
        self.n_planes = 1

        self.__dict__.update(kwargs)
        self._check_minimal_keys()

        # check for VR data
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
                if _vr_only.lower in ('true', 1, 't', 'y', 'yes', 'totes', 'yep',
                                      'yeperoni', 'yeppers', 'roger'):
                    print("Setting VR_only to True, skipping VR alignment to 2P")
                    self.VR_only = True
                elif _vr_only.lower in ('false', 0, 'f', 'n', 'no', 'nope', 'negative',
                                        'no way', 'hell nah', 'get out of town!'):
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
        # for each field name
        for attr in dir(self):
            # if not reserved field name and not function
            if not attr.startswith('__') and not callable(getattr(self, attr)):
                print(attr, ":", getattr(self, attr))

    def _check_for_VR_data(self, **kwargs):

        # look for VR Data
        if self.vr_filename is None:
            self.vr_filename = os.path.join(self.basedir, self.mouse,
                                            self.date, "%s_%d.sqlite" % (self.scene, self.session))

        if not os.path.exists(self.vr_filename):
            warnings.warn("VR File %s does not exist!" % self.vr_filename)

    def _check_for_2P_data(self):
        # look for raw 2P data

        # set paths

        if self.scanner == "NLW":
            # find paths to sbx file and mat file
            matpath = os.path.join(self.basedir, self.mouse, self.date,
                                   self.scene, "%s_%03d_%03d.mat" % (self.scene, self.session, self.scan_number))
            if os.path.exists(matpath):
                self.scanheader_file = matpath
            else:
                warnings.warn("Could not find sbxmat file at %s" % matpath)
                self.scanheader_file = None

            sbxpath = os.path.join(self.basedir, self.mouse, self.date,
                                   self.scene, "%s_%03d_%03d.sbx" % (self.scene, self.session, self.scan_number))
            if os.path.exists(sbxpath):
                self.scan_file = sbxpath
            else:
                warnings.warn("Could not find sbx file at %s" % sbxpath)
                self.scan_file = None

        elif self.scanner == "ThorLabs":
            raise NotImplementedError

        elif self.scanner == "Bruker":
            raise NotImplementedError

    def _check_for_suite2P_data(self):
        # look for suite2p results
        raise NotImplementedError

    def _check_for_aligned_suite2p_sessions(self):
        # look for file that points to which session share ROIs
        warnings.warn("Looking for coaligned suite2p sessions is not implemented yet")
        # raise NotImplementedError


class Session(SessionInfo, ABC):

    def __init__(self, load_pickled_sess=False, **kwargs):
        super(Session, self).__init__(**kwargs)

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

        # check for pickled instance of Session class
        if load_pickled_sess:
            self.load_pickled_session()

    def load_pickled_session(self, pklfile=None):
        if pklfile is None:
            pklfile = self._check_for_pickled_session()

        with open(pklfile, 'rb') as f:
            pklsess = pickle.load(f)

            for attr in dir(pklsess):
                if not attr.startswith('__') and not callable(getattr(pklsess, attr)):
                    setattr(self, attr, getattr(pklsess, attr))

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

        if self.vr_data is not None or overwrite:
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

    def load_suite2p_data(self, nchan=1, F=True, Fneu=True, S=True):
        '''

        :param
        :return:
        '''

        pass

        # setting suite2p path to be default

    def save(self, output_basedir):
        # save pickled instance of class
        pkldir = os.path.join(output_basedir, self.mouse,
                              self.date)
        pklfile = os.path.join(pkldir, "%s_%d.pkl" % (self.scene, self.session))
        os.makedirs(pkldir, exist_ok=True)
        with open(pklfile, 'wb') as f:
            pickle.dump(self, f)
