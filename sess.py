import os
import warnings

from . import preprocessing as pp


class Session:
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

        self.__dict__.update(kwargs)
        self._check_minimal_keys()

        # check for pickled instance of ImagingSession class
        self._check_for_pickled_session()

        # check for VR data
        self._check_for_VR_data()

        if not self.VR_only:
            # check for suite 2P data
            self._check_for_suite2P_data()

            # check for other sessions that 2P data is aligned to
            self._check_for_aligned_suite2p_sessions()

            # check for aligned VR data
            self._check_for_aligned_VRData()

        # print available fields
        self.print_session_info()

    def _check_minimal_keys(self):
        """
        checks to make sure initialization of class has proper attributes to prevent other functions from failing
        :return:
        """

        print(
            "Expected directory tree for VR Data base_dir\\VR_Data\\mouse\\date_folder\\scene\\scene_sessionnumber.sql")
        print(
            "Expected directory tree for 2P Data base_dir\\2P_Data\\mouse\\date_folder\\scene\\scene_sessionnumber_scannumber.sbx\mat")
        print(
            "Thorlabs B scope and Bruker compatibility to be added"
        )
        if not hasattr(self, 'basedir'):
            print("What is the base directory for your VR and 2P data?")
            self.basedir = input()

        if not hasattr(self, 'mouse'):
            print("Mouse ID?")
            self.mouse = input()

        if not hasattr(self, 'date'):
            print("Date of experiment (dd_mm_yyy)?")
            self.date = input()

        if not hasattr(self, 'scene'):
            print("Name of Unity scene?")
            self.scene = input()

        if not hasattr(self, 'session'):
            print("Session number")
            _session_number = input()
            self.session = int(_session_number)

        if not hasattr(self, 'VR_only'):


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
            if not hasattr(self,'scanner'):
                print("Which microscope? [NLW,ThorLabs,Bruker]")
                scanner = input()
                self.scanner = scanner

            if self.scanner == "NLW":
                print("Scan Number?")
                scannumber = input()
                self.scan_number = int(scannumber)

    def print_session_info(self):
        # for each field name

        # if field name is not a function

        # print value

        raise NotImplementedError

    def _check_for_pickled_session(self):
        # look for pickled session
        raise NotImplementedError

    def _check_for_VR_data(self, **kwargs):

        # look for VR Data
        raise NotImplementedError

    def _check_for_2P_data(self, **kwargs):
        # look for raw 2P data

        # set paths

        # if scanner=="NeuroLabware"
        if self.scanner == "NLW":
            # find paths to sbx file and mat file
            matpath = os.path.join(self.basedir, self.mouse, self.date,
                                   self.scene, "%s_%03d_%03d.mat" % (self.scene, self.session, self.scan_number))
            if os.path.exists(matpath):
                self.sbxmat_file = matpath
            else:
                warnings.warn("Could not find sbxmat file at %s" % matpath)
                self.sbxmat_file = None

            sbxpath = os.path.join(self.basedir, self.mouse, self.date,
                                   self.scene, "%s_%03d_%03d.sbx" % (self.scene, self.session, self.scan_number))
            if os.path.exists(sbxpath):
                self.sbx_file = sbxpath
            else:
                warnings.warn("Could not find sbx file at %s" % sbxpath)
                self.sbx_file = None

        elif self.scanner == "ThorLabs":
            raise NotImplementedError

        elif self.scanner == "Bruker":
            raise NotImplementedError

    def _check_for_suite2P_data(self):
        # look for suite2p results
        raise NotImplementedError

    def _check_for_aligned_VRData(self):
        # check if aligned VRData field name exists
        raise NotImplementedError

    def _check_for_aligned_suite2p_sessions(self):
        # look for file that points to which session share ROIs

        raise NotImplementedError

    def save(self):
        # save pickled instance of class
        raise NotImplementedError

    def run_suite2P(self, ops={}):
        # run suite2p

        raise NotImplementedError

    def align_VR_to_2P(self):

        # load sqlite file as pandas array
        df = pp.load_sqlite(self.vr_filename)
        if not self.VR_only:
            # feed pandas array and scene name to alignment function
            self.vr_data = pp.align_VR_2P(df, self.sbxmat)
        else:
            self.vr_data = df
