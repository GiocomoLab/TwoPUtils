
class Session:
    """Base class for any 2P session"""

    def __init__(self, basedir: str, mouse: str, date: str, scene: str,session_number: str,scan_number: int,
                 scanner="NLW",VR_only=False):
        """

        :type mouse: object
        """
        self.basedir = str
        self.mouse = mouse
        self.date = date
        self.session = session_number
        self.scanner = scanner
        self.scene = scene
        self.scan_number = scan_number

        # check for pickled instance of ImagingSession class
        self._check_for_pickled_session()

        # check for VR data
        self._check_for_VR_data()

        if not VR_only:
            # check for suite 2P data
            self._check_for_suite2P_data()

            # check for other sessions that 2P data is aligned to
            self._check_for_aligned_suite2p_sessions()

            # check for aligned VR data
            self._check_for_aligned_VRData()

        # print available fields
        self.print_session_info()

    def print_session_info(self):
        # for each field name

        # if field name is not a function

        # print value

        raise NotImplementedError

    def _check_for_pickled_session(self):
        # look for pickled session
        raise NotImplementedError

    def _check_for_VR_data(self,**kwargs):

        # look for VR Data
        raise NotImplementedError

    def _check_for_2P_data(self,**kwargs):
        # look for raw 2P data

        # set paths


        # if scanner=="NeuroLabware"
        if self.scanner=="NLW":
            # find paths to sbx file and mat file
            matpath = os.path.join(self.basedir,self.mouse,self.date,
                                   self.scene,"%s_%03d_%03d.mat" %(self.scene,self.session,self.scan_number))
            if os.path.exists(matpath):
                self.sbxmat_file = matpath
            else:
                warnings.warn("Could not find sbxmat file at %s" % matpath)
                self.sbxmat_file = None


            self.sbxmat_file =

        elif self.scanner=="ThorLabs":
            raise NotImplementedError

        elif self.scanner=="Bruker":
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
        df = pd.
        if self.scan_number is not None:
            # feed pandas array and scene name to alignment function
            self.vr_data = preprocessing.align_VR_2P(vr_dataframe,self.scene)
        else:
            self.vr_data =

