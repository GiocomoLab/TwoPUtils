class ImagingSession:
    """Base class for any 2P session"""

    def __init__(self, mouse, date, session_number, scanner="NLW"):
        """

        :type mouse: object
        """
        self.mouse = mouse
        self.date = date
        self.session = session_number
        self.scanner = scanner

        # check for pickled instance of ImagingSession class
        self._check_for_pickled_session()

        # check for VR data
        self._check_for_VR_data()

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

    def _check_for_VR_data(self):
        # look for VR Data
        raise NotImplementedError

    def _check_for_2P_data(self):
        # look for raw 2P data

        # if scanner=="NeuroLabware"

        # elif scanner=="ThorLabs"

        # elif scanner=="Bruker"

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
