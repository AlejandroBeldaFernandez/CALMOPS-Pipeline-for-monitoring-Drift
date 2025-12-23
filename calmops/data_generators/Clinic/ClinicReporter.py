from ..Synthetic.SyntheticReporter import SyntheticReporter


class ClinicReporter(SyntheticReporter):
    """
    Reporter specialized for Clinical data blocks.
    Inherits all functionality from SyntheticReporter but allows for future specializations.
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        self.logger.name = "ClinicReporter"
