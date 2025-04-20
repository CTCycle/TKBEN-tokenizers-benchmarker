import traceback
from PySide6.QtCore import QObject, Signal, QRunnable, Slot

from TokenBenchy.commons.constants import ROOT_DIR, DATA_PATH
from TokenBenchy.commons.logger import logger


# [MAIN WINDOW]
###############################################################################
class DatasetWorkerSignals(QObject):
    finished = Signal(object)      
    error = Signal(tuple)       


# [MAIN WINDOW]
###############################################################################
class DatasetWorker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = DatasetWorkerSignals()

    #--------------------------------------------------------------------------
    @Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:        
            tb = traceback.format_exc()
            # pack both exception and traceback into one tuple
            self.signals.error.emit((e, tb))