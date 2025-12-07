import sys

import warnings

from PySide6.QtWidgets import QApplication

warnings.simplefilter(action="ignore", category=Warning)

# [IMPORT CUSTOM MODULES]
from TKBEN_desktop.app.utils.variables import env_variables
from TKBEN_desktop.app.client.window import MainWindow, apply_style
from TKBEN_desktop.app.utils.constants import UI_PATH

# [RUN MAIN]
###############################################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app = apply_style(app)
    main_window = MainWindow(UI_PATH, env_variables)                            
    main_window.show()
    sys.exit(app.exec())
