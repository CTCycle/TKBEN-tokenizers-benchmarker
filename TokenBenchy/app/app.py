import sys

# [SETTING WARNINGS]
import warnings

from PySide6.QtWidgets import QApplication

warnings.simplefilter(action="ignore", category=Warning)

# [IMPORT CUSTOM MODULES]
from TokenBenchy.app.client.window import MainWindow, apply_style
from TokenBenchy.app.constants import UI_PATH

# [RUN MAIN]
###############################################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app = apply_style(app)
    main_window = MainWindow(UI_PATH)
    main_window.show()
    sys.exit(app.exec())
