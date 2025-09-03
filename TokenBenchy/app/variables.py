import os

from dotenv import load_dotenv

from TokenBenchy.app.constants import PROJECT_DIR
from TokenBenchy.app.logger import logger


# [IMPORT CUSTOM MODULES]
###############################################################################
class EnvironmentVariables:
    def __init__(self) -> None:
        self.env_path = os.path.join(PROJECT_DIR, "setup", ".env")
        if os.path.exists(self.env_path):
            load_dotenv(dotenv_path=self.env_path, override=True)
        else:
            logger.error(f".env file not found at: {self.env_path}")

    # -------------------------------------------------------------------------
    def get_environment_variables(self) -> dict[str, str]:
        return {
            "HF_ACCESS_TOKEN": os.getenv("HF_ACCESS_TOKEN", ""),
            "TF_CPP_MIN_LOG_LEVEL": os.getenv("TF_CPP_MIN_LOG_LEVEL", "1"),
            "MPLBACKEND": os.getenv("MPLBACKEND", "Agg"),
        }

    # -------------------------------------------------------------------------
    def get_HF_access_token(self) -> str:
        return os.getenv("HF_ACCESS_TOKEN", "")
