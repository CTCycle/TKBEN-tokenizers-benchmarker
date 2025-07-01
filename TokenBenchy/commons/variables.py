import os
from dotenv import load_dotenv

from TokenBenchy.commons.constants import PROJECT_DIR
from TokenBenchy.commons.logger import logger

# [IMPORT CUSTOM MODULES]
###############################################################################
class EnvironmentVariables:

    def __init__(self):        
        self.env_path = os.path.join(PROJECT_DIR, 'app', '.env')        
        if os.path.exists(self.env_path):
            load_dotenv(dotenv_path=self.env_path, override=True)
            logger.info('Environment variables successfully loaded from .env file')
        else:
            logger.error(f".env file not found at: {self.env_path}")    
    
    #--------------------------------------------------------------------------
    def get_environment_variables(self):                  
        return {"HF_ACCESS_TOKEN": os.getenv("HF_ACCESS_TOKEN", None),                
                "TF_CPP_MIN_LOG_LEVEL": os.getenv("TF_CPP_MIN_LOG_LEVEL", "1")}
    
    #--------------------------------------------------------------------------
    def get_HF_access_token(self):                  
        return os.getenv("HF_ACCESS_TOKEN", None)
       
