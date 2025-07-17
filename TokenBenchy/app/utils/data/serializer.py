

from TokenBenchy.app.utils.data.database import TokenBenchyDatabase

from TokenBenchy.app.constants import TOKENIZER_PATH
from TokenBenchy.app.logger import logger


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:

    def __init__(self, configuration):        
        self.img_shape = (128, 128, 3)
        self.num_channels = self.img_shape[-1] 
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}        
        self.seed = configuration.get('general_seed', 42)
        self.configuration = configuration
        self.database = TokenBenchyDatabase()

    # get all valid images within a specified directory and return a list of paths
    #--------------------------------------------------------------------------
    def load_benchmark_results(self):            
        benchmarks, vocab_stats = self.database.load_benchmark_results()

        return benchmarks, vocab_stats
    
    # get all valid images within a specified directory and return a list of paths
    #--------------------------------------------------------------------------
    def load_vocabulary_tokens(self, table_name):            
        vocabulary = self.database.load_vocabulary_tokens(table_name)

        return vocabulary
    
    # get all valid images within a specified directory and return a list of paths
    #--------------------------------------------------------------------------
    def load_text_dataset(self):            
        text_dataset = self.database.load_text_dataset() 

        return text_dataset

    # get all valid images within a specified directory and return a list of paths
    #--------------------------------------------------------------------------
    def save_text_dataset(self, text_dataset):            
        self.database.save_text_dataset(text_dataset) 

    # get all valid images within a specified directory and return a list of paths
    #--------------------------------------------------------------------------
    def save_dataset_statistics(self, text_dataset):            
        self.database.save_dataset_statistics(text_dataset)  

    # get all valid images within a specified directory and return a list of paths
    #--------------------------------------------------------------------------
    def save_vocabulary_tokens(self, data, table_name=None):            
        self.database.save_vocabulary_tokens(data, table_name=table_name)   

    # get all valid images within a specified directory and return a list of paths
    #--------------------------------------------------------------------------
    def save_vocabulary_results(self, data):            
        self.database.save_vocabulary_results(data)     

    # get all valid images within a specified directory and return a list of paths
    #--------------------------------------------------------------------------
    def save_benchmark_results(self, data, table_name=None):            
        self.database.save_benchmark_results(data, table_name=table_name)            
    
    
