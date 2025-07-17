import os
import json

from keras.utils import plot_model
from keras.models import load_model
from datetime import datetime

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
    def save_dataset_statistics(self, dataset_stats):            
        self.database.save_dataset_statistics(dataset_stats)   

    # get all valid images within a specified directory and return a list of paths
    #--------------------------------------------------------------------------
    def save_vocabulary_tokens(self, dataset_stats, table_name=None):            
        self.database.save_vocabulary_tokens(dataset_stats, table_name=table_name)   

    # get all valid images within a specified directory and return a list of paths
    #--------------------------------------------------------------------------
    def save_vocabulary_results(self, dataset_stats):            
        self.database.save_vocabulary_results(dataset_stats)     

    # get all valid images within a specified directory and return a list of paths
    #--------------------------------------------------------------------------
    def save_benchmark_results(self, dataset_stats, table_name=None):            
        self.database.save_benchmark_results(dataset_stats, table_name=table_name)            
    
    
