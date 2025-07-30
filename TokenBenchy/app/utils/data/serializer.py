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
        self.seed = configuration.get('seed', 42)
        self.configuration = configuration
        self.database = TokenBenchyDatabase()

    
    #--------------------------------------------------------------------------
    def load_benchmark_results(self):            
        benchmarks, vocab_stats = self.database.load_benchmark_results()

        return benchmarks, vocab_stats
    
    
    #--------------------------------------------------------------------------
    def load_vocabularies(self):            
        vocabulary = self.database.load_vocabularies()

        return vocabulary
    
    
    #--------------------------------------------------------------------------
    def load_text_dataset(self):            
        text_dataset = self.database.load_text_dataset() 

        return text_dataset

    
    #--------------------------------------------------------------------------
    def save_text_dataset(self, text_dataset):            
        self.database.save_text_dataset(text_dataset) 

    
    #--------------------------------------------------------------------------
    def save_dataset_statistics(self, text_dataset):            
        self.database.save_dataset_statistics(text_dataset)  

    
    #--------------------------------------------------------------------------
    def save_vocabulary_tokens(self, data):            
        self.database.save_vocabulary_tokens(data)   

    
    #--------------------------------------------------------------------------
    def save_vocabulary_statistics(self, data):            
        self.database.save_vocabulary_statistics(data)     

    
    #--------------------------------------------------------------------------
    def save_benchmark_results(self, data):            
        self.database.save_benchmark_results(data)       

    
    #--------------------------------------------------------------------------
    def save_NSL_benchmark(self, data):            
        self.database.save_NSL_benchmark(data)       
    
    
