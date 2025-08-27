
import pandas as pd

from TokenBenchy.app.utils.data.database import database


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:

    def __init__(self):        
        pass      
                  
    #--------------------------------------------------------------------------    
    def load_benchmark_results(self) -> pd.DataFrame:                
        return database.load_from_database('BENCHMARK_RESULTS')           
    
    #--------------------------------------------------------------------------
    def load_vocabularies(self) -> pd.DataFrame:              
        return database.load_from_database('VOCABULARY')
            
    #--------------------------------------------------------------------------
    def load_text_dataset(self):            
        return database.load_from_database('TEXT_DATASET')
    
    #--------------------------------------------------------------------------
    def save_text_dataset(self, dataset : pd.DataFrame):            
        database.save_into_database(dataset, 'TEXT_DATASET') 
    
    #--------------------------------------------------------------------------
    def save_dataset_statistics(self, dataset : pd.DataFrame):          
        database.upsert_into_database(dataset, 'TEXT_DATASET_STATISTICS')  
    
    #--------------------------------------------------------------------------
    def save_vocabulary_tokens(self, dataset : pd.DataFrame):            
        database.upsert_into_database(dataset, 'VOCABULARY') 
    
    #--------------------------------------------------------------------------
    def save_vocabulary_statistics(self, dataset : pd.DataFrame):        
        database.save_into_database(dataset, 'VOCABULARY_STATISTICS') 
    
    #--------------------------------------------------------------------------
    def save_benchmark_results(self, dataset : pd.DataFrame):             
        database.save_into_database(dataset, 'BENCHMARK_RESULTS')

    #--------------------------------------------------------------------------
    def save_NSL_benchmark(self, dataset : pd.DataFrame):            
        database.save_into_database(dataset, 'NSL_RESULTS') 
    
    
    
