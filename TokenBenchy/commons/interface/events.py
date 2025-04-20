from PySide6.QtWidgets import QMessageBox

from TokenBenchy.commons.utils.data.downloads import DatasetDownloadManager, TokenizersDownloadManager
from TokenBenchy.commons.utils.evaluation.benchmarks import BenchmarkTokenizers
from TokenBenchy.commons.utils.data.processing import ProcessDataset
from TokenBenchy.commons.constants import ROOT_DIR, DATA_PATH
from TokenBenchy.commons.logger import logger


# [MAIN WINDOW]
###############################################################################
class LoadingEvents:

    def __init__(self, configurations, hf_access_token):
        self.configurations = configurations
        self.hf_access_token = hf_access_token  
        self.dataset_handler = DatasetDownloadManager(
            self.configurations, self.hf_access_token)    
        self.token_handler = TokenizersDownloadManager(
            self.configurations, self.hf_access_token)
           
    #--------------------------------------------------------------------------
    def load_and_process_dataset(self):
        dataset = self.dataset_handler.dataset_download()
        processor = ProcessDataset(self.configurations, dataset) 
        documents, clean_documents = processor.split_text_dataset()  
        logger.info(f'Total number of documents: {len(documents)}')
        logger.info(f'Number of valid documents: {len(clean_documents)}')  

        return clean_documents
    
    #--------------------------------------------------------------------------
    def load_tokenizers(self):
        tokenizers = self.token_handler.tokenizer_download()

        return tokenizers
    
    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_dataset_success(self, window, config):        
        corpus = config.get('corpus', 'NA')  
        config = config.get('config', 'NA')         
        message = f'Text dataset has been loaded: {corpus} with config {config}'        
        QMessageBox.information(
        window, 
        "Loading dataset",
        message,
        QMessageBox.Ok)

        # send message to status bar
        window.statusBar().showMessage(message)     

    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_tokenizers_success(self, window):              
        message = 'Tokenizers have been loaded'        
        QMessageBox.information(
        window, 
        "Loading tokenizers",
        message,
        QMessageBox.Ok)

        # send message to status bar
        window.statusBar().showMessage(message)  
    

    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        QMessageBox.critical(window, 'Download Failed', f"{exc}\n\n{tb}") 



    

# [MAIN WINDOW]
###############################################################################
class BenchmarkEvents:

    def __init__(self, configurations):
        self.configurations = configurations      
        self.benchmarker = BenchmarkTokenizers(configurations)          
           
    #--------------------------------------------------------------------------
    def calculate_dataset_statistics(self, documents):
        self.benchmarker.calculate_dataset_stats(documents) 

        return True

    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_analysis_success(self, window, configs):        
        corpus = configs.get('corpus', 'NA')  
        config = configs.get('config', 'NA')         
        message = f'{corpus} - {config} analysis is finished'        
        QMessageBox.information(
        window, 
        "Loading dataset",
        message,
        QMessageBox.Ok)

        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_analysis_error(self, window, err_tb):
        exc, tb = err_tb
        QMessageBox.critical(window, 'Analysis failed', f"{exc}\n\n{tb}")  

        

