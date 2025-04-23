from PySide6.QtWidgets import QMessageBox

from TokenBenchy.commons.utils.data.downloads import DatasetDownloadManager, TokenizersDownloadManager
from TokenBenchy.commons.utils.benchmarks.core import BenchmarkTokenizers
from TokenBenchy.commons.utils.benchmarks.visualizer import VisualizeBenchmarkResults
from TokenBenchy.commons.utils.data.processing import ProcessDataset
from TokenBenchy.commons.constants import ROOT_DIR, DATA_PATH
from TokenBenchy.commons.logger import logger


# [MAIN WINDOW]
###############################################################################
class DatasetEvents:

    def __init__(self, configurations, hf_access_token):
        self.configurations = configurations
        self.hf_access_token = hf_access_token  
        self.dataset_handler = DatasetDownloadManager(
            self.configurations, self.hf_access_token)    
        
           
    #--------------------------------------------------------------------------
    def load_and_process_dataset(self):
        dataset = self.dataset_handler.dataset_download()
        processor = ProcessDataset(self.configurations, dataset) 
        documents, clean_documents = processor.split_text_dataset()  
        logger.info(f'Total number of documents: {len(documents)}')
        logger.info(f'Number of valid documents: {len(clean_documents)}')  

        return clean_documents    
  
    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message):            
        QMessageBox.information(
        window, 
        "Task successful",
        message,
        QMessageBox.Ok)

        # send message to status bar
        window.statusBar().showMessage(message)  
    

    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        QMessageBox.critical(window, 'Dataset loading failed!', f"{exc}\n\n{tb}") 

    

# [MAIN WINDOW]
###############################################################################
class BenchmarkEvents:

    def __init__(self, configurations, hf_access_token):
        self.configurations = configurations    
        self.hf_access_token = hf_access_token  
        self.token_handler = TokenizersDownloadManager(
            self.configurations, self.hf_access_token)
        self.benchmarker = BenchmarkTokenizers(configurations)                                 
           
    #--------------------------------------------------------------------------
    def calculate_dataset_statistics(self, documents):
        self.benchmarker.calculate_dataset_stats(documents) 
        return True
    
    #--------------------------------------------------------------------------
    def execute_benchmarks(self, documents, progress_callback=None):
        tokenizers = self.token_handler.tokenizer_download()
        results = self.benchmarker.run_tokenizer_benchmarks(
           documents, tokenizers, progress_callback=progress_callback) 

        return results   
    
    #--------------------------------------------------------------------------
    def visualize_benchmark_results(self, tokenizers):
        visualizer = VisualizeBenchmarkResults(self.configurations, tokenizers)

        visualizer.get_vocabulary_report()          
        visualizer.plot_vocabulary_size()
        visualizer.plot_histogram_tokens_length()
        visualizer.plot_boxplot_tokens_length()
        visualizer.plot_subwords_vs_words()
        visualizer.boxplot_from_benchmarks_dataset() 

        return True 

    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message):                 
        QMessageBox.information(
        window, 
        "Task successful",
        message,
        QMessageBox.Ok)

        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")  

        

