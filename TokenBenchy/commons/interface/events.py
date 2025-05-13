import io
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage, QPixmap

from TokenBenchy.commons.utils.data.downloads import DatasetDownloadManager, TokenizersDownloadManager
from TokenBenchy.commons.utils.benchmarks.core import BenchmarkTokenizers
from TokenBenchy.commons.utils.benchmarks.visualizer import VisualizeBenchmarkResults
from TokenBenchy.commons.utils.data.processing import ProcessDataset
from TokenBenchy.commons.constants import ROOT_DIR, DATA_PATH
from TokenBenchy.commons.logger import logger



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
        # send message to status bar
        window.statusBar().showMessage(message)    

    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        logger.error(f"{exc}\n\n{tb}")
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")


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

        return tokenizers     

    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message): 
        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        logger.error(f"{exc}\n\n{tb}")
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")
        


###############################################################################
class VisualizationEnvents:

    def __init__(self, configurations):
        self.configurations = configurations     
        self.visualizer = VisualizeBenchmarkResults(self.configurations)
        self.DPI = 600

    #--------------------------------------------------------------------------
    def visualize_benchmark_results(self): 
        figures = []                
        figures.append(self.visualizer.plot_vocabulary_size())
        figures.extend(self.visualizer.plot_histogram_tokens_length())
        figures.append(self.visualizer.plot_boxplot_tokens_length())
        figures.append(self.visualizer.plot_subwords_vs_words())       

        return figures  
    
    #--------------------------------------------------------------------------
    def convert_fig_to_qpixmap(self, fig):    
        buf = io.bytesIO()
        fig.savefig(buf, format="png", dpi=self.DPI)
        buf.seek(0)
        img_data = buf.read()       
        qimg = QImage.fromData(img_data)

        return QPixmap.fromImage(qimg)
    
    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message):         
        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        logger.error(f"{exc}\n\n{tb}")
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")