from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg

from TokenBenchy.commons.utils.data.downloads import DatasetDownloadManager, TokenizersDownloadManager
from TokenBenchy.commons.utils.benchmarks.core import BenchmarkTokenizers
from TokenBenchy.commons.utils.benchmarks.visualizer import VisualizeBenchmarkResults
from TokenBenchy.commons.utils.data.processing import ProcessDataset
from TokenBenchy.commons.constants import ROOT_DIR, DATA_PATH
from TokenBenchy.commons.logger import logger



###############################################################################
class DatasetEvents:

    def __init__(self, configuration, hf_access_token):
        self.configuration = configuration
        self.hf_access_token = hf_access_token  
        self.dataset_handler = DatasetDownloadManager(
            self.configuration, self.hf_access_token)          
           
    #--------------------------------------------------------------------------
    def load_and_process_dataset(self):
        dataset = self.dataset_handler.dataset_download()
        processor = ProcessDataset(self.configuration, dataset) 
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

    def __init__(self, configuration, hf_access_token):
        self.configuration = configuration    
        self.hf_access_token = hf_access_token  
        self.token_handler = TokenizersDownloadManager(
            self.configuration, self.hf_access_token)
        self.benchmarker = BenchmarkTokenizers(configuration)                                 
           
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

    def __init__(self, configuration):
        self.configuration = configuration     
        self.visualizer = VisualizeBenchmarkResults(self.configuration)
        self.DPI = 600

    #--------------------------------------------------------------------------
    def visualize_benchmark_results(self): 
        figures = []                
        figures.append(self.visualizer.plot_vocabulary_size())
        figures.extend(self.visualizer.plot_tokens_length_distribution())        
        figures.append(self.visualizer.plot_subwords_vs_words())       

        return figures  
    
    #--------------------------------------------------------------------------
    def convert_fig_to_qpixmap(self, fig):    
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        # get the size in pixels and initialize raw RGBA buffer
        width, height = canvas.get_width_height()        
        buf = canvas.buffer_rgba()
        # construct a QImage pointing at that memory (no PNG decoding)
        qimg = QImage(buf, width, height, QImage.Format_RGBA8888)

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