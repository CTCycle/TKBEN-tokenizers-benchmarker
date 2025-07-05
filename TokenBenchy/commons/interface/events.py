from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_agg import FigureCanvasAgg

from TokenBenchy.commons.utils.downloads import DatasetDownloadManager, TokenizersDownloadManager
from TokenBenchy.commons.utils.benchmarks import BenchmarkTokenizers, VisualizeBenchmarkResults
from TokenBenchy.commons.utils.processing import ProcessDataset
from TokenBenchy.commons.interface.workers import check_thread_status
from TokenBenchy.commons.logger import logger



###############################################################################
class DatasetEvents:

    def __init__(self, database, configuration, hf_access_token):
        self.database = database
        self.configuration = configuration
        self.hf_access_token = hf_access_token           
           
    #--------------------------------------------------------------------------
    def load_and_process_dataset(self):
        downloader = DatasetDownloadManager(self.configuration, self.hf_access_token)      
        dataset = downloader.dataset_download()
        
        processor = ProcessDataset(self.configuration, dataset) 
        documents, clean_documents = processor.split_text_dataset()  
        logger.info(f'Total number of documents: {len(documents)}')
        logger.info(f'Number of valid documents: {len(clean_documents)}')  

        return clean_documents    
  
   



###############################################################################
class BenchmarkEvents:

    def __init__(self, database, configuration, hf_access_token): 
        self.database = database
        self.configuration = configuration    
        self.hf_access_token = hf_access_token                                   
           
    #--------------------------------------------------------------------------
    def run_dataset_evaluation_pipeline(self, documents, progress_callback=None, worker=None):        
        benchmarker = BenchmarkTokenizers(self.database, self.configuration)
        benchmarker.calculate_dataset_statistics(
            documents, progress_callback=progress_callback, worker=worker)         
    
    #--------------------------------------------------------------------------
    def get_tokenizer_identifiers(self, limit=1000, worker=None):
        downloader = TokenizersDownloadManager(self.configuration, self.hf_access_token)
        identifiers = downloader.get_tokenizer_identifiers(limit=limit, worker=worker)

        return identifiers
    
    #--------------------------------------------------------------------------
    def execute_benchmarks(self, documents, progress_callback=None, worker=None):
        benchmarker = BenchmarkTokenizers(self.database, self.configuration)
        downloader = TokenizersDownloadManager(self.configuration, self.hf_access_token)
        tokenizers = downloader.tokenizer_download(worker=worker)
        results = benchmarker.run_tokenizer_benchmarks(
           documents, tokenizers, progress_callback=progress_callback, worker=worker) 

        return tokenizers     


###############################################################################
class VisualizationEnvents:

    def __init__(self, database, configuration):        
        self.database = database
        self.configuration = configuration 
        self.DPI = 600 

    #--------------------------------------------------------------------------
    def visualize_benchmark_results(self, worker=None):
        visualizer = VisualizeBenchmarkResults(self.database, self.configuration)

        figures = []      
        # 1. generate plot of different vocabulary sizes  
        check_thread_status(worker)
        figures.append(visualizer.plot_vocabulary_size())
        # 2. generate plot of token length distribution
        check_thread_status(worker)
        figures.extend(visualizer.plot_tokens_length_distribution())
        # 2. generate plot of words versus subwords
        check_thread_status(worker)      
        figures.append(visualizer.plot_subwords_vs_words())       

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
    
