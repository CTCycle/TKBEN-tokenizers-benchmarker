from functools import partial
from PySide6.QtWidgets import (QPushButton, QCheckBox, QPlainTextEdit, QSpinBox, 
                               QMessageBox, QComboBox, QTextEdit)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QThreadPool

from TokenBenchy.commons.variables import EnvironmentVariables
from TokenBenchy.commons.interface.events import LoadingEvents, BenchmarkEvents
from TokenBenchy.commons.interface.configurations import Configurations
from TokenBenchy.commons.interface.workers import DatasetWorker, BenchmarkWorker
from TokenBenchy.commons.constants import UI_PATH
from TokenBenchy.commons.logger import logger
        

# [MAIN WINDOW]
###############################################################################
class MainWindow:
    
    def __init__(self, ui_file_path: str): 
        super().__init__()           
        loader = QUiLoader()
        ui_file = QFile(ui_file_path)
        ui_file.open(QIODevice.ReadOnly)
        self.main_win = loader.load(ui_file)
        ui_file.close()  
        self.main_win.showMaximized()

        self.text_dataset = None
        self.tokenizers = []

        # initial settings
        self.config_manager = Configurations()
        self.configurations = self.config_manager.get_configurations()
    
        self.threadpool = QThreadPool.globalInstance()
        self._dataset_worker = None
        self._tokenizer_worker = None
        self._benchmark_worker = None

        # get Hugging Face access token from environmental variables
        EV = EnvironmentVariables()
        self.hf_access_token = EV.get_HF_access_token()

        # --- Create persistent handlers ---
        # These objects will live as long as the MainWindow instance lives 
        self.loading_handler = LoadingEvents(self.configurations, self.hf_access_token) 
        self.benchmark_handler = BenchmarkEvents(self.configurations)           
        
        # --- modular checkbox setup ---
        self._setup_configurations()

        # --- Connect signals to slots ---
        self._connect_signals()           

    #--------------------------------------------------------------------------
    def _connect_button(self, button_name: str, slot):        
        button = self.main_win.findChild(QPushButton, button_name)
        button.clicked.connect(slot) 

    #--------------------------------------------------------------------------
    def _connect_combo_box(self, combo_name: str, slot):        
        combo = self.main_win.findChild(QComboBox, combo_name)
        combo.currentTextChanged.connect(slot)

    #--------------------------------------------------------------------------
    def _setup_configurations(self):              
        self.check_custom_data = self.main_win.findChild(QCheckBox, "includeCustomDataset")
        self.check_custom_token = self.main_win.findChild(QCheckBox, "includeCustomToken")
        self.check_include_NSL = self.main_win.findChild(QCheckBox, "includeNSL")
        self.check_reduce = self.main_win.findChild(QCheckBox, "reduceSize")
        # set the default value of the wait time box to the current wait time
        self.set_num_docs = self.main_win.findChild(QSpinBox, "numDocs")
        self.set_num_docs.setValue(self.configurations.get('num_docs', 0))       

        # connect their toggled signals to our updater
        self.check_custom_data.toggled.connect(self._update_settings)
        self.check_custom_token.toggled.connect(self._update_settings) 
        self.check_include_NSL.toggled.connect(self._update_settings)
        self.check_reduce.toggled.connect(self._update_settings) 
        self.set_num_docs.valueChanged.connect(self._update_settings)         
      
    #--------------------------------------------------------------------------
    def _connect_signals(self):        
        self._connect_combo_box("selectTokenizers", self.on_tokenizer_selection_from_combo)
        self._connect_button("loadDataset", self.load_and_process_dataset)
        self._connect_button("analyzeDataset", self.run_dataset_analysis)
        self._connect_button("loadTokenizers", self.load_tokenizers)
        

    # --- Slots ---
    # It's good practice to define methods that act as slots within the class
    # that manages the UI elements. These slots can then call methods on the
    # handler objects. Using @Slot decorator is optional but good practice
    #--------------------------------------------------------------------------
    @Slot()
    def _update_settings(self):        
        self.config_manager.update_value('include_custom_dataset', self.check_custom_data.isChecked())
        self.config_manager.update_value('include_custom_tokenizer', self.check_custom_token.isChecked())
        self.config_manager.update_value('include_NSL', self.check_include_NSL.isChecked())
        self.config_manager.update_value('reduce_output_size', self.check_reduce.isChecked())
        self.config_manager.update_value('num_documents', self.set_num_docs.value())      

    #--------------------------------------------------------------------------
    @Slot()
    def load_and_process_dataset(self):  
        set_data_corpus = self.main_win.findChild(QTextEdit, "datasetCorpus")
        set_data_config = self.main_win.findChild(QTextEdit, "datasetConfig")
        # extract text from input text boxes and strip leading/trailing whitespace and newlines
        corpus_text = set_data_corpus.toPlainText()
        config_text = set_data_config.toPlainText()        
        corpus_text = corpus_text.replace('\n', ' ').strip()
        config_text = config_text.replace('\n', ' ').strip()

        # update configurations with the text from the input boxes and reinitialize
        # the loading handler with the new configurations
        dataset_config = {'corpus': corpus_text, 'config': config_text} 
        self.config_manager.update_value('DATASET', dataset_config)
        self.configurations = self.config_manager.get_configurations() 
        self.loading_handler = LoadingEvents(self.configurations, self.hf_access_token) 
        
        # send message to status bar
        self.main_win.statusBar().showMessage(f"Downloading dataset {corpus_text} (configuration: {config_text})")

        # initialize worker for asynchronous loading of the dataset
        # functions that are passed to the worker will be executed in a separate thread
        self._dataset_worker = DatasetWorker(
            self.loading_handler.load_and_process_dataset)
        worker = self._dataset_worker       
        worker.signals.finished.connect(self.on_dataset_loaded)
        worker.signals.error.connect(self.on_dataset_error)
        self.threadpool.start(worker)      

    #--------------------------------------------------------------------------
    @Slot(object)
    def on_dataset_loaded(self, datasets):             
        self.text_dataset = datasets
        cfg = self.config_manager.get_configurations().get('DATASET', {})
        self.loading_handler.handle_dataset_success(self.main_win, cfg)

    #--------------------------------------------------------------------------
    @Slot(tuple)
    def on_dataset_error(self, err_tb):
        self.loading_handler.handle_error(self.main_win, err_tb)  

    #--------------------------------------------------------------------------
    @Slot()
    def run_dataset_analysis(self): 
        # make sure you've actually loaded a dataset
        if self.text_dataset is None:
            QMessageBox.warning(self.main_win,
                                "Missing dataset",
                                "Please load a dataset before running analysis!")
            return None
            
        self.configurations = self.config_manager.get_configurations() 
        self.benchmark_handler = BenchmarkEvents(self.configurations)

        # send message to status bar
        self.main_win.statusBar().showMessage("Computing statistics for the selected dataset")

        analysis_fn = partial(
            self.benchmark_handler.calculate_dataset_statistics,
            self.text_dataset)

        # initialize worker for asynchronous loading of the dataset
        self._dataset_worker = DatasetWorker(analysis_fn)
        worker = self._dataset_worker      
        worker.signals.finished.connect(self.on_analysis_success)
        worker.signals.error.connect(self.on_analysis_error)
        self.threadpool.start(worker)   

    #--------------------------------------------------------------------------
    @Slot(object)
    def on_analysis_success(self, result):           
        cfg = self.config_manager.get_configurations().get('DATASET', {})
        self.benchmark_handler.handle_analysis_success(self.main_win, cfg)

    #--------------------------------------------------------------------------
    @Slot(tuple)
    def on_analysis_error(self, err_tb):
        self.benchmark_handler.handle_analysis_error(self.main_win, err_tb) 

    #--------------------------------------------------------------------------
    @Slot(str)
    def on_tokenizer_selection_from_combo(self, text: str):
        tokenizers = self.main_win.findChild(QPlainTextEdit, "tokenizersToBenchmark")  
        existing = set(tokenizers.toPlainText().splitlines())
        if text not in existing:
            tokenizers.appendPlainText(text) 

    #--------------------------------------------------------------------------
    @Slot()
    def load_tokenizers(self):  
        tokenizers = self.main_win.findChild(QPlainTextEdit, "tokenizersToBenchmark") 
        tokenizers_name = tokenizers.toPlainText().splitlines()
        tokenizers_name = [x.replace('\n', ' ').strip() for x in tokenizers_name]

        self.config_manager.update_value('TOKENIZERS', tokenizers_name)
        self.configurations = self.config_manager.get_configurations() 
        self.loading_handler = LoadingEvents(self.configurations, self.hf_access_token) 

        # send message to status bar
        self.main_win.statusBar().showMessage("Downloading selected tokenizers")

        # initialize worker for asynchronous loading of the dataset
        # functions that are passed to the worker will be executed in a separate thread
        self._dataset_worker = BenchmarkWorker(
            self.loading_handler.load_tokenizers)
        worker = self._dataset_worker       
        worker.signals.finished.connect(self.on_tokenizers_loaded)
        worker.signals.error.connect(self.on_tokenizers_error)
        self.threadpool.start(worker) 

    #--------------------------------------------------------------------------
    @Slot(object)
    def on_tokenizers_loaded(self, tokenizers):             
        self.tokenizers = tokenizers
        cfg = self.config_manager.get_configurations().get('TOKENIZERS', [])
        self.loading_handler.handle_tokenizers_success(self.main_win)

    #--------------------------------------------------------------------------
    @Slot(tuple)
    def on_tokenizers_error(self, err_tb):
        self.loading_handler.handle_error(self.main_win, err_tb)     


    #--------------------------------------------------------------------------
    def show(self):        
        self.main_win.show()   

    
