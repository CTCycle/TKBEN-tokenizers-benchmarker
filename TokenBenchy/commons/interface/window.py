from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QThreadPool, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QPushButton, QCheckBox, QPlainTextEdit, QSpinBox,
                               QMessageBox, QComboBox, QTextEdit, QProgressBar,
                               QGraphicsScene, QGraphicsPixmapItem, QGraphicsView)

from TokenBenchy.commons.variables import EnvironmentVariables
from TokenBenchy.commons.interface.events import DatasetEvents, BenchmarkEvents, VisualizationEnvents
from TokenBenchy.commons.configurations import Configurations
from TokenBenchy.commons.interface.workers import Worker
from TokenBenchy.commons.constants import UI_PATH
from TokenBenchy.commons.logger import logger
        


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
        self.tokenizers = None       
        self.figures = []
        self.pixmaps = None
        self.current_fig = 0

        # initial settings
        self.config_manager = Configurations()
        self.configurations = self.config_manager.get_configurations()
    
        self.threadpool = QThreadPool.globalInstance()
        self._data_worker = None
        self._tokenizer_worker = None
        self._benchmark_worker = None

        # get Hugging Face access token
        EV = EnvironmentVariables()
        self.hf_access_token = EV.get_HF_access_token()

        # persistent handlers
        self.loading_handler = DatasetEvents(self.configurations, self.hf_access_token)
        self.benchmark_handler = BenchmarkEvents(self.configurations, self.hf_access_token)
        self.figures_handler = VisualizationEnvents(self.configurations)              
        
        # setup UI elements
        self._setup_configurations()
        self._connect_signals()
        self._set_states()

        # --- prepare graphics view for figures ---
        QGuiApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        self.view = self.main_win.findChild(QGraphicsView, "figureCanvas")
        self.scene = QGraphicsScene()
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.view.setScene(self.scene)

    # [SHOW WINDOW]
    ###########################################################################
    def show(self):        
        self.main_win.show()

    # [HELPERS FOR SETTING CONNECTIONS]
    ###########################################################################
    def _set_states(self): 
        self.progress_bar = self.main_win.findChild(QProgressBar, "progressBar")
        self.progress_bar.setValue(0)   

    #--------------------------------------------------------------------------
    def _connect_button(self, button_name: str, slot):        
        button = self.main_win.findChild(QPushButton, button_name)
        button.clicked.connect(slot) 

    #--------------------------------------------------------------------------
    def _connect_combo_box(self, combo_name: str, slot):        
        combo = self.main_win.findChild(QComboBox, combo_name)
        combo.currentTextChanged.connect(slot)

    #--------------------------------------------------------------------------
    def _send_message(self, message): 
        self.main_win.statusBar().showMessage(message)

    # [SETUP]
    ###########################################################################
    def _setup_configurations(self):              
        self.use_custom_data = self.main_win.findChild(QCheckBox, "useCustomDataset")
        self.set_remove_invalid = self.main_win.findChild(QCheckBox, "removeInvalid")
        self.check_custom_token = self.main_win.findChild(QCheckBox, "includeCustomToken")
        self.check_include_NSL = self.main_win.findChild(QCheckBox, "includeNSL")
        self.check_reduce = self.main_win.findChild(QCheckBox, "reduceSize")
        self.set_save_imgs = self.main_win.findChild(QCheckBox, "saveImages")
        
        self.set_num_docs = self.main_win.findChild(QSpinBox, "numDocs")
       
        # connect their toggled signals to our updater
        self.use_custom_data.toggled.connect(self._update_settings)
        self.set_remove_invalid.toggled.connect(self._update_settings) 
        self.check_custom_token.toggled.connect(self._update_settings) 
        self.check_include_NSL.toggled.connect(self._update_settings)
        self.check_reduce.toggled.connect(self._update_settings) 
        self.set_num_docs.valueChanged.connect(self._update_settings)         
      
    #--------------------------------------------------------------------------
    def _connect_signals(self):        
        self._connect_combo_box("selectTokenizers", self.update_tokenizers_from_combo)
        self._connect_button("loadDataset", self.load_and_process_dataset)
        self._connect_button("analyzeDataset", self.run_dataset_analysis)       
        self._connect_button("runBenchmarks", self.run_tokenizers_benchmark)        
        self._connect_button("visualizeResults", self.generate_figures)  
        self._connect_button("previousImg", self.show_previous_figure)
        self._connect_button("nextImg", self.show_next_figure)       
       
    # [SLOT]
    ###########################################################################
    # It's good practice to define methods that act as slots within the class
    # that manages the UI elements. These slots can then call methods on the
    # handler objects. Using @Slot decorator is optional but good practice
    #--------------------------------------------------------------------------
    @Slot()
    def _update_settings(self):        
        self.config_manager.update_value('use_custom_dataset', self.use_custom_data.isChecked())
        self.config_manager.update_value('remove_invalid_documents', self.set_remove_invalid.isChecked())
        self.config_manager.update_value('include_custom_tokenizer', self.check_custom_token.isChecked())
        self.config_manager.update_value('include_NSL', self.check_include_NSL.isChecked())
        self.config_manager.update_value('reduce_output_size', self.check_reduce.isChecked())
        self.config_manager.update_value('save_images', self.set_save_imgs.isChecked())  
        self.config_manager.update_value('num_documents', self.set_num_docs.value())    

    #--------------------------------------------------------------------------
    @Slot()
    def load_and_process_dataset(self): 
        self.main_win.findChild(QPushButton, "loadDataset").setEnabled(False)

        corpus_text = self.main_win.findChild(QTextEdit, "datasetCorpus").toPlainText()
        config_text = self.main_win.findChild(QTextEdit, "datasetConfig").toPlainText()         
        corpus_text = corpus_text.replace('\n', ' ').strip()
        config_text = config_text.replace('\n', ' ').strip()     

        # update configurations with the text from the input boxes and reinitialize
        # the loading handler with the new configurations
        dataset_config = {'corpus': corpus_text, 'config': config_text} 
        self.config_manager.update_value('DATASET', dataset_config)
        self.configurations = self.config_manager.get_configurations() 
        self.loading_handler = DatasetEvents(self.configurations, self.hf_access_token) 
        
        # send message to status bar
        self._send_message(
            f"Downloading dataset {corpus_text} (configuration: {config_text})")

        # initialize worker for asynchronous loading of the dataset
        # functions that are passed to the worker will be executed in a separate thread
        self._data_worker = Worker(self.loading_handler.load_and_process_dataset)
        worker = self._data_worker       
        worker.signals.finished.connect(self.on_dataset_loaded)
        worker.signals.error.connect(self.on_dataset_error)
        self.threadpool.start(worker)

    #--------------------------------------------------------------------------
    @Slot()
    def run_dataset_analysis(self):
        if self.text_dataset is None:
            message = "Please load a dataset before running analysis!"
            QMessageBox.warning(self.main_win,
                                "Missing dataset",
                                message)
            return None    
            
        self.main_win.findChild(QPushButton, "analyzeDataset").setEnabled(False)

        self.configurations = self.config_manager.get_configurations() 
        self.benchmark_handler = BenchmarkEvents(
            self.configurations, self.hf_access_token)  

        # send message to status bar        
        self._send_message("Computing statistics for the selected dataset")
       
        # initialize worker for asynchronous loading of the dataset
        self._data_worker = Worker(
            self.benchmark_handler.calculate_dataset_statistics,
            self.text_dataset)
        
        worker = self._data_worker      
        worker.signals.finished.connect(self.on_analysis_success)
        worker.signals.error.connect(self.on_analysis_error)
        self.threadpool.start(worker)  

    #--------------------------------------------------------------------------
    @Slot(str)
    def update_tokenizers_from_combo(self, text: str):
        tokenizers = self.main_win.findChild(QPlainTextEdit, "tokenizersToBenchmark")  
        existing = set(tokenizers.toPlainText().splitlines())
        if text not in existing:
            tokenizers.appendPlainText(text)   

    #--------------------------------------------------------------------------
    @Slot()
    def run_tokenizers_benchmark(self):
        self.main_win.findChild(QPushButton, "runBenchmarks").setEnabled(False)

        tokenizers = self.main_win.findChild(QPlainTextEdit, "tokenizersToBenchmark") 
        tokenizers_name = tokenizers.toPlainText().splitlines()
        if len(tokenizers_name) == 0 or self.text_dataset is None:
            message = "Please load both the tokenizers and the text dataset before running benchmarks!"
            QMessageBox.warning(self.main_win,
                                "Cannot run benchmarks",
                                message)
            return None

        tokenizers_name = [x.replace('\n', ' ').strip() for x in tokenizers_name]
        self.config_manager.update_value('TOKENIZERS', tokenizers_name)

        # initialize the benchmark handler with the current configurations
        self.configurations = self.config_manager.get_configurations() 
        self.benchmark_handler = BenchmarkEvents(
            self.configurations, self.hf_access_token)              

        # send message to status bar
        self._send_message("Running tokenizers benchmark...")  

        # initialize worker for asynchronous loading of the dataset
        # functions that are passed to the worker will be executed in a separate thread
        self._benchmark_worker = Worker(
           self.benchmark_handler.execute_benchmarks, 
           self.text_dataset)  
        
        worker = self._benchmark_worker   

        # inject the progress signal into the worker   
        self.progress_bar.setValue(0)    
        worker.signals.progress.connect(self.progress_bar.setValue)
        # connect the finished and error signals to their respective slots 
        worker.signals.finished.connect(self.on_benchmark_finished)
        worker.signals.error.connect(self.on_benchmark_error)
        self.threadpool.start(worker)  

    #--------------------------------------------------------------------------
    @Slot()
    def generate_figures(self):     
        self.main_win.findChild(QPushButton, "visualizeResults").setEnabled(False)

        self.configurations = self.config_manager.get_configurations() 
        self.figures_handler = VisualizationEnvents(self.configurations)
        
        # send message to status bar
        self._send_message("Generating benchmark results figures")  

        # initialize worker for asynchronous loading of the dataset
        # functions that are passed to the worker will be executed in a separate thread
        self._benchmark_worker = Worker(
           self.figures_handler.visualize_benchmark_results)  
        
        worker = self._benchmark_worker         
       
        # connect the finished and error signals to their respective slots 
        worker.signals.finished.connect(self.on_plots_generated)
        worker.signals.error.connect(self.on_plots_error)
        self.threadpool.start(worker) 

    #--------------------------------------------------------------------------
    @Slot()
    def _update_graphics_view(self):
        if not self.figures:
            return      
        self.pixmap_item.setPixmap(self.pixmaps[self.current_fig])
        self.scene.setSceneRect(self.pixmaps[self.current_fig].rect())
        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    #--------------------------------------------------------------------------
    @Slot()
    def show_previous_figure(self):       
        if self.current_fig > 0:
            self.current_fig -= 1
            self._update_graphics_view()

    #--------------------------------------------------------------------------
    @Slot()
    def show_next_figure(self):       
        if self.current_fig < len(self.figures) - 1:
            self.current_fig += 1
            self._update_graphics_view()


    # [POSITIVE OUTCOME HANDLERS]
    ###########################################################################    
    @Slot(object)
    def on_dataset_loaded(self, datasets):             
        self.text_dataset = datasets
        config = self.config_manager.get_configurations().get('DATASET', {})
        corpus = config.get('corpus', 'NA')  
        config = config.get('config', 'NA')         
        message = f'text dataset has been loaded: {corpus} with config {config}' 
        self.loading_handler.handle_success(self.main_win, message)  
        self.main_win.findChild(QPushButton, "loadDataset").setEnabled(True)

    #--------------------------------------------------------------------------
    @Slot(object)
    def on_analysis_success(self, result):                  
        config = self.config_manager.get_configurations().get('DATASET', {})
        corpus = config.get('corpus', 'NA')  
        config = config.get('config', 'NA')         
        message = f'{corpus} - {config} analysis is finished' 
        self.benchmark_handler.handle_success(self.main_win, message)
        self.main_win.findChild(QPushButton, "analyzeDataset").setEnabled(True) 
    
    #--------------------------------------------------------------------------
    @Slot(object)
    def on_benchmark_finished(self, tokenizers):
        self.tokenizers = tokenizers               
        message = f'{len(tokenizers)} selected tokenizers have been benchmarked'   
        self.benchmark_handler.handle_success(self.main_win, message)   
        self.main_win.findChild(QPushButton, "runBenchmarks").setEnabled(True)    
    
    #--------------------------------------------------------------------------
    @Slot(object)    
    def on_plots_generated(self, figures):        
        self.figures = figures
        self.pixmaps = [self.figures_handler.convert_fig_to_qpixmap(p) for p in self.figures]
        self.current_fig = 0
        self._update_graphics_view()
        self.figures_handler.handle_success(
            self.main_win, 'Benchmark results plots have been generated')
        self.main_win.findChild(QPushButton, "visualizeResults").setEnabled(True)       
    
    # [NEGATIVE OUTCOME HANDLERS]
    ###########################################################################    
    @Slot(tuple)
    def on_dataset_error(self, err_tb):
        self.loading_handler.handle_error(self.main_win, err_tb)  
        self.main_win.findChild(QPushButton, "loadDataset").setEnabled(True)

    #--------------------------------------------------------------------------
    @Slot(tuple)
    def on_analysis_error(self, err_tb):
        self.benchmark_handler.handle_error(self.main_win, err_tb) 
        self.main_win.findChild(QPushButton, "analyzeDataset").setEnabled(True) 

    #--------------------------------------------------------------------------
    @Slot(tuple)
    def on_benchmark_error(self, err_tb):
        self.benchmark_handler.handle_error(self.main_win, err_tb)         
        self.progress_bar.setValue(0) 
        self.main_win.findChild(QPushButton, "runBenchmarks").setEnabled(True) 

    #--------------------------------------------------------------------------
    @Slot(tuple)
    def on_plots_error(self, err_tb):
        self.figures_handler.handle_error(self.main_win, err_tb) 
        self.main_win.findChild(QPushButton, "visualizeResults").setEnabled(True)
    

    
        
   

    
