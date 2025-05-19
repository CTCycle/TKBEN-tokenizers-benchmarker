from TokenBenchy.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QThreadPool, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QPushButton, QCheckBox, QPlainTextEdit, QSpinBox,
                               QMessageBox, QComboBox, QTextEdit, QProgressBar,
                               QGraphicsScene, QGraphicsPixmapItem, QGraphicsView)

from TokenBenchy.commons.variables import EnvironmentVariables
from TokenBenchy.commons.interface.events import DatasetEvents, BenchmarkEvents, VisualizationEnvents
from TokenBenchy.commons.configuration import Configuration
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
        self.config_manager = Configuration()
        self.configuration = self.config_manager.get_configuration()
    
        self.threadpool = QThreadPool.globalInstance()
        self._data_worker = None
        self._tokenizer_worker = None
        self._benchmark_worker = None

        # get Hugging Face access token        
        self.hf_access_token = EV.get_HF_access_token()

        # persistent handlers
        self.loading_handler = DatasetEvents(self.configuration, self.hf_access_token)
        self.benchmark_handler = BenchmarkEvents(self.configuration, self.hf_access_token)
        self.figures_handler = VisualizationEnvents(self.configuration)         

        # setup UI elements
        self._set_states()
        self.widgets = {}
        self._setup_configuration([
            (QCheckBox, "useCustomDataset", 'use_custom_data'),
            (QCheckBox, "removeInvalid", 'set_remove_invalid'),
            (QCheckBox, "includeCustomToken", 'check_custom_token'),
            (QCheckBox, "includeNSL", 'check_include_NSL'),
            (QCheckBox, "reduceSize", 'check_reduce'),
            (QCheckBox, "saveImages", 'set_save_img'),
            (QSpinBox,  "numDocs", 'set_num_docs'),
            (QComboBox, "selectTokenizers", 'combo_tokenizers'),
            (QPushButton, "loadDataset", 'load_dataset'),
            (QPushButton, "analyzeDataset", 'analyze_dataset'),
            (QPushButton, "runBenchmarks", 'run_benchmarks'),
            (QPushButton, "visualizeResults", 'visualize_results'),
            (QPushButton, "previousImg", 'prev_img'),
            (QPushButton, "nextImg", 'next_img'),
            (QPushButton, "clearImg", 'clear_img'),
            (QProgressBar, "progressBar", 'progress_bar'),
            (QPlainTextEdit, "tokenizersToBenchmark",'tokenizers_to_bench'),
            (QTextEdit, "datasetCorpus", 'text_corpus'),
            (QTextEdit, "datasetConfig", 'text_config'),
            (QGraphicsView, "figureCanvas",  'view')])
        
        self._connect_signals([
            ('use_custom_data', 'toggled', self._update_settings),
            ('set_remove_invalid', 'toggled', self._update_settings),
            ('check_custom_token', 'toggled', self._update_settings),
            ('check_include_NSL', 'toggled', self._update_settings),
            ('check_reduce', 'toggled', self._update_settings),
            ('set_save_img', 'toggled', self._update_settings),
            ('set_num_docs', 'valueChanged', self._update_settings),
            ('combo_tokenizers', 'currentTextChanged', self.update_tokenizers_from_combo),
            ('load_dataset', 'clicked', self.load_and_process_dataset),
            ('analyze_dataset', 'clicked', self.run_dataset_analysis),
            ('run_benchmarks', 'clicked', self.run_tokenizers_benchmark),
            ('visualize_results', 'clicked', self.generate_figures),
            ('prev_img', 'clicked', self.show_previous_figure),
            ('next_img', 'clicked', self.show_next_figure),
            ('clear_img', 'clicked', self.clear_figures)])
        
        # --- prepare graphics view for figures ---
        self.view = self.main_win.findChild(QGraphicsView, "figureCanvas")
        self.scene = QGraphicsScene()
        self.pixmap_item = QGraphicsPixmapItem()
        # make pixmap scaling use smooth interpolation
        self.pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self.scene.addItem(self.pixmap_item)
        self.view.setScene(self.scene)
        # set canvas hints
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.view.setRenderHint(QPainter.TextAntialiasing, True) 

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
    def _setup_configuration(self, widget_defs):
        for cls, name, attr in widget_defs:
            w = self.main_win.findChild(cls, name)
            setattr(self, attr, w)
            self.widgets[attr] = w

    #--------------------------------------------------------------------------
    def _connect_signals(self, connections):
        for attr, signal, slot in connections:
            widget = self.widgets[attr]
            getattr(widget, signal).connect(slot)
       
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
        self.config_manager.update_value('save_images', self.set_save_img.isChecked())  
        self.config_manager.update_value('num_documents', self.set_num_docs.value())    

    #--------------------------------------------------------------------------
    @Slot()
    def load_and_process_dataset(self): 
        self.main_win.findChild(QPushButton, "loadDataset").setEnabled(False)
        corpus_text = self.main_win.findChild(QTextEdit, "datasetCorpus").toPlainText()
        config_text = self.main_win.findChild(QTextEdit, "datasetConfig").toPlainText()         
        corpus_text = corpus_text.replace('\n', ' ').strip()
        config_text = config_text.replace('\n', ' ').strip()     

        # update configuration with the text from the input boxes and reinitialize
        # the loading handler with the new configuration
        dataset_config = {'corpus': corpus_text, 'config': config_text} 
        self.config_manager.update_value('DATASET', dataset_config)
        self.configuration = self.config_manager.get_configuration() 
        self.loading_handler = DatasetEvents(self.configuration, self.hf_access_token) 
        
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

        self.configuration = self.config_manager.get_configuration() 
        self.benchmark_handler = BenchmarkEvents(
            self.configuration, self.hf_access_token)  

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

        # initialize the benchmark handler with the current configuration
        self.configuration = self.config_manager.get_configuration() 
        self.benchmark_handler = BenchmarkEvents(
            self.configuration, self.hf_access_token)              

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

        self.configuration = self.config_manager.get_configuration() 
        self.figures_handler = VisualizationEnvents(self.configuration)
        
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

        raw_pix = self.pixmaps[self.current_fig]
        view_size = self.view.viewport().size()
        # scale images to the canvas pixel dimensions with smooth filtering
        scaled = raw_pix.scaled(
            view_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        self.pixmap_item.setPixmap(scaled)
        self.scene.setSceneRect(scaled.rect())

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

    #--------------------------------------------------------------------------
    @Slot()
    def clear_figures(self):       
        self.figures = []
        self.pixmaps = None


    # [POSITIVE OUTCOME HANDLERS]
    ###########################################################################    
    @Slot(object)
    def on_dataset_loaded(self, datasets):             
        self.text_dataset = datasets
        config = self.config_manager.get_configuration().get('DATASET', {})
        corpus = config.get('corpus', 'NA')  
        config = config.get('config', 'NA')         
        message = f'text dataset has been loaded: {corpus} with config {config}' 
        self.loading_handler.handle_success(self.main_win, message)  
        self.main_win.findChild(QPushButton, "loadDataset").setEnabled(True)

    #--------------------------------------------------------------------------
    @Slot(object)
    def on_analysis_success(self, result):                  
        config = self.config_manager.get_configuration().get('DATASET', {})
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
    

    
        
   

    
