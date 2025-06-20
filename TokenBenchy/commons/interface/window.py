from TokenBenchy.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

from functools import partial
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QThreadPool, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (QPushButton, QCheckBox, QPlainTextEdit, QSpinBox,
                               QMessageBox, QComboBox, QTextEdit, QProgressBar,
                               QGraphicsScene, QGraphicsPixmapItem, QGraphicsView)


from TokenBenchy.commons.utils.database import TokenBenchyDatabase
from TokenBenchy.commons.interface.events import DatasetEvents, BenchmarkEvents, VisualizationEnvents
from TokenBenchy.commons.configuration import Configuration
from TokenBenchy.commons.interface.workers import Worker
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
        self.tokenizers = []            
        self.pixmaps = []
        self.current_fig = 0

        # initial settings
        self.config_manager = Configuration()
        self.configuration = self.config_manager.get_configuration()
    
        # set thread pool for the workers
        self.threadpool = QThreadPool.globalInstance()
        self.worker = None
        self.worker_running = False          

        # get Hugging Face access token        
        self.hf_access_token = EV.get_HF_access_token()

        # initialize database
        self.database = TokenBenchyDatabase(self.configuration)
        self.database.initialize_database() 

        # persistent handlers
        self.loading_handler = DatasetEvents(
            self.database, self.configuration, self.hf_access_token)        
        self.benchmark_handler = BenchmarkEvents(
            self.database, self.configuration, self.hf_access_token)
        self.figures_handler = VisualizationEnvents(
            self.database, self.configuration)

        # setup UI elements
        self._set_states()
        self.widgets = {}
        self._setup_configuration([
            (QPushButton,'stopThread','stop_thread'),
            (QProgressBar, "progressBar", 'progress_bar'), 
            (QCheckBox, "useCustomDataset", 'use_custom_dataset'),
            (QCheckBox, "removeInvalid", 'remove_invalid_docs'),
            (QCheckBox, "includeCustomToken", 'custom_tokenizer'),
            (QCheckBox, "includeNSL", 'include_NSL'),
            (QCheckBox, "reduceSize", 'reduce_size'),           
            (QSpinBox,  "numDocs", 'num_documents'),
            (QPushButton,'scanHF','scan_for_tokenizers'),
            (QComboBox, "selectTokenizers", 'combo_tokenizers'),
            (QPushButton, "loadDataset", 'load_dataset'),
            (QPushButton, "analyzeDataset", 'analyze_dataset'),
            (QPushButton, "runBenchmarks", 'run_benchmarks'),
            (QPushButton, "visualizeResults", 'visualize_results'),
            (QPushButton, "previousImg", 'prev_img'),
            (QPushButton, "nextImg", 'next_img'),
            (QPushButton, "clearImg", 'clear_img'),            
            (QPlainTextEdit, "tokenizersToBenchmark",'tokenizers_to_bench'),
            (QTextEdit, "datasetCorpus", 'text_corpus'),
            (QTextEdit, "datasetConfig", 'text_config'),
            (QGraphicsView, "figureCanvas",  'view')])
        
        self._connect_signals([
            ('stop_thread','clicked',self.stop_running_worker),           
            ('combo_tokenizers', 'currentTextChanged', self.update_tokenizers_from_combo),
            ('scan_for_tokenizers', 'clicked', self.find_tokenizers_identifiers),
            ('load_dataset', 'clicked', self.load_and_process_dataset),
            ('analyze_dataset', 'clicked', self.run_dataset_analysis),
            ('run_benchmarks', 'clicked', self.run_tokenizers_benchmark),
            ('visualize_results', 'clicked', self.generate_figures),
            ('prev_img', 'clicked', self.show_previous_figure),
            ('next_img', 'clicked', self.show_next_figure),
            ('clear_img', 'clicked', self.clear_figures)])
        
        self._auto_connect_settings() 
        self._set_graphics() 

    # [SHOW WINDOW]
    ###########################################################################
    def show(self):        
        self.main_win.show()    
    
    # [HELPERS]
    ###########################################################################
    def connect_update_setting(self, widget, signal_name, config_key, getter=None):
        if getter is None:
            if isinstance(widget, (QCheckBox)):
                getter = widget.isChecked
            elif isinstance(widget, (QSpinBox)):
                getter = widget.value
            elif isinstance(widget, QComboBox):
                getter = widget.currentText
           
        signal = getattr(widget, signal_name)
        signal.connect(partial(self._update_single_setting, config_key, getter))

    #--------------------------------------------------------------------------
    def _update_single_setting(self, config_key, getter, *args):
        value = getter()
        self.config_manager.update_value(config_key, value)

    #--------------------------------------------------------------------------
    def _auto_connect_settings(self):
        connections = [            
            ('use_custom_dataset', 'toggled', 'use_custom_dataset'),
            ('remove_invalid_docs', 'toggled', 'remove_invalid_documents'),
            ('custom_tokenizer', 'toggled', 'include_custom_tokenizer'),
            ('include_NSL', 'toggled', 'include_NSL'),      
            ('num_documents', 'valueChanged', 'num_documents')]    

        for attr, signal_name, config_key in connections:
            widget = self.widgets[attr]
            self.connect_update_setting(widget, signal_name, config_key)       

    #--------------------------------------------------------------------------
    def _set_states(self): 
        self.progress_bar = self.main_win.findChild(QProgressBar, "progressBar")
        self.progress_bar.setValue(0)   

    #--------------------------------------------------------------------------
    def _set_graphics(self):
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

    #--------------------------------------------------------------------------
    def _connect_button(self, button_name: str, slot):        
        button = self.main_win.findChild(QPushButton, button_name)
        button.clicked.connect(slot) 

    #--------------------------------------------------------------------------
    def _connect_combo_box(self, combo_name: str, slot):        
        combo = self.main_win.findChild(QComboBox, combo_name)
        combo.currentTextChanged.connect(slot)

    #--------------------------------------------------------------------------
    def _start_worker(self, worker : Worker, on_finished, on_error, on_interrupted,
                      update_progress=True): 
        if update_progress:       
            self.progress_bar.setValue(0)
            worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.finished.connect(on_finished)
        worker.signals.error.connect(on_error)        
        worker.signals.interrupted.connect(on_interrupted)
        self.threadpool.start(worker)
        self.worker_running = True

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
    Slot()
    def stop_running_worker(self):
        if self.worker is not None:
            self.worker.stop()       
        self._send_message("Interrupt requested. Waiting for threads to stop...")

    #--------------------------------------------------------------------------
    @Slot()
    def load_and_process_dataset(self): 
        if self.worker_running:            
            return 
        
        corpus_text = self.main_win.findChild(QTextEdit, "datasetCorpus").toPlainText()
        config_text = self.main_win.findChild(QTextEdit, "datasetConfig").toPlainText()  
        dataset_config = {'corpus': corpus_text.replace('\n', ' ').strip(), 
                          'config': config_text.replace('\n', ' ').strip()} 
        
        # update configuration with the text from the input boxes and reinitialize
        # the loading handler with the new configuration        
        self.config_manager.update_value('DATASET', dataset_config)
        self.configuration = self.config_manager.get_configuration() 
        self.loading_handler = DatasetEvents(self.database, self.configuration, self.hf_access_token) 
        
        # send message to status bar
        self._send_message(
            f"Downloading dataset {corpus_text} (configuration: {config_text})")        
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(self.loading_handler.load_and_process_dataset)

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_dataset_loaded,
            on_error=self.on_dataset_error,
            on_interrupted=self.on_task_interrupted)       

    #--------------------------------------------------------------------------
    @Slot()
    def run_dataset_analysis(self):
        if self.worker_running:            
            return 

        if self.text_dataset is None:
            message = "Please load a dataset before running analysis!"
            QMessageBox.warning(self.main_win,
                                "Missing dataset",
                                message)
            return    
        
        self.configuration = self.config_manager.get_configuration() 
        self.benchmark_handler = BenchmarkEvents(
            self.database, self.configuration, self.hf_access_token)  

        # send message to status bar        
        self._send_message("Computing statistics for the selected dataset")       
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.benchmark_handler.run_dataset_evaluation_pipeline,
            self.text_dataset)   

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_analysis_success,
            on_error=self.on_benchmark_error,
            on_interrupted=self.on_task_interrupted)            

    #--------------------------------------------------------------------------
    @Slot(str)
    def find_tokenizers_identifiers(self):
        if self.worker_running:            
            return 
             
        self.configuration = self.config_manager.get_configuration() 
        self.benchmark_handler = BenchmarkEvents(
            self.database, self.configuration, self.hf_access_token)

        # send message to status bar        
        self._send_message("Looking for available tokenizers in Hugging Face")       
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
            self.benchmark_handler.get_tokenizer_identifiers, limit=1000)
          
        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_tokenizers_fetched,
            on_error=self.on_benchmark_error,
            on_interrupted=self.on_task_interrupted)          

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
        if self.worker_running:            
            return 
        
        tokenizers = self.main_win.findChild(QPlainTextEdit, "tokenizersToBenchmark") 
        tokenizers_name = tokenizers.toPlainText().splitlines()
        if len(tokenizers_name)==0 or self.text_dataset is None:            
            return None

        tokenizers_name = [x.replace('\n', ' ').strip() for x in tokenizers_name]
        self.config_manager.update_value('TOKENIZERS', tokenizers_name)

        # initialize the benchmark handler with the current configuration
        self.configuration = self.config_manager.get_configuration() 
        self.benchmark_handler = BenchmarkEvents(
            self.database, self.configuration, self.hf_access_token)              

        # send message to status bar
        self._send_message("Running tokenizers benchmark...")          
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(
           self.benchmark_handler.execute_benchmarks, 
           self.text_dataset)         
        
        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_benchmark_finished,
            on_error=self.on_benchmark_error,
            on_interrupted=self.on_task_interrupted)      

    #--------------------------------------------------------------------------
    @Slot()
    def generate_figures(self):     
        if self.worker_running:            
            return 
        
        self.configuration = self.config_manager.get_configuration() 
        self.figures_handler = VisualizationEnvents(self.configuration)        
        # send message to status bar
        self._send_message("Generating benchmark results figures")   
        # functions that are passed to the worker will be executed in a separate thread
        self.worker = Worker(self.figures_handler.visualize_benchmark_results) 

        # start worker and inject signals
        self._start_worker(
            self.worker, on_finished=self.on_plots_generated,
            on_error=self.on_plots_error,
            on_interrupted=self.on_task_interrupted)        

    #--------------------------------------------------------------------------
    @Slot()
    def _update_graphics_view(self):
        if not self.pixmaps:
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
        if self.current_fig < len(self.pixmaps) - 1:
            self.current_fig += 1
            self._update_graphics_view()

    #--------------------------------------------------------------------------
    @Slot()
    def clear_figures(self):
        self.pixmaps.clear()
        self.current_fig = 0
        # set the existing pixmap_item to an empty QPixmap
        self.pixmap_item.setPixmap(QPixmap())
        # Shrink the scene rect so nothing is visible
        self.scene.setSceneRect(0, 0, 0, 0)
        # Force an immediate repaint
        self.view.viewport().update()


    # [POSITIVE OUTCOME HANDLERS]
    ###########################################################################    
    @Slot(object)
    def on_dataset_loaded(self, datasets):             
        self.text_dataset = datasets
        config = self.config_manager.get_configuration().get('DATASET', {})
        corpus = config.get('corpus', 'NA')  
        config = config.get('config', 'NA')             
        message = f'Text dataset has been loaded: {corpus} with config {config}' 
        logger.info(message)

        self.loading_handler.handle_success(self.main_win, message)  
        self.worker_running = False 

    #--------------------------------------------------------------------------
    @Slot(object)
    def on_analysis_success(self, result):                  
        config = self.config_manager.get_configuration().get('DATASET', {})
        corpus = config.get('corpus', 'NA')  
        config = config.get('config', 'NA')         
        message = f'{corpus} - {config} analysis is finished' 
        self.benchmark_handler.handle_success(self.main_win, message)
        self.worker_running = False

    #--------------------------------------------------------------------------
    @Slot(object)
    def on_tokenizers_fetched(self, identifiers):
        combo = self.main_win.findChild(QComboBox, "selectTokenizers")
        existing = {combo.itemText(i) for i in range(combo.count())}
        for identifier in identifiers:
            if identifier not in existing:
                combo.addItem(identifier)
                  
        message = f'{len(identifiers)} tokenizer identifiers fetched from HuggingFace'   
        self.benchmark_handler.handle_success(self.main_win, message)   
        self.worker_running = False           
    
    #--------------------------------------------------------------------------
    @Slot(object)
    def on_benchmark_finished(self, tokenizers):
        self.tokenizers = tokenizers               
        message = f'{len(tokenizers)} selected tokenizers have been benchmarked'   
        self.benchmark_handler.handle_success(self.main_win, message)   
        self.worker_running = False 
    
    #--------------------------------------------------------------------------
    @Slot(object)    
    def on_plots_generated(self, figures): 
        if figures:        
            self.pixmaps.extend(
                [self.figures_handler.convert_fig_to_qpixmap(p) for p in figures])
        self.current_fig = 0
        self._update_graphics_view()
        self.figures_handler.handle_success(
            self.main_win, 'Benchmark results plots have been generated')
        self.worker_running = False      
    
    # [NEGATIVE OUTCOME HANDLERS]
    ###########################################################################    
    @Slot(tuple)
    def on_dataset_error(self, err_tb):
        self.loading_handler.handle_error(self.main_win, err_tb)  
        self.worker_running = False     

    #--------------------------------------------------------------------------
    @Slot(tuple)
    def on_benchmark_error(self, err_tb):
        self.benchmark_handler.handle_error(self.main_win, err_tb)         
        self.progress_bar.setValue(0) 
        self.worker_running = False

    #--------------------------------------------------------------------------
    @Slot(tuple)
    def on_plots_error(self, err_tb):
        self.figures_handler.handle_error(self.main_win, err_tb) 
        self.worker_running = False  

    #--------------------------------------------------------------------------
    def on_task_interrupted(self):
        self.progress_bar.setValue(0)
        self._send_message('Current task has been interrupted by user') 
        logger.warning('Current task has been interrupted by user')   
        self.worker_running = False     
    

    
        
   

    
