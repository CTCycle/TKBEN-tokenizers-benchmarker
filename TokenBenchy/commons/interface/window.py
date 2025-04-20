from PySide6.QtWidgets import (QPushButton, QCheckBox, QPlainTextEdit, QSpinBox, 
                               QMessageBox, QComboBox, QTextEdit)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, Slot, QObject, Signal, QRunnable, QThreadPool

from TokenBenchy.commons.variables import EnvironmentVariables
from TokenBenchy.commons.utils.data.downloads import DatasetDownloadManager, TokenizersDownloadManager
from TokenBenchy.commons.interface.configurations import Configurations
from TokenBenchy.commons.interface.workers import Worker
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

        # initial settings
        self.config_manager = Configurations()
        self.configurations = self.config_manager.get_configurations()
    
        self.threadpool = QThreadPool.globalInstance()

        # get Hugging Face access token from environmental variables
        EV = EnvironmentVariables()
        self.hf_access_token = EV.get_HF_access_token()

        # --- Create persistent handlers ---
        # These objects will live as long as the MainWindow instance lives 
        self.dataset_handler = DatasetDownloadManager(
            self.configurations, self.hf_access_token)    
        self.token_handler = TokenizersDownloadManager(
            self.configurations, self.hf_access_token)     
        
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
        # set the default value of the wait time box to the current wait time
        self.set_num_docs = self.main_win.findChild(QSpinBox, "numDocs")
        self.set_num_docs.setValue(self.configurations.get('num_docs', 0))       

        # connect their toggled signals to our updater
        self.check_custom_data.toggled.connect(self._update_settings)
        self.check_custom_token.toggled.connect(self._update_settings) 
        self.check_include_NSL.toggled.connect(self._update_settings) 
        self.set_num_docs.valueChanged.connect(self._update_settings) 
      
    #--------------------------------------------------------------------------
    def _connect_signals(self):        
        self._connect_combo_box("selectTokenizers", self.on_tokenizer_selection_from_combo)
        self._connect_button("loadDataset", self.load_selected_dataset)
        

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
        self.config_manager.update_value('wait_time', self.set_num_docs.value())
        self.configurations = self.config_manager.get_configurations()
        
    #--------------------------------------------------------------------------
    @Slot(str)
    def on_tokenizer_selection_from_combo(self, text: str):
        text_edit = self.main_win.findChild(QPlainTextEdit, "tokenizersToBenchmark")  
        existing = set(text_edit.toPlainText().splitlines())
        if text not in existing:
            text_edit.appendPlainText(text)     

    #--------------------------------------------------------------------------
    @Slot()
    def load_selected_dataset(self):  
        set_data_corpus = self.main_win.findChild(QTextEdit, "datasetCorpus")
        set_data_config = self.main_win.findChild(QTextEdit, "datasetConfig")
        dataset_config = {'corpus' : set_data_corpus.toPlainText(),
                          'config' : set_data_config.toPlainText()}     

        self.config_manager.update_value('DATASET', dataset_config)
        self.configurations = self.config_manager.get_configurations()  

        # 4. Kick off the background download
        self.download_handler = DatasetDownloadManager(
            self.configurations, self.hf_access_token)
        
        # 3) Wrap it in a Worker and hook up inline callbacks:
        worker = Worker(self.download_handler.dataset_download)

        # -- on success: stash the dict and pop an info box:
        worker.signals.finished.connect(
        lambda datasets: setattr(self, 'text_dataset', datasets) or
            QMessageBox.information(
            self.main_win, "Text dataset", f"Loaded dataset {dataset_config['corpus']}"))
        worker.signals.error.connect(
            lambda err: QMessageBox.critical(
            self.main_win, "Error", str(err[0])))
       
        self.threadpool.start(worker)

         
       

    #--------------------------------------------------------------------------                 
        # worker = Worker(self.download_handler.dataset_download)
        # worker.signals.finished.connect(
        # lambda ds: QMessageBox.information(
        #     self.main_win, "Downloaded", f"Got {len(ds)} items."))
        # worker.signals.error.connect(lambda err_tb: QMessageBox.critical(
        #     self.main_win, "Download Failed", str(err_tb[0])))
    
        # self.threadpool.start(worker)

    #--------------------------------------------------------------------------
    # @Slot()
    # def verify_webdriver_slot(self):        
    #     is_installed = self.webdriver_handler.is_chromedriver_installed()
    #     QMessageBox.information(
    #     self.main_win,
    #     "Verify Chrome webdriver installation",
    #     is_installed,
    #     QMessageBox.Ok)

    # #--------------------------------------------------------------------------
    # @Slot()
    # def check_webdriver_version_slot(self):        
    #     version = self.webdriver_handler.check_chrome_version()
    #     QMessageBox.information(
    #     self.main_win,
    #     "WebDriver Version",
    #     f"Current ChromeDriver version: {version}",
    #     QMessageBox.Ok)
        
    # #--------------------------------------------------------------------------
    def show(self):        
        self.main_win.show()   

    
