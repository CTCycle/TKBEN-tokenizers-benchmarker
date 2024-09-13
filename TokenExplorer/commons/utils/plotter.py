import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
tqdm.pandas()

from TokenExplorer.commons.constants import CONFIG, BENCHMARK_RESULTS_PATH, BENCHMARK_PATH
from TokenExplorer.commons.logger import logger

             
# [TOKENIZERS EXPLORER]
###############################################################################
class DataPlotter:

    def __init__(self):

        try:
            filepath = os.path.join(BENCHMARK_RESULTS_PATH, 'tokenizers_benchmark.csv')                
            self.benchmark_data = pd.read_csv(filepath, sep=';', encoding='utf-8')
        except:
            logger.error(f'Could not load benchmark results from {BENCHMARK_RESULTS_PATH}')
            self.benchmark_data = None
        try:
            filepath = os.path.join(BENCHMARK_PATH, 'NSL_benchmark.csv')                
            self.NSL_data = pd.read_csv(filepath, sep=';', encoding='utf-8')
        except:
            logger.error(f'Could not load benchmark results from {BENCHMARK_PATH}')    
            self.NSL_data = None    

    #--------------------------------------------------------------------------
    def benchmarks_boxplot(self, data, path, x_vals, y_vals, y_label, hue=None, title=''):        

        if data is not None and hue is not None:
            hue = data[hue]
        
        if data is not None:
            plt.figure(figsize=(14, 16))
            sns.boxplot(x=data[x_vals], y=data[y_vals], hue=hue, data=data)            
            plt.xticks(rotation=45, ha='right', fontsize=14)
            plt.yticks(fontsize=14)          
            plt.xlabel('', fontsize=14)
            plt.ylabel(y_label, fontsize=14)
            plt.title(title, fontsize=14, y=1.02) 
            plt.legend(fontsize=14)               
            plt.tight_layout()               
            plot_loc = os.path.join(path, f'{title}_boxplot.jpeg')
            plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi=600) 
            plt.show(block=False)       
            plt.close()  


    
    
    
