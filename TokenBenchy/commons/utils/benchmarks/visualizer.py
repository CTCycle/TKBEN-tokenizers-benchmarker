import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from TokenBenchy.commons.constants import EVALUATION_PATH
from TokenBenchy.commons.logger import logger


# [TOKENIZERS EXPLORER]
###############################################################################
class VisualizeBenchmarkResults:

    def __init__(self, configuration : dict): 
        self.tokenizers = None       
        self.configuration = configuration                
        self.save_images = configuration.get('save_images', True)
        self.observed_features = [
            'tokens_to_words_ratio', 'AVG_tokens_length', 'bytes_per_token'] 
        self.DPI = 400
        
    #--------------------------------------------------------------------------
    def update_tokenizers_dictionaries(self, tokenizers): 
        self.tokenizers = tokenizers
        self.vocabularies = {k : v.get_vocab() for k, v in tokenizers.items()}        
        self.vocab_len = {k: len(v) for k, v in self.vocabularies.items()}  
        self.vocab_decoded = {}
        self.vocab_len_decoded = {}
        for k, v in tokenizers.items():
            vocabulary = self.vocabularies[k] 
            vocab_indexes = list(vocabulary.values())
            decoded_words = v.decode(vocab_indexes).split() 
            len_vocab = len(decoded_words)
            self.vocab_decoded[k] = decoded_words
            self.vocab_len_decoded[k] = len_vocab

    #--------------------------------------------------------------------------
    def get_vocabulary_report(self):        
        for k, v in self.tokenizers.items():  
            vocab = self.vocabularies[k]                     
            vocab_indexes = list(vocab.values())                    
            vocab_words = list(vocab.keys())
            decoded_words = v.decode(vocab_indexes)   
            decoded_words = decoded_words.split()                  
            intersection = set(vocab_words).intersection(set(decoded_words))    
            not_intersecting = set(vocab_words).symmetric_difference(set(decoded_words))        
            logger.info(f'Tokenizer: {k}')
            logger.info(f'Number of tokens (from vocabulary): {len(vocab_words)}')
            logger.info(f'Number of tokens (from decoding): {len(decoded_words)}')
            logger.info(f'Number of common words: {len(intersection)}')
            logger.info(f'Number of words not in common: {len(not_intersecting)}')

    #--------------------------------------------------------------------------
    def plot_vocabulary_size(self):            
        data = []
        for k, v in self.vocab_len.items():
            data.append(
            {'Tokenizer': k, 'Length': v, 'Type': 'Vocab Length'})
        for k, v in self.vocab_len_decoded.items():
            data.append(
                {'Tokenizer': k, 'Length': v, 'Type': 'Decoded Length'})
        df = pd.DataFrame(data)      
        fig, ax = plt.subplots(figsize=(18, 20), dpi=self.DPI) 
        plt.subplot() 
        sns.barplot(
            x='Tokenizer', y='Length', hue='Type', data=df, 
            palette='viridis', edgecolor='black')        
        ax.set_xlabel('', fontsize=20)        
        ax.set_ylabel('Vocabulary size', fontsize=20)
        ax.set_title('Vocabulary size by tokenizer', fontsize=20, y=1.05)
        ax.tick_params(axis='x', rotation=45, labelsize=20, labelright=False)
        ax.tick_params(axis='y', labelsize=20)
        ax.legend(fontsize=16)
        plt.tight_layout()

        if self.save_images:     
            plot_loc = os.path.join(EVALUATION_PATH, 'vocabulary_size.jpeg')
            plt.savefig(
                plot_loc, bbox_inches='tight', format='jpeg', dpi=self.DPI)  

        return fig     
           
    #--------------------------------------------------------------------------
    def plot_subwords_vs_words(self):
        word_types_data = []        
        for k, v in self.vocabularies.items():
            vocab_words = list(v.keys())
            subwords = [x for x in vocab_words if '##' in x]
            words = [x for x in vocab_words if '##' not in x]
            subwords_perc = len(subwords)/(len(words) + len(subwords)) * 100
            words_perc = len(words)/(len(words) + len(subwords)) * 100
            word_types_data.append({'Vocabulary': k, 'Type': 'Subwords', 'Percentage': subwords_perc})
            word_types_data.append({'Vocabulary': k, 'Type': 'words', 'Percentage': words_perc})
        
        df = pd.DataFrame(word_types_data)
        fig, ax = plt.subplots(figsize=(16, 18), dpi=self.DPI)    
        sns.barplot(
            data=df, x='Vocabulary', y='Percentage', hue='Type', 
            palette='viridis', edgecolor='black')
        
        ax.set_xlabel('', fontsize=16)
        ax.set_ylabel('Percentage (%)', fontsize=16)        
        ax.set_title('Subwords vs Complete words', fontsize=16, y=1.05)
        ax.tick_params(axis='x', rotation=45, labelsize=16, labelright=False)
        ax.tick_params(axis='y', labelsize=16)
        ax.legend(fontsize=16)       
        plt.tight_layout()

        if self.save_images:      
            plot_loc = os.path.join(EVALUATION_PATH, 'subwords_vs_words.jpeg')
            plt.savefig(
                plot_loc, bbox_inches='tight', format='jpeg', dpi=self.DPI)     

        return fig   
          
    #--------------------------------------------------------------------------
    def plot_histogram_tokens_length(self):
        histograms = []       
        for k, v in self.vocabularies.items():
            k_rep = k.replace('/', '_')
            vocab_words = list(v.keys()) 
            decoded_words = self.vocab_decoded[k]               
            vocab_word_lens = [len(x) for x in vocab_words]
            decoded_word_lens = [len(x) for x in decoded_words]
            fig, axs = plt.subplots(2, 1, figsize=(16, 18), sharex=False, dpi=self.DPI)          
            sns.histplot(vocab_word_lens, ax=axs[0], color='skyblue', edgecolor='black', 
                         label='Vocab words', binwidth=1)
            axs[0].set_title(f'Vocab words - {k}', fontsize=16)
            axs[0].set_ylabel('Frequency', fontsize=16)
            axs[0].set_xlabel('word Length', fontsize=16)                         
            sns.histplot(decoded_word_lens, ax=axs[1], color='orange', edgecolor='black', 
                         label='Decoded words', binwidth=1)
            axs[1].set_title(f'Decoded words - {k}', fontsize=16)
            axs[1].set_ylabel('Frequency', fontsize=16)
            axs[1].set_xlabel('word Length', fontsize=16)
            plt.tight_layout()
            histograms.append(fig)              

            if self.save_images:            
                plot_loc = os.path.join(EVALUATION_PATH, f'{k_rep}_words_by_len.jpeg')
                plt.savefig(
                    plot_loc, bbox_inches='tight', format='jpeg', dpi=self.DPI)                   

        return histograms       
                        

    #--------------------------------------------------------------------------
    def plot_boxplot_tokens_length(self):        
        word_lengths = {
            k : [len(x) for x in list(v.keys())] 
            for k, v in self.vocabularies.items()}
        word_lengths_decoded = {
            k : [len(x) for x in v] 
            for k, v in self.vocab_decoded.items()} 
        
        data = []
        for key in word_lengths.keys():
            for length in word_lengths[key]:
                data.append(
                    {'Tokenizer': key, 'word Length': length, 'Type': 'Original'})
            for length in word_lengths_decoded.get(key, []):
                data.append(
                    {'Tokenizer': key, 'word Length': length, 'Type': 'Decoded'})
                
        df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(16, 18), dpi=self.DPI) 
        sns.boxplot(x='Tokenizer', y='word Length', hue='Type', data=df)   
        ax.set_xlabel('', fontsize=16)
        ax.set_ylabel('word Length', fontsize=16)        
        ax.set_title('Distribution of words by length', fontsize=16, y=1.05)
        ax.tick_params(axis='x', rotation=45, labelsize=16, labelright=False)
        ax.tick_params(axis='y', labelsize=16)
        ax.legend(fontsize=16)       

        if self.save_images:
            plot_loc = os.path.join(EVALUATION_PATH, 'boxplot_words_by_len.jpeg')
            plt.savefig(
                plot_loc, bbox_inches='tight', format='jpeg', dpi=self.DPI)  
            
        return fig
        
