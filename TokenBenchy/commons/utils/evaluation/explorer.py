import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from TokenBenchy.commons.constants import CONFIG, EVALUATION_PATH
from TokenBenchy.commons.logger import logger


# [TOKENIZERS EXPLORER]
###############################################################################
class ExploreTokenizers:

    def __init__(self, tokenizers):
        self.DPI = 600
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
        # Convert dictionaries to a DataFrame for easier plotting
        data = []
        for k, v in self.vocab_len.items():
            data.append(
            {'Tokenizer': k, 'Length': v, 'Type': 'Vocab Length'})
        for k, v in self.vocab_len_decoded.items():
            data.append(
                {'Tokenizer': k, 'Length': v, 'Type': 'Decoded Length'})
        df = pd.DataFrame(data)      
        plt.figure(figsize=(16, 18)) 
        plt.subplot() 
        sns.barplot(
            x='Tokenizer', y='Length', hue='Type', data=df, 
            palette='viridis', edgecolor='black')        
        plt.xlabel('', fontsize=14)
        plt.ylabel('Vocabulary size', fontsize=14)
        plt.title('Vocabulary size by tokenizer', fontsize=14, y=1.05)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()        
        plot_loc = os.path.join(
            EVALUATION_PATH, 'vocabulary_size.jpeg')
        plt.savefig(
            plot_loc, bbox_inches='tight', format='jpeg', dpi=self.DPI)        
        plt.close() 

    #--------------------------------------------------------------------------
    def plot_subwords_vs_words(self):
        word_types_data = []
        # Preparing data for plotting
        for k, v in self.vocabularies.items():
            vocab_words = list(v.keys())
            subwords = [x for x in vocab_words if '##' in x]
            words = [x for x in vocab_words if '##' not in x]
            subwords_perc = len(subwords)/(len(words) + len(subwords)) * 100
            words_perc = len(words)/(len(words) + len(subwords)) * 100
            word_types_data.append({'Vocabulary': k, 'Type': 'Subwords', 'Percentage': subwords_perc})
            word_types_data.append({'Vocabulary': k, 'Type': 'Words', 'Percentage': words_perc})

        # Converting data to DataFrame for easier plotting
        df = pd.DataFrame(word_types_data)
        plt.figure(figsize=(18, 16))       
        sns.barplot(
            data=df, x='Vocabulary', y='Percentage', hue='Type', 
            palette='viridis', edgecolor='black')
        plt.ylabel('Percentage (%)', fontsize=14)
        plt.xlabel('', fontsize=14)
        plt.title('Subwords vs Complete Words', fontsize=14, y=1.05)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout() 
        plot_loc = os.path.join(EVALUATION_PATH, 'subwords_vs_words.jpeg')
        plt.savefig(
            plot_loc, bbox_inches='tight', format='jpeg', dpi=self.DPI)        
        plt.close()       
        
    #--------------------------------------------------------------------------
    def plot_histogram_tokens_length(self):        
        for k, v in self.vocabularies.items():
            k_rep = k.replace('/', '_')
            vocab_words = list(v.keys()) 
            decoded_words = self.vocab_decoded[k]               
            vocab_word_lens = [len(x) for x in vocab_words]
            decoded_word_lens = [len(x) for x in decoded_words]
            fig, axs = plt.subplots(2, 1, figsize=(14, 16), sharex=False)          
            sns.histplot(vocab_word_lens, ax=axs[0], color='skyblue', edgecolor='black', 
                         label='Vocab Words', binwidth=1)
            axs[0].set_title(f'Vocab Words - {k}', fontsize=16)
            axs[0].set_ylabel('Frequency', fontsize=14)
            axs[0].set_xlabel('Word Length', fontsize=14)                         
            sns.histplot(decoded_word_lens, ax=axs[1], color='orange', edgecolor='black', 
                         label='Decoded Words', binwidth=1)
            axs[1].set_title(f'Decoded Words - {k}', fontsize=14)
            axs[1].set_ylabel('Frequency', fontsize=14)
            axs[1].set_xlabel('Word Length', fontsize=14)
            plt.tight_layout()                  
            plot_loc = os.path.join(EVALUATION_PATH, f'{k_rep}_words_by_len.jpeg')
            plt.savefig(
                plot_loc, bbox_inches='tight', format='jpeg', dpi=self.DPI)              
            plt.close()               

    #--------------------------------------------------------------------------
    def plot_boxplot_tokens_length(self):        
        word_lengths = {k : [len(x) for x in list(v.keys())] 
                        for k, v in self.vocabularies.items()}
        word_lengths_decoded = {k : [len(x) for x in v] 
                                for k, v in self.vocab_decoded.items()} 

        # Combine both dictionaries into a single DataFrame
        data = []
        for key in word_lengths.keys():
            for length in word_lengths[key]:
                data.append(
                    {'Tokenizer': key, 'Word Length': length, 'Type': 'Original'})
            for length in word_lengths_decoded.get(key, []):
                data.append(
                    {'Tokenizer': key, 'Word Length': length, 'Type': 'Decoded'})
        df = pd.DataFrame(data)
        plt.figure(figsize=(14, 16))
        sns.boxplot(x='Tokenizer', y='Word Length', hue='Type', data=df)
        plt.xticks(rotation=45, ha='right', va='top', fontsize=12)
        plt.ylabel('Word Length', fontsize=14)
        plt.xlabel('', fontsize=14)
        plt.title('Distribution of Word Lengths by Tokenizer', 
                  fontsize=14, y=1.02)
        plt.yticks(fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()       
        plot_loc = os.path.join(EVALUATION_PATH, 'boxplot_words_by_len.jpeg')
        plt.savefig(
            plot_loc, bbox_inches='tight', format='jpeg', dpi=self.DPI)  
        plt.close()

    #--------------------------------------------------------------------------
    def boxplot_from_benchmarks_dataset(self, dataset):
        observed_features = [
            'Tokens to words ratio', 'AVG tokens length', 'Bytes per token']    
        plt.figure(figsize=(14, 16))
        for y in observed_features:
            sns.boxplot(x=dataset['Tokenizer'], y=dataset[y], data=dataset)            
            plt.xticks(rotation=45, ha='right', fontsize=14)
            plt.yticks(fontsize=14)          
            plt.xlabel('', fontsize=14)
            plt.ylabel(y, fontsize=14)
            plt.title(f'Boxplot of {y}', fontsize=14, y=1.02) 
            plt.legend(fontsize=14)               
            plt.tight_layout()               
            plot_loc = os.path.join(EVALUATION_PATH, f'boxplot_{y}.jpeg')                   
            plt.savefig(
            plot_loc, bbox_inches='tight', format='jpeg', dpi=self.DPI)  
            plt.close()

        
    