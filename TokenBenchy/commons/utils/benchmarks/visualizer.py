import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import seaborn as sns

from TokenBenchy.commons.utils.data.database import TokenBenchyDatabase
from TokenBenchy.commons.constants import EVALUATION_PATH
from TokenBenchy.commons.logger import logger


# [TOKENIZERS EXPLORER]
###############################################################################
class VisualizeBenchmarkResults:

    def __init__(self, configuration : dict):         
        self.database = TokenBenchyDatabase(configuration)                               
        self.save_images = configuration.get('save_images', True)
        self.observed_features = [
            'tokens_to_words_ratio', 'AVG_tokens_length', 'bytes_per_token']
        self.configuration = configuration    
        self.DPI = 400 

        self.benchmarks, self.vocab_stats = self.database.load_benchmark_results()
        self.tokenizers = self.vocab_stats['tokenizer'].to_list()  

    #--------------------------------------------------------------------------
    def plot_vocabulary_size(self):                        
        df = (self.vocab_stats.melt(id_vars="tokenizer",
              value_vars=["number_tokens_from_vocabulary",
                         "number_tokens_from_decode"],
                    var_name="Type",
                    value_name="Count")         
                .assign(Type=lambda d: d["Type"].map({
                        "number_tokens_from_vocabulary": "Vocabulary",
                        "number_tokens_from_decode": "Decoded"})))

        fig, ax = plt.subplots(figsize=(18, 6), dpi=self.DPI)
        sns.barplot(
            x="tokenizer", y="Count", hue="Type", data=df,
            palette="viridis", edgecolor="black", ax=ax)
        ax.set_xlabel("", fontsize=16)
        ax.set_ylabel("Number of tokens", fontsize=16)
        ax.set_title("Vocabulary size by tokenizer", fontsize=18, y=1.02)
        ax.tick_params(axis="x", rotation=45, labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.legend(title="", fontsize=14)
        plt.tight_layout()
        if self.save_images:
            plot_loc = os.path.join(EVALUATION_PATH, "vocabulary_size.jpeg")
            plt.savefig(plot_loc, bbox_inches="tight", dpi=self.DPI)

        return fig
           
    #--------------------------------------------------------------------------
    def plot_subwords_vs_words(self):
        for name in self.tokenizers:
            vocabulary = self.database.load_vocabulary_tokens()
            df = (self.vocab_stats.melt(id_vars="tokenizer", value_vars=[
                            "percentage_subwords",
                            "percentage_true_words"],
                        var_name="Type",
                        value_name="Percentage")
                    .assign(Type=lambda d: d["Type"].map({
                            "percentage_subwords": "Subwords",
                            "percentage_true_words": "True words"})))

        fig, ax = plt.subplots(figsize=(18, 6), dpi=self.DPI)
        sns.barplot(
            x="tokenizer", y="Percentage", hue="Type", data=df,
            palette="viridis", edgecolor="black", ax=ax)
        ax.set_xlabel("", fontsize=16)
        ax.set_ylabel("Percentage (%)", fontsize=16)
        ax.set_title("Subwords vs complete words", fontsize=18, y=1.02)
        ax.tick_params(axis="x", rotation=45, labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.legend(title="", fontsize=14)

        plt.tight_layout()
        if self.save_images:
            plot_loc = os.path.join(EVALUATION_PATH, "subwords_vs_words.jpeg")
            plt.savefig(plot_loc, bbox_inches="tight", dpi=self.DPI)

        return fig
          
    #--------------------------------------------------------------------------
    def plot_histogram_tokens_length(self):
        histograms = []
        for tokenizer, grp in self.benchmarks.groupby('tokenizer'):           
            vocab_counts = grp['number_tokens_from_vocabulary']
            decode_counts = grp['number_tokens_from_decode']         

            fig, axs = plt.subplots(2, 1, figsize=(16, 18), dpi=self.DPI)

            # Vocabulary tokens histogram
            if not vocab_counts.empty:
                sns.histplot(
                    data=vocab_counts,
                    ax=axs[0],
                    binwidth=1,
                    edgecolor='black')
                
            axs[0].set_title(f'Tokens from Vocabulary – {tokenizer}', fontsize=16)
            axs[0].set_ylabel('Frequency', fontsize=14)
            axs[0].set_xlabel('Number of Tokens (vocab)', fontsize=14)

            # Decode tokens histogram
            if not decode_counts.empty:
                sns.histplot(
                    data=decode_counts,
                    ax=axs[1],
                    binwidth=1,
                    edgecolor='black')
                
            axs[1].set_title(f'Tokens from Decode {tokenizer}', fontsize=16)
            axs[1].set_ylabel('Frequency', fontsize=14)
            axs[1].set_xlabel('Number of Tokens (decode)', fontsize=14)

            plt.tight_layout()
            histograms.append(fig)
            
            if self.save_images:
                out_path = os.path.join(EVALUATION_PATH, f'{tokenizer}_histogram_tokens.jpeg')
                fig.savefig(out_path, bbox_inches='tight', dpi=self.DPI)

       

        return histograms               

    #--------------------------------------------------------------------------
    def plot_boxplot_tokens_length(self):        
        df = self.vocab_stats.melt(id_vars=['tokenizer'], value_vars=[
                'number_tokens_from_vocabulary',
                'number_tokens_from_decode'],
            var_name='Type',
            value_name='Token Count')
   
        df['Type'] = df['Type'].map({
            'number_tokens_from_vocabulary': 'Vocabulary',
            'number_tokens_from_decode': 'Decoded'})

        fig, ax = plt.subplots(figsize=(16, 18), dpi=self.DPI)
        sns.boxplot(x='tokenizer', y='Token Count', hue='Type', data=df, ax=ax)
    
        ax.set_xlabel('', fontsize=16)
        ax.set_ylabel('Number of Tokens', fontsize=16)
        ax.set_title('Distribution of Token Counts by Tokenizer', fontsize=16, y=1.02)
        ax.tick_params(axis='x', rotation=45, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(title='', fontsize=14)
        plt.tight_layout()

        # save if requested
        if self.save_images:
            out_path = os.path.join(EVALUATION_PATH, 'boxplot_token_counts.jpeg')
            fig.savefig(out_path, bbox_inches='tight', dpi=self.DPI)

        return fig
        



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
            {'tokenizer': k, 'Length': v, 'Type': 'Vocab Length'})
        for k, v in self.vocab_len_decoded.items():
            data.append(
                {'tokenizer': k, 'Length': v, 'Type': 'Decoded Length'})
        df = pd.DataFrame(data)      
        plt.figure(figsize=(16, 18)) 
        plt.subplot() 
        sns.barplot(
            x='tokenizer', y='Length', hue='Type', data=df, 
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
                    {'tokenizer': key, 'Word Length': length, 'Type': 'Original'})
            for length in word_lengths_decoded.get(key, []):
                data.append(
                    {'tokenizer': key, 'Word Length': length, 'Type': 'Decoded'})
        df = pd.DataFrame(data)
        plt.figure(figsize=(14, 16))
        sns.boxplot(x='tokenizer', y='Word Length', hue='Type', data=df)
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
            sns.boxplot(x=dataset['tokenizer'], y=dataset[y], data=dataset)            
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

        
    