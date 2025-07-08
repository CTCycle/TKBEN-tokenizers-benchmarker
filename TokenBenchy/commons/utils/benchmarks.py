import os
import re
import numpy as np
import pandas as pd
from transformers.utils.logging import set_verbosity_error
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
from seaborn import barplot, boxplot, histplot

from TokenBenchy.commons.utils.database import TokenBenchyDatabase
from TokenBenchy.commons.interface.workers import check_thread_status, update_progress_callback
from TokenBenchy.commons.constants import EVALUATION_PATH
from TokenBenchy.commons.logger import logger

             
# [TOKENIZERS EXPLORER]
###############################################################################
class BenchmarkTokenizers:

    def __init__(self, database : TokenBenchyDatabase, configuration : dict):
        set_verbosity_error()        
        self.max_docs_number = configuration.get('num_documents', 0)
        self.reduce_data_size = configuration.get("reduce_output_size", False)
        self.include_custom_tokenizer = configuration.get("include_custom_tokenizer", False)
        self.include_NSL = configuration.get("include_NSL", False)        
        self.vocab_columns = ['id', 'vocabulary_tokens', 'decoded_tokens'] 
        self.database = database
        self.configuration = configuration       

    #--------------------------------------------------------------------------
    def calculate_dataset_statistics(self, documents, **kwargs):
        max_documents = min(self.max_docs_number, len(documents))
        documents = documents[:max_documents] if max_documents > 0 else documents
        dataset_stats = pd.DataFrame(columns=['text'], data=documents)

        # 1/3 - Word count
        dataset_stats['words_count'] = dataset_stats['text'].apply(
            lambda doc: len(doc.split()))
        
        check_thread_status(kwargs.get('worker', None))
        update_progress_callback(0, 2, kwargs.get('progress_callback', None))

        # 2/3 - Average word length
        dataset_stats['AVG word length'] = dataset_stats['text'].apply(
            lambda doc: np.mean([len(w) for w in doc.split()]))
        
        check_thread_status(kwargs.get('worker', None))
        update_progress_callback(1, 2, kwargs.get('progress_callback', None))

        # 3/3 - Standard deviation of word length
        dataset_stats['STD word length'] = dataset_stats['text'].apply(
            lambda doc: np.std([len(w) for w in doc.split()]))
        
        check_thread_status(kwargs.get('worker', None))
        update_progress_callback(2, 2, kwargs.get('progress_callback', None))    
        
        self.database.save_dataset_statistics(dataset_stats)

    #--------------------------------------------------------------------------
    def calculate_vocabulary_statistics(self, tokenizers, **kwargs):
        rows = []        
        for i, (name, tokenizer) in enumerate(tokenizers.items()):            
            vocab = tokenizer.get_vocab()              
            vocab_words = list(vocab.keys())
            vocab_indices = list(vocab.values())
            subwords = [x for x in vocab_words if '##' in x]
            true_words = [x for x in vocab_words if x not in subwords]
            
            decoded_words = tokenizer.decode(vocab_indices).split()            
            shared = set(vocab_words).intersection(decoded_words)
            unshared = set(vocab_words).symmetric_difference(decoded_words)            
           
            subwords_perc = len(subwords)/(len(true_words) + len(subwords)) * 100
            words_perc = len(true_words)/(len(true_words) + len(subwords)) * 100           
            
            rows.append({
                'tokenizer': name,
                'number_tokens_from_vocabulary': len(vocab_words),
                'number_tokens_from_decode': len(decoded_words),
                'number_shared_tokens': len(shared),
                'number_unshared_tokens': len(unshared),
                'percentage_subwords': subwords_perc,
                'percentage_true_words' : words_perc})            

            # save entire vocabulary into database
            vocabulary = pd.DataFrame({
                self.vocab_columns[0]: pd.Series(vocab_indices),
                self.vocab_columns[1]: pd.Series(vocab_words),
                self.vocab_columns[2]: pd.Series(decoded_words)})
            self.database.save_vocabulary_tokens(vocabulary, name)
            # check for worker thread status and update progress callback
            check_thread_status(kwargs.get('worker', None))            

        # save vocabulary statistics into database 
        vocabulary_stats = pd.DataFrame(rows)
        self.database.save_vocabulary_results(vocabulary_stats)           
        
        return vocabulary, vocabulary_stats
    
    #--------------------------------------------------------------------------
    def run_tokenizer_benchmarks(self, documents, tokenizers : dict, **kwargs):        
        # calculate basic dataset statistics and extract vocabulary for each tokenizer
        vocab, vocab_stats = self.calculate_vocabulary_statistics(tokenizers)
        # filter only a fraction of documents if requested by the user
        if self.max_docs_number is not None and self.max_docs_number <= len(documents):
            documents = documents[:self.max_docs_number]      
        
        all_tokenizers = []
        for i, (name, tokenizer) in enumerate(tokenizers.items()):
            k = name.replace('/', '_')
            logger.info(f'Decoding documents with {name}')
            # initialize dataframe with tokenizers name and extracted dataset document
            data = pd.DataFrame({'tokenizer': name, 'text': documents})            
            
            # calculate basic statistics such as:
            # 1. number of characters in text
            # 2. number of words in text
            # 3. average word length in characters            
            data['num_characters'] = data['text'].apply(lambda x : len(str(x)))       
            data['words_count'] = data['text'].apply(lambda x : len(x.split()))
            data['AVG_words_length'] = data['text'].apply(
                lambda text: np.mean([len(word) for word in text.split()]) if text else 0)            
            # If this is a custom tokenizer and the user decided to include it:
            # 1. encode raw text into token IDs
            # 2. decode those IDs back into a single string to ensure getting
            # the the tokenizer’s canonical spacing/pieces
            if 'CUSTOM' in name and self.include_custom_tokenizer:
                data['tokens'] = data['text'].apply(
                    lambda text: tokenizer.decode(tokenizer.encode(text).ids))
                data['tokens split'] = data['tokens'].apply(
                    lambda tok: tok.split() if isinstance(tok, str) else [])
            # If this is not a custom tokenizer or the user decided to avoid including those:
            # 1. encode raw text into token IDs
            # 2. decode those IDs back into a single string to ensure getting
            # the the tokenizer’s canonical spacing/pieces
            else:
                data['tokens split'] = data['text'].apply(tokenizer.tokenize)
                data['tokens'] = data['tokens split'].apply(
                    lambda toks: ' '.join(toks) if isinstance(toks, (list, tuple)) else '')
                
            # count number of tokens from text encoding
            data['tokens_count'] = data['tokens split'].apply(
                lambda toks: len(toks) if isinstance(toks, (list, tuple)) else 0)
            # count number of characters in all tokens from text encoding
            data['tokens_characters'] = data['tokens'].apply(
                lambda s: len(s) if isinstance(s, str) else 0)
            # calculate the average token length
            data['AVG_tokens_length'] = data['tokens split'].apply(
                lambda tokens: np.mean([len(tok) for tok in tokens]) if tokens else 0)
            # calculate the ratio between number of tokens and number of original words
            data['tokens_to_words_ratio'] = np.where(
                data['words_count'] > 0, data['tokens_count'] / data['words_count'], 0)
            # calculate varchar bytes occupied by each token
            data['bytes_per_token'] = np.where(
                data['tokens_count'] > 0, data['num_characters'] / data['tokens_count'], 0)
            
            drop_cols = ['tokens split']
            if self.reduce_data_size:
                drop_cols.extend(['text', 'tokens'])
            data = data.drop(columns=drop_cols)           
            
            self.database.save_benchmark_results(data, table_name=k)            
            all_tokenizers.append(data)

            # check for worker thread status and update progress callback          
            check_thread_status(kwargs.get('worker', None))
            update_progress_callback(
                i, len(tokenizers.items()), kwargs.get('progress_callback', None))         

        benchmark_results = pd.concat(all_tokenizers, ignore_index=True)
        self.database.save_benchmark_results(benchmark_results)

        if self.include_NSL and self.include_custom_tokenizer:
            NSL_results = self.normalized_sequence_length(benchmark_results)

        return benchmark_results

    #--------------------------------------------------------------------------
    def normalized_sequence_length(self, benchmark_results):                            
        data_custom = benchmark_results[
            benchmark_results['tokenizer'].str.contains(
                'custom tokenizer', case=False, na=False)]   

        data = []
        names = list(benchmark_results['tokenizer'].unique())
        if data_custom.empty:
            logger.warning('NSL value cannot be calculated without a custom tokenizer as reference')
            return None
        else:
            for tok in tqdm(names):
                logger.info(f'NSL value is calculated for {tok} versus custom tokenizers')
                data_chunk = benchmark_results[benchmark_results['tokenizer'] == tok]                                                 
                data_chunk['NSL'] = [
                    x/y if y != 0 else 0 for x, y in zip(
                    data_custom['tokens_count'].to_list(),
                    data_chunk['tokens_count'].to_list())]            
                data.append(data_chunk)
            
            data_NSL = pd.concat(data, ignore_index=True)
            self.database.save_benchmark_results(data_NSL, table_name='NSL')          

        return data_NSL 


# [TOKENIZERS EXPLORER]
###############################################################################
class VisualizeBenchmarkResults:

    def __init__(self, database, configuration : dict):        
        self.database = database
        self.configuration = configuration    
        self.DPI = configuration.get('image_resolution', 400)

        self.observed_features = [
            'tokens_to_words_ratio', 'AVG_tokens_length', 'bytes_per_token']

        self.benchmarks, self.vocab_stats = self.database.load_benchmark_results()
        self.tokenizers = self.vocab_stats['tokenizer'].to_list() 

    #--------------------------------------------------------------------------
    def save_image(self, fig, name):
        name = re.sub(r'[^0-9A-Za-z_]', '_', name)
        out_path = os.path.join(EVALUATION_PATH, name)
        fig.savefig(out_path, bbox_inches='tight', dpi=self.DPI)   

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

        fig, ax = plt.subplots(figsize=(24, 22), dpi=self.DPI)
        barplot(x="tokenizer", y="Count", hue="Type", data=df,
                palette="viridis", edgecolor="black", ax=ax)
               
        ax.set_xlabel("", fontsize=20, fontweight="bold")
        ax.set_ylabel("Number of tokens", fontsize=20, fontweight="bold")
        ax.set_title("Vocabulary size by tokenizer", fontsize=24, fontweight="bold", y=1.02)
        ax.tick_params(axis="x", rotation=45, labelsize=18)
        ax.tick_params(axis="y", labelsize=18)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        
        # Legend: bold and larger
        legend = ax.legend(title="", fontsize=18)
        for text in legend.get_texts():
            text.set_fontweight("bold")
        
        plt.tight_layout()                
        self.save_image(fig, "vocabulary_size.jpeg") 
        plt.close()          

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

        fig, ax = plt.subplots(figsize=(24, 22), dpi=self.DPI)
        barplot(x="tokenizer", y="Percentage", hue="Type", data=df,
                palette="viridis", edgecolor="black", ax=ax)
        # Axis labels and title
        ax.set_xlabel("", fontsize=20, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=20, fontweight="bold")
        ax.set_title("Subwords vs complete words", fontsize=24, fontweight="bold", y=1.02)        
        ax.tick_params(axis="x", rotation=45, labelsize=18)
        ax.tick_params(axis="y", labelsize=18)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")        
        legend = ax.legend(title="", fontsize=18)
        for text in legend.get_texts():
            text.set_fontweight("bold")
        plt.tight_layout()
        self.save_image(fig, "subwords_vs_words.jpeg") 
        plt.close()          

        return fig
            
    #--------------------------------------------------------------------------
    def plot_tokens_length_distribution(self):
        distributions = []   
        records = []
        # Loop through each tokenizer to plot histograms
        for tokenizer, _ in self.vocab_stats.groupby('tokenizer'):
            vocabulary = self.database.load_vocabulary_tokens(tokenizer)
            # Histogram of vocabulary token lengths
            fig, axs = plt.subplots(2, 1, figsize=(24, 22), dpi=self.DPI)
            tokens_len_vocab = vocabulary['vocabulary_tokens'].dropna().str.len().to_numpy()
            histplot(data=tokens_len_vocab, ax=axs[0], binwidth=1, edgecolor='black')
            axs[0].set_title(f'Token Lengths from {tokenizer} Vocabulary (Raw Tokens)', fontsize=20, fontweight="bold")
            axs[0].set_xlabel('Length of Tokens', fontsize=18, fontweight="bold")
            axs[0].set_ylabel('Frequency', fontsize=18, fontweight="bold")
            axs[0].tick_params(axis='x', labelsize=16)
            axs[0].tick_params(axis='y', labelsize=16)
            for label in axs[0].get_xticklabels() + axs[0].get_yticklabels():
                label.set_fontweight("bold")

            # Histogram of decoded token lengths
            tokens_len_decoded = vocabulary['decoded_tokens'].dropna().str.len().to_numpy()
            histplot(data=tokens_len_decoded, ax=axs[1], binwidth=1, edgecolor='black', color='skyblue')
            axs[1].set_title(f'Token Lengths from Decoding {tokenizer} Tokens', fontsize=20, fontweight="bold")
            axs[1].set_xlabel('Length of Decoded Tokens', fontsize=18, fontweight="bold")
            axs[1].set_ylabel('Frequency', fontsize=18, fontweight="bold")
            axs[1].tick_params(axis='x', labelsize=16)
            axs[1].tick_params(axis='y', labelsize=16)
            for label in axs[1].get_xticklabels() + axs[1].get_yticklabels():
                label.set_fontweight("bold")

            plt.tight_layout()
            distributions.append(fig)
            self.save_image(fig, f'{tokenizer}_histogram_tokens.jpeg') 
            plt.close(fig)

            # Accumulate data for boxplot
            records += [
                {'tokenizer': tokenizer, 'type': 'vocabulary', 'length': length}
                for length in tokens_len_vocab]
            records += [
                {'tokenizer': tokenizer, 'type': 'decoded', 'length': length}
                for length in tokens_len_decoded]

        # Create combined boxplot across tokenizers
        df = pd.DataFrame(records)
        fig, ax = plt.subplots(figsize=(24, 22), dpi=self.DPI)
        boxplot(x='tokenizer', y='length', hue='type',  data=df, ax=ax)        
        ax.set_title('Token Length Distribution by Tokenizer and Type', fontsize=22, fontweight="bold", y=1.02)
        ax.set_xlabel('Tokenizer', fontsize=18, fontweight="bold")
        ax.set_ylabel('Token Length', fontsize=18, fontweight="bold")
        ax.tick_params(axis='x', rotation=45, labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        legend = plt.legend(fontsize=16)
        for text in legend.get_texts():
            text.set_fontweight("bold")
        plt.tight_layout()

        distributions.append(fig)
        self.save_image(fig, 'boxplot_token_lengths_by_tokenizer.jpeg')   
        plt.close(fig)

        return distributions
    
    


    
            
                    
        
      


