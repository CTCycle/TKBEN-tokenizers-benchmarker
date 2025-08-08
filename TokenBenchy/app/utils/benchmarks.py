import os
import re

import numpy as np
import pandas as pd
from transformers.utils.logging import set_verbosity_error
import matplotlib.pyplot as plt
from seaborn import barplot, boxplot, histplot
from tqdm import tqdm

from TokenBenchy.app.utils.data.serializer import DataSerializer
from TokenBenchy.app.interface.workers import check_thread_status, update_progress_callback
from TokenBenchy.app.constants import EVALUATION_PATH
from TokenBenchy.app.logger import logger

             
# [TOKENIZERS EXPLORER]
###############################################################################
class BenchmarkTokenizers:

    def __init__(self, configuration : dict):
        set_verbosity_error()        
        self.max_docs_number = configuration.get('num_documents', 0)
        self.reduce_data_size = configuration.get("reduce_output_size", False)
        self.include_custom_tokenizer = configuration.get("include_custom_tokenizer", False)
        self.include_NSL = configuration.get("include_NSL", False)  
        self.configuration = configuration 

    #--------------------------------------------------------------------------
    def process_tokens(text, tokenizer):
        ids = tokenizer.encode(text).ids
        decoded = tokenizer.decode(ids)
        toks = decoded.split()
        return decoded, toks      

    #--------------------------------------------------------------------------
    def calculate_text_statistics(self, **kwargs):
        # load previously saved text dataset from database using the serializer
        serializer = DataSerializer(self.configuration)        
        documents = serializer.load_text_dataset()
        # interrupt the operation if no text dataset is available
        if documents.empty:
            logger.info('No text dataset available for statistics calculation')
            return 

        dataset_name = documents["dataset_name"].iloc[0]
        logger.info(f'Loaded dataset {dataset_name} with {documents.shape[0]} records')

        max_documents = min(self.max_docs_number, len(documents))
        documents = documents[:max_documents] if max_documents > 0 else documents

        # 1/3 - Word count
        logger.info('Calculating word count for each document')
        documents['words_count'] = documents['text'].apply(
            lambda doc: len(doc.split()))
        
        check_thread_status(kwargs.get('worker', None))
        update_progress_callback(1, 4, kwargs.get('progress_callback', None))

        # 2/3 - Average word length
        logger.info('Calculating average word length for each document')
        documents['AVG_words_length'] = documents['text'].apply(
            lambda doc: np.mean([len(w) for w in doc.split()]))
        
        check_thread_status(kwargs.get('worker', None))
        update_progress_callback(2, 4, kwargs.get('progress_callback', None))

        # 3/3 - Standard deviation of word length
        logger.info('Calculating standard deviation of words length for each document')
        documents['STD_words_length'] = documents['text'].apply(
            lambda doc: np.std([len(w) for w in doc.split()]))
        
        check_thread_status(kwargs.get('worker', None))
        update_progress_callback(3, 4, kwargs.get('progress_callback', None))    
        
        # save dataset statistics through upserting into the the text dataset table
        serializer.save_dataset_statistics(documents)

        update_progress_callback(4, 4, kwargs.get('progress_callback', None))    

    #--------------------------------------------------------------------------
    def calculate_vocabulary_statistics(self, tokenizers, **kwargs):
        vocabulary_stats = [] 
        serializer = DataSerializer(self.configuration) 
        # Iterate over each selected tokenizer from the combobox      
        for i, (name, tokenizer) in enumerate(tokenizers.items()):            
            vocab = tokenizer.get_vocab()              
            vocab_words = list(vocab.keys())
            vocab_indices = list(vocab.values())
            # Identify subwords (words containing '##', typical for BERT-like tokenizers)
            subwords = [x for x in vocab_words if '##' in x]
            # Identify "true words" as elements that are not subwords
            true_words = [x for x in vocab_words if x not in subwords]
            # Decode the indices back to words
            decoded_words = tokenizer.decode(vocab_indices).split() 
            # Identify tokens that are present in both the vocabulary and the decoded output           
            shared = set(vocab_words).intersection(decoded_words)
            # Identify tokens that are in one but not both (symmetric difference)
            unshared = set(vocab_words).symmetric_difference(decoded_words)            
            # Calculate percentage of subwords and true words in the vocabulary
            subwords_perc = len(subwords)/(len(true_words) + len(subwords)) * 100
            words_perc = len(true_words)/(len(true_words) + len(subwords)) * 100           
            # Collect statistics for the current tokenizer
            vocabulary_stats.append({
                'tokenizer': name,
                'number_tokens_from_vocabulary': len(vocab_words),
                'number_tokens_from_decode': len(decoded_words),
                'number_shared_tokens': len(shared),
                'number_unshared_tokens': len(unshared),
                'percentage_subwords': subwords_perc,
                'percentage_true_words' : words_perc})

            vocabulary = pd.DataFrame({
            'tokenizer': [name] * len(vocab_words),
            'token_id': vocab_indices,
            'vocabulary_tokens': vocab_words,
            'decoded_tokens': [tokenizer.decode([idx]) for idx in vocab_indices]})
            
            # check for worker thread status and update progress callback
            check_thread_status(kwargs.get('worker', None))
        
            serializer.save_vocabulary_tokens(vocabulary)
        
        # save vocabulary statistics into database 
        vocabulary_stats = pd.DataFrame(vocabulary_stats)
        serializer.save_vocabulary_statistics(vocabulary_stats)           
        
        return vocabulary_stats
    
    #--------------------------------------------------------------------------
    def run_tokenizer_benchmarks(self, tokenizers : dict, **kwargs):

        """
        Run benchmarking for each tokenizer over a set of text documents.
        Optimized for speed and robustness, using vectorized operations where possible.
        Maintains original comments and logic.

        Args:
            tokenizers (dict): Dictionary of tokenizer name -> tokenizer instance.
            **kwargs: Optional arguments such as worker and progress_callback.
        Returns:
            pd.DataFrame: Concatenated benchmark results for all tokenizers.

        """
        # Load previously saved text dataset from database using the serializer
        serializer = DataSerializer(self.configuration)
        documents = serializer.load_text_dataset()

        # Calculate basic dataset statistics and extract vocabulary for each tokenizer
        _ = self.calculate_vocabulary_statistics(tokenizers, worker=kwargs.get('worker', None))

        # Filter only a fraction of documents if requested by the user
        if 0 < self.max_docs_number <= len(documents):
            documents = documents.iloc[:self.max_docs_number].reset_index(drop=True)

        # Prepare texts series only once for efficiency
        texts = documents['text'].astype(str)
        num_docs = len(texts)
        all_tokenizers = []

        for i, (name, tokenizer) in enumerate(tokenizers.items()):
            k = name.replace('/', '_')
            logger.info(f'Decoding documents with {name}')

            # Initialize dataframe with tokenizer's name and dataset documents
            data = pd.DataFrame({'tokenizer': [name] * num_docs, 'text': texts})

            # Calculate basic statistics using vectorized operations
            data['num_characters'] = texts.str.len()
            # Split words only once, keep as list for later calculations
            data['words_split'] = texts.str.split()
            data['words_count'] = data['words_split'].apply(len)
            data['AVG_words_length'] = data['words_split'].apply(
                lambda ws: np.mean([len(w) for w in ws]) if ws else 0)
            
            if 'CUSTOM' in name and self.include_custom_tokenizer:
                decoded_and_toks = data['text'].apply(lambda text: pd.Series(self.process_tokens(text, tokenizer)))
                data['tokens'] = decoded_and_toks[0]
                data['tokens_split'] = decoded_and_toks[1]
            else:
                data['tokens_split'] = data['text'].apply(tokenizer.tokenize)
                data['tokens'] = data['tokens_split'].apply(
                    lambda toks: ' '.join(toks) if isinstance(toks, (list, tuple)) else '')

            # Calculate number of tokens, token characters, average token length, ratios
            data['tokens_count'] = data['tokens_split'].apply(
                lambda toks: len(toks) if isinstance(toks, (list, tuple)) else 0)
            data['tokens_characters'] = data['tokens'].str.len()
            data['AVG_tokens_length'] = data['tokens_split'].apply(
                lambda toks: np.mean([len(tok) for tok in toks]) if toks else 0)
            # Use np.where for ratio calculations to avoid division by zero
            data['tokens_to_words_ratio'] = np.where(
                data['words_count'] > 0, data['tokens_count'] / data['words_count'], 0)
            data['bytes_per_token'] = np.where(
                data['tokens_count'] > 0, data['num_characters'] / data['tokens_count'], 0)

            # Drop intermediate columns if reduce_data_size is set
            drop_cols = ['tokens', 'tokens_split', 'words_split']
            data.drop(columns=drop_cols, inplace=True)

            # Save results for this tokenizer and append
            serializer.save_benchmark_results(data)
            all_tokenizers.append(data)

            # Progress update and thread safety
            check_thread_status(kwargs.get('worker', None))
            update_progress_callback(
                i+1, len(tokenizers), kwargs.get('progress_callback', None))

        # Concatenate all tokenizer benchmark results
        benchmark_results = pd.concat(all_tokenizers, ignore_index=True)
        serializer.save_benchmark_results(benchmark_results)

        # Calculate NSL if required
        if self.include_NSL and self.include_custom_tokenizer:
            self.calculate_normalized_sequence_length(benchmark_results)

        return benchmark_results

    #--------------------------------------------------------------------------
    def calculate_normalized_sequence_length(self, benchmark_results): 
        serializer = DataSerializer(self.configuration)                           
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
            serializer.save_NSL_benchmark(data_NSL)          

        return data_NSL 


# [TOKENIZERS EXPLORER]
###############################################################################
class VisualizeBenchmarkResults:

    def __init__(self, configuration : dict):  
        self.configuration = configuration    
        self.DPI = configuration.get('image_resolution', 400)
        self.observed_features = [
            'tokens_to_words_ratio', 'AVG_tokens_length', 'bytes_per_token']
        
        self.serializer = DataSerializer(self.configuration)   
        _, self.vocab_stats = self.serializer.load_benchmark_results()
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
            vocabulary = self.serializer.load_vocabularies()
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
        vocabularies = self.serializer.load_vocabularies()
        tokenizers = self.vocab_stats['tokenizer'].to_list()
        # Loop through each tokenizer to plot histograms
        for tk in tokenizers:
            tokenizer_data = vocabularies[vocabularies['tokenizer'] == tk]
            
            # Histogram of vocabulary token lengths
            fig, axs = plt.subplots(2, 1, figsize=(24, 22), dpi=self.DPI)
            tokens_len_vocab = vocabularies['vocabulary_tokens'].dropna().str.len().to_numpy()
            histplot(data=tokens_len_vocab, ax=axs[0], binwidth=1, edgecolor='black')
            axs[0].set_title(f'Token Lengths from {tk} Vocabulary (Raw Tokens)', fontsize=20, fontweight="bold")
            axs[0].set_xlabel('Length of Tokens', fontsize=18, fontweight="bold")
            axs[0].set_ylabel('Frequency', fontsize=18, fontweight="bold")
            axs[0].tick_params(axis='x', labelsize=16)
            axs[0].tick_params(axis='y', labelsize=16)
            for label in axs[0].get_xticklabels() + axs[0].get_yticklabels():
                label.set_fontweight("bold")

            # Histogram of decoded token lengths
            tokens_len_decoded = tokenizer_data['decoded_tokens'].dropna().str.len().to_numpy()
            histplot(data=tokens_len_decoded, ax=axs[1], binwidth=1, edgecolor='black', color='skyblue')
            axs[1].set_title(f'Token Lengths from Decoding {tk} Tokens', fontsize=20, fontweight="bold")
            axs[1].set_xlabel('Length of Decoded Tokens', fontsize=18, fontweight="bold")
            axs[1].set_ylabel('Frequency', fontsize=18, fontweight="bold")
            axs[1].tick_params(axis='x', labelsize=16)
            axs[1].tick_params(axis='y', labelsize=16)
            for label in axs[1].get_xticklabels() + axs[1].get_yticklabels():
                label.set_fontweight("bold")

            plt.tight_layout()
            distributions.append(fig)
            self.save_image(fig, f'{tk}_histogram_tokens.jpeg') 
            plt.close(fig)

            # Accumulate data for boxplot
            records += [
                {'tokenizer': tk, 'type': 'vocabulary', 'length': length}
                for length in tokens_len_vocab]
            records += [
                {'tokenizer': tk, 'type': 'decoded', 'length': length}
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
    
    


    
            
                    
        
      


