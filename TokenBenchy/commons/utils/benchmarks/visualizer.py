import os
import re
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
        self.save_image(fig, "vocabulary_size.jpeg") if self.save_images else None
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
        self.save_image(fig, "subwords_vs_words.jpeg") if self.save_images else None
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
            fig, axs = plt.subplots(2, 1, figsize=(16, 18), dpi=self.DPI)
            tokens_len_vocab = vocabulary['vocabulary_tokens'].dropna().str.len().to_numpy()
            sns.histplot(data=tokens_len_vocab, ax=axs[0], binwidth=1, edgecolor='black')
            axs[0].set_title(f'Token Lengths from {tokenizer} Vocabulary (Raw Tokens)', fontsize=16)
            axs[0].set_xlabel('Length of Tokens', fontsize=14)
            axs[0].set_ylabel('Frequency', fontsize=14)
            # Histogram of decoded token lengths
            tokens_len_decoded = vocabulary['decoded_tokens'].dropna().str.len().to_numpy()
            sns.histplot(data=tokens_len_decoded, ax=axs[1], binwidth=1, edgecolor='black', color='skyblue')
            axs[1].set_title(f'Token Lengths from Decoding {tokenizer} Tokens', fontsize=16)
            axs[1].set_xlabel('Length of Decoded Tokens', fontsize=14)
            axs[1].set_ylabel('Frequency', fontsize=14)
            plt.tight_layout()
            distributions.append(fig)
            self.save_image(fig, f'{tokenizer}_histogram_tokens.jpeg') if self.save_images else None
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
        fig, ax = plt.subplots(figsize=(16, 9), dpi=self.DPI)
        sns.boxplot(x='tokenizer', y='length', hue='type',  data=df, ax=ax)        
        ax.set_title('Token Length Distribution by Tokenizer and Type', fontsize=16, y=1.02)
        ax.set_xlabel('Tokenizer', fontsize=14)
        ax.set_ylabel('Token Length', fontsize=14)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()

        distributions.append(fig)
        self.save_image(fig, 'boxplot_token_lengths_by_tokenizer.jpeg') if self.save_images else None           
        plt.close(fig)

        return distributions