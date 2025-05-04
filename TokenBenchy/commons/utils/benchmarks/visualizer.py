import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from TokenBenchy.commons.utils.data.database import TOKENDatabase
from TokenBenchy.commons.constants import EVALUATION_PATH
from TokenBenchy.commons.logger import logger


# [TOKENIZERS EXPLORER]
###############################################################################
class VisualizeBenchmarkResults:

    def __init__(self, configuration : dict):         
        self.database = TOKENDatabase(configuration) 
        self.benchmarks, self.vocab_stats = self.database.load_benchmark_results()                       
        self.save_images = configuration.get('save_images', True)
        self.observed_features = [
            'tokens_to_words_ratio', 'AVG_tokens_length', 'bytes_per_token']
        self.configuration = configuration    
        self.DPI = 400   

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
        for tokenizer, grp in self.vocab_stats.groupby('tokenizer'):
            fig, axs = plt.subplots(2, 1, figsize=(16,18), dpi=self.DPI)
            for ax, col in zip(axs, ['number_tokens_from_vocabulary',
                                    'number_tokens_from_decode']):
                vals = grp[col]                
                num_bins = max(int(vals.max() - vals.min() + 1), 1)
                sns.histplot(
                    data=grp, x=col, ax=ax,
                    bins=num_bins,
                    discrete=True,
                    edgecolor='black')

            # Plot vocab tokens count
            sns.histplot(data=grp, x='number_tokens_from_vocabulary',
                ax=axs[0], binwidth=1, edgecolor='black')
            axs[0].set_title(f'Tokens from Vocabulary – {tokenizer}', fontsize=16)
            axs[0].set_ylabel('Frequency', fontsize=14)
            axs[0].set_xlabel('Number of Tokens (vocab)', fontsize=14)

            sns.histplot(data=grp, x='number_tokens_from_decode', ax=axs[1],
                binwidth=1, edgecolor='black')
            axs[1].set_title(f'Tokens from Decode – {tokenizer}', fontsize=16)
            axs[1].set_ylabel('Frequency', fontsize=14)
            axs[1].set_xlabel('Number of Tokens (decode)', fontsize=14)

            plt.tight_layout()
            histograms.append(fig)

            # save if requested
            if self.save_images:
                fname = f"{tokenizer.replace('/', '_')}_token_counts.jpeg"
                out_path = os.path.join(EVALUATION_PATH, fname)
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
        
