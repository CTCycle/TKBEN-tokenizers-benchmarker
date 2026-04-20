from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


###############################################################################
class BenchmarkPlottingMixin:
    def generate_plots(
        self,
        vocabularies: list[pd.DataFrame]        
    ) -> list[dict[str, str]]:
        plots: list[dict[str, str]] = []

        # Plot vocabulary size
        if vocabularies:
            combined_vocab = pd.concat(vocabularies, ignore_index=True)
            if not combined_vocab.empty:
                fig = self.plot_vocabulary_size(combined_vocab)
                if fig:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                    buf.seek(0)
                    plots.append(
                        {
                            "name": "vocabulary_size",
                            "data": base64.b64encode(buf.read()).decode("utf-8"),
                        }
                    )
                    plt.close(fig)

        # Plot subwords vs words
        if vocabularies:
            combined_vocab = pd.concat(vocabularies, ignore_index=True)
            if not combined_vocab.empty:
                fig = self.plot_subwords_vs_words(combined_vocab)
                if fig:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                    buf.seek(0)
                    plots.append(
                        {
                            "name": "subwords_vs_words",
                            "data": base64.b64encode(buf.read()).decode("utf-8"),
                        }
                    )
                    plt.close(fig)

        return plots

    # -------------------------------------------------------------------------
    def plot_vocabulary_size(self, data: pd.DataFrame) -> Figure | None:
        df = data.dropna(subset=["vocabulary_tokens"])
        df["vocabulary_tokens"] = df["vocabulary_tokens"].astype(str)
        df = df[df["vocabulary_tokens"].str.len() > 0]

        if df.empty:
            return None

        counts = (
            df.groupby("tokenizer", sort=False)["vocabulary_tokens"]
            .nunique()
            .sort_values(ascending=True)
        )

        n_tok = len(counts)
        width_in = 14
        height_in = max(4, 0.7 * n_tok + 2)

        fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=400)

        y_pos = np.arange(n_tok, dtype=float)
        widths = counts.to_numpy(dtype=float)

        ax.barh(y_pos, widths, edgecolor="black")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(counts.index))

        ax.set_title(
            "Vocabulary size by tokenizer",
            fontsize=16,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Number of tokens", fontsize=13, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        fig.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    def plot_subwords_vs_words(self, data: pd.DataFrame) -> Figure | None:
        df = data.loc[:, ["tokenizer", "vocabulary_tokens"]].copy()
        df = df.dropna(subset=["tokenizer", "vocabulary_tokens"])
        if df.empty:
            return None

        df["tokenizer"] = df["tokenizer"].astype(str)
        df["vocabulary_tokens"] = df["vocabulary_tokens"].astype(str).str.strip()
        df = df[df["vocabulary_tokens"].str.len() > 0]
        if df.empty:
            return None

        special_pat = r"^(?:\[.*\]|<.*>|\{.*\}|</?s>|</?pad>|UNK|PAD)$"
        is_special = df["vocabulary_tokens"].str.match(special_pat, case=False)
        df = df[~is_special]
        if df.empty:
            return None

        bert_sub = df["vocabulary_tokens"].str.startswith("##")
        sp_has = df["vocabulary_tokens"].str.contains("▁", regex=False)
        sp_word_start = df["vocabulary_tokens"].str.startswith("▁")
        sp_sub = sp_has & (~sp_word_start)
        bbpe_has = df["vocabulary_tokens"].str.contains("Ġ", regex=False)
        bbpe_word_start = df["vocabulary_tokens"].str.startswith("Ġ")
        bbpe_sub = bbpe_has & (~bbpe_word_start)

        is_subword = bert_sub | sp_sub | bbpe_sub
        is_word = ~is_subword

        df["is_subword"] = is_subword.astype(int)
        df["is_word"] = is_word.astype(int)

        grouped = (
            df.groupby("tokenizer", sort=False)
            .agg(subwords_count=("is_subword", "sum"), words_count=("is_word", "sum"))
            .reset_index()
        )

        grouped = grouped.sort_values("subwords_count", ascending=False).reset_index(
            drop=True
        )

        if grouped.empty:
            return None

        tokenizers = grouped["tokenizer"].tolist()
        n = len(tokenizers)
        x = np.arange(n, dtype=float)
        width = 0.4

        subwords = grouped["subwords_count"].to_numpy(dtype=float)
        words = grouped["words_count"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 7), dpi=400)
        ax.bar(
            x - width / 2,
            subwords,
            width=width,
            edgecolor="black",
            color="#105D8D",
            label="Subwords (vocabulary)",
        )
        ax.bar(
            x + width / 2,
            words,
            width=width,
            edgecolor="black",
            color="#107F40",
            label="Words (vocabulary)",
        )

        ax.set_title(
            "Subwords vs Words by tokenizer (vocabulary)",
            fontsize=16,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Tokenizer", fontsize=13, fontweight="bold")
        ax.set_ylabel("Count", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tokenizers, rotation=35, ha="right")

        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(title="", fontsize=11)

        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            try:
                lbl.set_fontweight("bold")
            except Exception:
                pass

        fig.tight_layout()
        return fig
