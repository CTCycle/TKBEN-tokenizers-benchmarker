from __future__ import annotations

from typing import Any

import pandas as pd

from TKBEN.app.client.workers import (
    ThreadWorker,
    check_thread_status,
    update_progress_callback,
)
from TKBEN.app.utils.logger import logger
from TKBEN.app.utils.repository.serializer import DataSerializer
from TKBEN.app.utils.services.benchmarks import (
    BenchmarkTokenizers,
    VisualizeBenchmarkResults,
)
from TKBEN.app.utils.services.downloads import DatasetManager, TokenizersDownloadManager
from TKBEN.app.utils.services.processing import ProcessDataset


###############################################################################
class DatasetEvents:
    def __init__(
        self, configuration: dict[str, Any], hf_access_token: str | None
    ) -> None:
        self.serializer = DataSerializer()
        self.configuration = configuration
        self.hf_access_token = hf_access_token

    # -------------------------------------------------------------------------
    def load_and_process_dataset(
        self, worker: ThreadWorker | None = None, progress_callback: Any | None = None
    ) -> str | Any | None:
        manager = DatasetManager(self.configuration, self.hf_access_token)
        dataset_name = manager.get_dataset_name()
        logger.info(f"Downloading and saving dataset: {dataset_name}")
        dataset = manager.dataset_download()
        if dataset is None:
            logger.warning(
                "Dataset could not be loaded, try again or change identifier"
            )
            return

        # check thread for interruption
        check_thread_status(worker)
        update_progress_callback(1, 3, progress_callback)

        # process text dataset to remove invalid documents
        processor = ProcessDataset(self.configuration, dataset)
        documents = processor.process_text_dataset()
        n_removed_docs = processor.num_documents - len(documents)
        logger.info(f"Total number of documents: {processor.num_documents}")
        logger.info(
            f"Number of filtered documents: {len(documents)} ({n_removed_docs} removed)"
        )

        # check thread for interruption
        check_thread_status(worker)
        update_progress_callback(2, 3, progress_callback)

        # create dataframe for text dataset
        text_dataset = pd.DataFrame(
            {"dataset_name": [dataset_name] * len(documents), "text": documents}
        )

        # serialize text dataset by saving it into database
        self.serializer.save_text_dataset(text_dataset)

        # check thread for interruption
        update_progress_callback(3, 3, progress_callback)

        return dataset_name


###############################################################################
class BenchmarkEvents:
    def __init__(
        self, configuration: dict[str, Any], hf_access_token: str | None
    ) -> None:
        self.serializer = DataSerializer()
        self.configuration = configuration
        self.hf_access_token = hf_access_token

    # -------------------------------------------------------------------------
    def run_dataset_evaluation_pipeline(
        self, progress_callback: Any | None = None, worker: ThreadWorker | None = None
    ) -> None:
        text_dataset = self.serializer.load_text_dataset()
        benchmarker = BenchmarkTokenizers(self.configuration)
        documents = benchmarker.calculate_text_statistics(
            text_dataset, progress_callback=progress_callback, worker=worker
        )

        # save dataset statistics through upserting into the the text dataset table
        if documents is not None:
            self.serializer.save_dataset_statistics(documents)

    # -------------------------------------------------------------------------
    def get_tokenizer_identifiers(
        self, limit=1000, worker: ThreadWorker | None = None
    ) -> list[Any]:
        downloader = TokenizersDownloadManager(self.configuration, self.hf_access_token)
        identifiers = downloader.get_tokenizer_identifiers(limit=limit, worker=worker)

        return identifiers

    # -------------------------------------------------------------------------
    def execute_benchmarks(
        self, progress_callback: Any | None = None, worker: ThreadWorker | None = None
    ) -> dict[Any, Any]:
        benchmarker = BenchmarkTokenizers(self.configuration)
        downloader = TokenizersDownloadManager(self.configuration, self.hf_access_token)
        text_dataset = self.serializer.load_text_dataset()
        tokenizers = downloader.tokenizer_download(worker=worker)

        if not tokenizers:
            logger.warning(
                "Tokenizers download returned no valid entries; skipping benchmarks"
            )
            return tokenizers

        vocabularies, vocab_stats, benchmarks, NSL_results, global_metrics = (
            benchmarker.run_tokenizer_benchmarks(
                text_dataset,
                tokenizers,
                progress_callback=progress_callback,
                worker=worker,
            )
        )
        # save results into database
        if not benchmarks.empty:
            self.serializer.save_local_metrics(benchmarks)
        else:
            logger.warning("Local benchmark metrics are empty and will not be stored")

        if not vocab_stats.empty:
            self.serializer.save_vocabulary_statistics(vocab_stats)
        else:
            logger.warning("Vocabulary statistics are empty and will not be stored")

        if not global_metrics.empty:
            self.serializer.save_global_metrics(global_metrics)
        else:
            logger.warning("Global benchmark metrics are empty and will not be stored")

        if NSL_results is not None and not NSL_results.empty:
            self.serializer.save_NSL_benchmark(NSL_results)
        elif self.configuration.get("perform_NSL", False):
            logger.warning("NSL results are unavailable and will not be stored")

        for voc in vocabularies:
            if not voc.empty:
                self.serializer.save_vocabulary_tokens(voc)

        return tokenizers


###############################################################################
class VisualizationEnvents:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.serializer = DataSerializer()
        self.img_resolution = 400
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def visualize_benchmark_results(
        self, worker: ThreadWorker | None = None, progress_callback: Any | None = None
    ) -> list[Any]:
        visualizer = VisualizeBenchmarkResults(self.configuration)
        figures = []

        vocab_stats = self.serializer.load_vocabularies()
        benchmark_results = self.serializer.load_local_metrics()
        logger.info(f"Vocabulary data loaded from database: {len(vocab_stats)} records")
        logger.info(
            f"Benchmarks results loaded from database: {len(benchmark_results)} records"
        )

        # 1. generate plot of different vocabulary sizes
        logger.info("Generating boxplots of vocabulary sizes")
        figures.append(visualizer.plot_vocabulary_size(vocab_stats))
        check_thread_status(worker)
        update_progress_callback(1, 3, progress_callback)

        # 2. generate plot of token length distribution
        logger.info("Generating plots of tokens distribution by length")
        figures.extend(visualizer.plot_tokens_length_distribution(vocab_stats))
        check_thread_status(worker)
        update_progress_callback(2, 3, progress_callback)

        # 2. generate plot of words versus subwords
        logger.info("Generating plots to compare subwords to words populations")
        figures.append(visualizer.plot_subwords_vs_words(vocab_stats))
        update_progress_callback(3, 3, progress_callback)

        return figures
