# TokenExplorer: Exploring Tokenizers and Their Characteristics

## 1. Project Overview
Tokenizers play a pivotal role in the preprocessing phase of text data, transforming raw text into a structured format that models can understand. The effectiveness of this step significantly impacts the overall performance of NLP models, making the choice of tokenizer a crucial decision in the development of language-based applications. However, with the plethora of tokenizers available, each with its unique approach and capabilities, selecting the most suitable one can be a daunting task. TokenExplorer aims to offer a comprehensive toolkit for analyzing and comparing the performance and characteristics of open source tokenizers (currently based on English language), through a simple yet effective jupyter notebook. This facilitates the exploration of tokenizer characteristics such as tokenization speed, token granularity, handling of special characters, language support, and adaptability to domain-specific vocabularies. Users can perform detailed comparisons between tokenizers, assessing their suitability for specific applications, such as text classification, sentiment analysis, language translation, or semantic search.

## 2. Installation 
The installation process is designed for simplicity, using .bat scripts to automatically create a virtual environment with all necessary dependencies. Please ensure that Anaconda or Miniconda is properly installed on your system before proceeding.

- To set up the environment, run `scripts/environment_setup.bat`. This file offers a convenient one-click solution to set up your virtual environment.
- **IMPORTANT:** if the path to the project folder is changed for any reason after installation, the app will cease to work. Run `scripts/package_setup.bat` or alternatively use `pip install -e . --use-pep517` from cmd when in the project folder (upon activating the conda environment).

## 3. How to use
Within the main project folder (TokenExplorer) you will find other folders, each designated to specific tasks. Run `tokenizers_benchmark.py` to run all the benchmarks using the selected dataset as references, and the list of downloaded tokenizers from HuggingFace (as well as the custom tokenizers is included). Otherwise, run `tokenizer_exploration.ipynb` to generate the plots of the saved benchmarks in a jupyter notebook. This notebook will work only if the benchmarks have been run beforehand, and the .csv files with the results are present in the correct folder

### 3.1 Resources
This folder is used to hold tokenizers and datasets, as well as to store the results of vaiorus benchmarks. Here are the key subfolders:

**benchmarks:** contains the results of the tokenizers benchmarks that have been ran. Within this folder, one can find `figures` and `results`; while the former will contain the generate plots, the latter is where the benchmark .csv files will be located.  

**datasets:** contains the downloaded datasets that are used to test the tokenizers performance. While the datasets that are automatically downloaded are saved in `datasets`, the custom dataset are saved in `custom dataset`.

**tokenizers:** contains the downloaded tokenizers that are used to run the benchmarks on the target dataset. While the tokenizers from HuggingFace are automatically saved in `tokenizers`, the custom tokenizers must be located in `custom tokenizers`.

## 4. Configurations
For customization, you can modify the main configuration parameters using `settings/configurations.json` 

#### General configuration

| Parameter          | Description                                                    |
|--------------------|----------------------------------------------------------------|
| ACCESS_TOKEN       | The personal access token from HuggingFace, required for the   |
|                    | download of certain tokenizers                                 |
| TOKENIZERS         | List of tokenizers to download                                 |
| DATASET            | Target dataset to benchmark tokenizers (it is needed to        |
|                    | provide the config and corpus references)                      |

#### Benchmark configuration

| Parameter                | Description                                              |
|--------------------------|----------------------------------------------------------|
| MAX_NUM_DOCS             | Maximum number of documents to use from the dataset      |
| REDUCE_CSV_SIZE          | List of tokenizers to download                           |
| INCLUDE_CUSTOM_DATASET   | Whether or not to include the custom dataset             |
| INCLUDE_CUSTOM_TOKENIZER | Whether or not to include the custom tokenizers          |

## 5. License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.

