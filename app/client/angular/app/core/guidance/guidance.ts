// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

export type TourPlacement = 'top' | 'right' | 'bottom' | 'left';

export interface TourStep {
  readonly target: string;
  readonly title: string;
  readonly body: string;
  readonly placement?: TourPlacement;
}

export interface Tour {
  readonly id: string;
  readonly label: string;
  readonly route: string;
  readonly steps: readonly TourStep[];
}

export interface Tip {
  readonly title: string;
  readonly body: string;
}

export const TOURS: readonly Tour[] = [
  {
    id: 'datasets',
    label: 'Datasets',
    route: '/dataset',
    steps: [
      {
        target: '[data-tour="dataset-filters"]',
        title: '1. Find or add a dataset',
        body: 'Search and filter the catalog, or use + to download a public dataset or upload your own CSV/Excel file.',
        placement: 'bottom',
      },
      {
        target: '[data-tour="dataset-preview"]',
        title: '2. Pick a dataset',
        body: 'Select a dataset here, then run a validation session to compute quality and lexical metrics for it.',
        placement: 'left',
      },
      {
        target: '[data-tour="dataset-dashboard"]',
        title: '3. Read the dashboard',
        body: 'Validation results are saved and shown here as statistics, histograms, and word charts. Export them when you are done.',
        placement: 'top',
      },
    ],
  },
  {
    id: 'tokenizers',
    label: 'Tokenizers',
    route: '/tokenizers',
    steps: [
      {
        target: '[data-tour="tokenizer-filters"]',
        title: '1. Find or add a tokenizer',
        body: 'Search the catalog, or use + to add Hugging Face tokenizers or upload a custom tokenizer JSON.',
        placement: 'bottom',
      },
      {
        target: '[data-tour="tokenizer-preview"]',
        title: '2. Select a tokenizer',
        body: 'Choose a tokenizer to open its saved report or generate a new one.',
        placement: 'left',
      },
      {
        target: '[data-tour="tokenizer-dashboard"]',
        title: '3. Review the report',
        body: 'Vocabulary statistics and charts for the selected tokenizer appear here.',
        placement: 'top',
      },
      {
        target: '[data-tour="tokenizer-playground"]',
        title: '4. Try the Playground',
        body: 'Paste a short text and compare how up to three tokenizers split it. It is a quick preview, not a benchmark.',
        placement: 'top',
      },
    ],
  },
  {
    id: 'cross-benchmark',
    label: 'Cross Benchmark',
    route: '/cross-benchmark',
    steps: [
      {
        target: '[data-tour="benchmark-report"]',
        title: '1. Choose a report',
        body: 'Open Reports to load a past benchmark or start a new run with a dataset and several tokenizers.',
        placement: 'bottom',
      },
      {
        target: '[data-tour="benchmark-actions"]',
        title: '2. Customize and reuse',
        body: 'Arrange the dashboard, clone the current configuration for a new run, or export the report.',
        placement: 'bottom',
      },
      {
        target: '[data-tour="benchmark-dashboard"]',
        title: '3. Compare the metrics',
        body: 'Each widget compares tokenizers on one metric. Drag widgets to reorder them and pick a baseline to see relative differences.',
        placement: 'top',
      },
    ],
  },
];

export const TIPS: readonly Tip[] = [
  {
    title: 'Track long tasks',
    body: 'Downloads, validations, and benchmarks run in the background. The Jobs button in the header shows progress and lets you cancel.',
  },
  {
    title: 'Hugging Face access',
    body: 'Gated or private datasets and tokenizers need an access key. Add it in Settings → Keys.',
  },
  {
    title: 'Validate before analyzing',
    body: 'Dataset charts appear after a validation session has run. Use the sampling options for a quick first pass.',
  },
  {
    title: 'Baselines are per report',
    body: 'Choose the baseline tokenizer inside a report; comparisons update for that report only.',
  },
  {
    title: 'Playground is a preview',
    body: 'Token counts in the Playground come from a single pass over your text. Use Cross Benchmark for real measurements.',
  },
];
