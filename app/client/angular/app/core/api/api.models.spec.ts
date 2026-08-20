import { describe, expect, it } from 'vitest';
import type { BenchmarkRunResponse } from './api.models';

const schema3Report5WithUnavailableMetrics: BenchmarkRunResponse = {
  status: 'success',
  schema_version: 3,
  methodology_version: 'semantic_honesty',
  report_id: 7,
  report_version: 5,
  created_at: null,
  run_name: null,
  selected_metric_keys: [],
  dataset_name: 'custom/example',
  documents_processed: 1,
  tokenizers_processed: ['custom/tokenizer'],
  tokenizers_count: 1,
  config: {
    max_documents: 1,
    warmup_trials: 0,
    timed_trials: 1,
    batch_size: 1,
    seed: 42,
    parallelism: 1,
    include_lm_metrics: false,
  },
  hardware_profile: {
    runtime: '',
    os: '',
    cpu_model: null,
    cpu_logical_cores: null,
    memory_total_mb: null,
  },
  trial_summary: { warmup_trials: 0, timed_trials: 1 },
  tokenizer_results: [{
    tokenizer: 'custom/tokenizer',
    status: 'success',
    tokenizer_family: 'unknown',
    runtime_backend: 'test',
    vocabulary_size: 0,
    added_tokens: 0,
    special_token_share: 0,
    efficiency: {
      encode_tokens_per_second_mean: null,
      encode_tokens_per_second_ci95_low: null,
      encode_tokens_per_second_ci95_high: null,
      encode_chars_per_second_mean: null,
      encode_bytes_per_second_mean: null,
      encode_only_wall_time_seconds: null,
      dataset_stream_wall_time_seconds: null,
      postprocess_wall_time_seconds: null,
      end_to_end_wall_time_seconds: null,
      load_time_seconds: null,
    },
    latency: {
      encode_latency_p50_ms: null,
      encode_latency_p95_ms: null,
      encode_latency_p99_ms: null,
      sample_count: null,
    },
    fidelity: {
      exact_round_trip_rate: null,
      normalized_round_trip_rate: null,
      unknown_token_rate: null,
      byte_fallback_rate: null,
      lossless_encodability_rate: null,
    },
    fragmentation: {
      tokens_per_character: null,
      characters_per_token: null,
      tokens_per_byte: null,
      bytes_per_token: null,
      pieces_per_word_mean: null,
      fragmentation_by_word_length_bucket: [],
    },
    resources: { peak_rss_mb: null, memory_delta_mb: null },
  }],
  dashboard: {
    widgets: [],
    available_widget_ids: [],
    available_metric_keys: [],
    unavailable_selected_metric_keys: [],
  },
  per_document_stats: [{
    tokenizer: 'custom/tokenizer',
    tokens_count: [],
    pieces_per_word: [null],
    bytes_per_token: [],
    encode_latency_ms: [null],
    peak_rss_mb: [null],
  }],
  runtime_metadata: {},
  raw_observations: {},
};

describe('benchmark API contracts', () => {
  it('preserves unavailable metric values as null and keeps default arrays present', () => {
    const result = schema3Report5WithUnavailableMetrics.tokenizer_results[0];

    expect(result.efficiency.encode_tokens_per_second_mean).toBeNull();
    expect(result.latency.sample_count).toBeNull();
    expect(result.fidelity.exact_round_trip_rate).toBeNull();
    expect(result.fragmentation.fragmentation_by_word_length_bucket).toEqual([]);
    expect(result.resources.peak_rss_mb).toBeNull();
    expect(schema3Report5WithUnavailableMetrics.per_document_stats[0].tokens_count).toEqual([]);
    expect(schema3Report5WithUnavailableMetrics.per_document_stats[0].pieces_per_word).toEqual([null]);
  });
});
