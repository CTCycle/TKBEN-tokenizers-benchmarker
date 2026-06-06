# Benchmark Contract
Last updated: 2026-06-06

## Benchmark Request Notes
Benchmark run request config includes tokenizer behavior flags and per-document controls:
- `add_special_tokens`
- `padding`
- `truncation`
- `max_length`
- `store_per_document_stats`
- `per_document_sample_size`

## Result and Runtime Metadata
Each tokenizer result includes status and optional error details for failure isolation:
- `status`
- `error_type`
- `error_message`

Runtime metadata includes benchmark config echo and dataset scope details:
- `dataset_total_documents_available`
- `dataset_documents_benchmarked`
- `benchmark_config`
- `metric_availability` indicates whether metric families are measured or available for the run payload
- unavailable benchmark metrics use `null` rather than synthetic zero values

## Fidelity Semantics
- `fid.exact_round_trip_rate` stores decode/re-encode token ID stability, not direct text preservation
- `fid.normalized_round_trip_rate` stores NFC-normalized text round-trip success
- `fid.lossless_encodability_rate` is currently a heuristic vocabulary character overlap percentage
- byte-fallback is not measured by the current engine and is marked unavailable

## Fragmentation Semantics
- fragmentation metrics are computed with `add_special_tokens=False`, `padding=False`, and `truncation=False`
- `fragmentation_by_word_length_bucket` is derived from actual regex-tokenized words bucketed into `short_1_4`, `medium_5_8`, and `long_9_plus`

## Efficiency Fields
Benchmark efficiency payload includes boundary-separated timing fields:
- `encode_only_wall_time_seconds`
- `dataset_stream_wall_time_seconds`
- `postprocess_wall_time_seconds`

## Reporting Rule
Chart aggregations are derived from successful tokenizer results only (`status="success"`), so failed tokenizers do not appear as misleading zero-value bars.

## Latency Notes
- latency percentiles use all timed observations, normalized to per-document batch latency
- latency payloads include `sample_count` for both summary metrics and distribution points
