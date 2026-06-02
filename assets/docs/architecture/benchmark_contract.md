# Benchmark Contract
Last updated: 2026-06-02

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

## Efficiency Fields
Benchmark efficiency payload includes boundary-separated timing fields:
- `encode_only_wall_time_seconds`
- `dataset_stream_wall_time_seconds`
- `postprocess_wall_time_seconds`

## Reporting Rule
Chart aggregations are derived from successful tokenizer results only (`status="success"`), so failed tokenizers do not appear as misleading zero-value bars.
