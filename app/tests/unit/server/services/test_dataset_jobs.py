from __future__ import annotations

from server.services.dataset_jobs import DatasetJobService

###############################################################################
def test_build_analysis_payload_preserves_contract() -> None:
    service = DatasetJobService()
    payload = service.build_analysis_payload(
        {
            "dataset_name": "custom/demo",
            "report_id": 1,
            "report_version": 2,
            "document_length_histogram": {"bins": ["1-2"], "counts": [1]},
            "word_length_histogram": {"bins": ["1-2"], "counts": [1]},
        }
    )
    assert payload["status"] == "success"
    assert payload["dataset_name"] == "custom/demo"
    assert payload["report_id"] == 1

###############################################################################
def test_extract_configuration_handles_missing() -> None:
    service = DatasetJobService()
    assert service.extract_configuration({}) is None
    assert (
        service.extract_configuration({"configs": {"configuration": " abc "}}) == "abc"
    )
