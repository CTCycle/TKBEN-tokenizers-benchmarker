from __future__ import annotations

from TKBEN.server.services.tokenizer_jobs import TokenizerJobService


class DummyJobManager:
    def should_stop(self, job_id: str) -> bool:
        del job_id
        return False

    def update_progress(self, job_id: str, value: float) -> None:
        del job_id, value


def test_run_download_job_delegates_to_service(monkeypatch) -> None:
    class FakeTokenizersService:
        def download_and_persist(self, **kwargs):
            assert kwargs["tokenizers"] == ["bert-base-uncased"]
            return {"status": "success", "downloaded": ["bert-base-uncased"]}

    monkeypatch.setattr("TKBEN.server.services.tokenizer_jobs.TokenizersService", FakeTokenizersService)

    service = TokenizerJobService()
    result = service.run_download_job(
        request_payload={"tokenizers": ["bert-base-uncased"]},
        job_manager=DummyJobManager(),
        job_id="job1",
    )
    assert result["status"] == "success"


def test_upload_and_clear_custom_tokenizers(monkeypatch) -> None:
    state = {"cleared": False}

    class FakeTokenizersService:
        def register_custom_tokenizer_from_upload(self, **kwargs):
            assert kwargs["safe_stem"] == "demo"
            return {
                "status": "success",
                "tokenizer_name": "CUSTOM_demo",
                "is_compatible": True,
            }

        def clear_custom_tokenizers(self):
            state["cleared"] = True

    monkeypatch.setattr("TKBEN.server.services.tokenizer_jobs.TokenizersService", FakeTokenizersService)

    service = TokenizerJobService()
    uploaded = service.upload_custom_tokenizer(b"{}", "demo.json", "demo")
    assert uploaded["is_compatible"] is True

    service.clear_custom_tokenizers()
    assert state["cleared"] is True
