from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from server.app import app

###############################################################################
def _key_item(key_id: int = 1) -> dict[str, object]:
    return {
        "id": key_id,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "is_active": False,
        "masked_preview": "********",
    }

###############################################################################
def test_key_routes_keep_raw_values_out_of_list_and_map_lifecycle_calls(
    monkeypatch,
) -> None:
    from server.api import keys as keys_api

    calls: list[tuple[str, object]] = []

    class FakeKeyService:
        def add_key(self, raw_key: str):
            calls.append(("add", raw_key))
            return _key_item()

        def list_keys(self):
            return [_key_item()]

        def set_active_key(self, key_id: int) -> None:
            calls.append(("activate", key_id))

        def clear_active_key(self, key_id: int) -> None:
            calls.append(("deactivate", key_id))

        def delete_key(self, key_id: int, confirm: bool) -> None:
            calls.append(("delete", (key_id, confirm)))

    monkeypatch.setattr(keys_api, "HFAccessKeyService", FakeKeyService)
    client = TestClient(app)

    created = client.post("/api/keys", json={"key_value": "hf_example"})
    listed = client.get("/api/keys")
    activated = client.post("/api/keys/1/activate")
    deactivated = client.post("/api/keys/1/deactivate")
    deleted = client.delete("/api/keys/1?confirm=true")

    assert created.status_code == 201
    assert listed.status_code == 200
    assert "key_value" not in created.json()
    assert "key_value" not in listed.json()["keys"][0]
    assert activated.json()["status"] == "success"
    assert deactivated.json()["message"] == "Active key cleared."
    assert deleted.json()["status"] == "success"
    assert calls == [
        ("add", "hf_example"),
        ("activate", 1),
        ("deactivate", 1),
        ("delete", (1, True)),
    ]

###############################################################################
def test_key_reveal_is_blocked_when_server_policy_disables_reveal(monkeypatch) -> None:
    from server.api import keys as keys_api

    monkeypatch.setattr(keys_api, "is_key_reveal_enabled", lambda: False)

    response = TestClient(app).post("/api/keys/1/reveal")

    assert response.status_code == 403
    assert response.json()["detail"] == "Key reveal is disabled by server policy."
