from __future__ import annotations

from TKBEN.server.common.utils.variables import EnvironmentVariables
import TKBEN.server.common.utils.variables as variables_module


###############################################################################
def test_runtime_env_values_are_loaded_from_dotenv_file(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "FASTAPI_HOST=0.0.0.0",
                "FASTAPI_PORT=5050",
                "UI_HOST=127.0.0.1",
                "UI_PORT=9000",
                "KERAS_BACKEND=tensorflow",
                "MPLBACKEND=Agg",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(variables_module, "ENV_FILE_PATH", str(env_file))
    for key in (
        "FASTAPI_HOST",
        "FASTAPI_PORT",
        "UI_HOST",
        "UI_PORT",
        "KERAS_BACKEND",
        "MPLBACKEND",
    ):
        monkeypatch.delenv(key, raising=False)

    env_variables = EnvironmentVariables()

    assert env_variables.get("FASTAPI_HOST") == "0.0.0.0"
    assert env_variables.get("FASTAPI_PORT") == "5050"
    assert env_variables.get("UI_HOST") == "127.0.0.1"
    assert env_variables.get("UI_PORT") == "9000"
    assert env_variables.get("KERAS_BACKEND") == "tensorflow"
    assert env_variables.get("MPLBACKEND") == "Agg"


###############################################################################
def test_runtime_env_get_falls_back_to_default_when_missing(
    tmp_path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")

    monkeypatch.setattr(variables_module, "ENV_FILE_PATH", str(env_file))
    monkeypatch.delenv("FASTAPI_HOST", raising=False)

    env_variables = EnvironmentVariables()

    assert env_variables.get("FASTAPI_HOST", "127.0.0.1") == "127.0.0.1"
