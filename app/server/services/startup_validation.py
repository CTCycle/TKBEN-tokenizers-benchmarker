from __future__ import annotations

from server.configurations import ServerSettings


###############################################################################
def ensure_runtime_directories(settings: ServerSettings) -> None:
    for directory in (
        settings.paths.logs,
        settings.paths.datasets,
        settings.paths.tokenizers,
        settings.paths.templates,
    ):
        directory.mkdir(parents=True, exist_ok=True)


###############################################################################
def build_cors_origins(settings: ServerSettings) -> list[str]:
    ui_host = _normalized_host(settings.network.ui_host)
    ui_port = settings.network.ui_port

    hosts = {ui_host}
    if ui_host == "127.0.0.1":
        hosts.add("localhost")
    elif ui_host == "localhost":
        hosts.add("127.0.0.1")

    return sorted(f"http://{host}:{ui_port}" for host in hosts)


###############################################################################
def run_startup_validations(settings: ServerSettings) -> None:
    ensure_runtime_directories(settings)


###############################################################################
def _normalized_host(raw_host: str) -> str:
    host = raw_host.strip() or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host
