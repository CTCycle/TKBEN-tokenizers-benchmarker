#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs::{self, File};
use std::net::{TcpListener, TcpStream};
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::{Manager, RunEvent};

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Clone)]
struct BackendState(Arc<Mutex<Option<Child>>>);

fn background(command: &mut Command) -> &mut Command {
    #[cfg(target_os = "windows")]
    command.creation_flags(CREATE_NO_WINDOW);
    command
}

fn escape_js(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\'', "\\'").replace('\n', "\\n").replace('\r', "")
}

fn render(app: &tauri::AppHandle, title: &str, message: &str, error: bool) {
    let color = if error { "#f87171" } else { "#38bdf8" };
    let script = format!(
        "document.body.style='margin:0;background:#0f172a;color:#e2e8f0;font-family:Segoe UI,sans-serif';document.body.innerHTML=\"<main style='min-height:100vh;display:grid;place-items:center;padding:32px;box-sizing:border-box'><section style='max-width:720px;border:1px solid #334155;border-radius:18px;padding:32px;background:#111c30'><h1 style='color:{color}'>{}</h1><pre style='white-space:pre-wrap;line-height:1.55'>{}</pre></section></main>\";",
        escape_js(title), escape_js(message)
    );
    if let Some(window) = app.get_webview_window("main") { let _ = window.eval(&script); }
}

fn copy_if_missing(source: &Path, target: &Path) -> Result<(), String> {
    if target.exists() { return Ok(()); }
    if let Some(parent) = target.parent() { fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
    fs::copy(source, target).map_err(|e| format!("Cannot seed {}: {e}", target.display()))?;
    Ok(())
}

fn free_port() -> Result<u16, String> {
    TcpListener::bind(("127.0.0.1", 0)).and_then(|s| s.local_addr()).map(|a| a.port()).map_err(|e| format!("Cannot allocate a local port: {e}"))
}

fn stop_backend(state: &BackendState) {
    if let Ok(mut guard) = state.0.lock() {
        if let Some(child) = guard.as_mut() {
            #[cfg(target_os = "windows")]
            {
                let mut taskkill = Command::new("taskkill");
                let _ = background(&mut taskkill).args(["/PID", &child.id().to_string(), "/T", "/F"]).stdout(Stdio::null()).stderr(Stdio::null()).status();
            }
            #[cfg(not(target_os = "windows"))]
            { let _ = child.kill(); }
            let _ = child.wait();
        }
        *guard = None;
    }
}

fn launch_backend(app: &tauri::AppHandle, state: &BackendState) -> Result<(), String> {
    let resource = app.path().resource_dir().map_err(|e| format!("Cannot locate packaged resources: {e}"))?;
    let python = resource.join("runtimes").join("python").join("python.exe");
    let app_dir = resource.join("app");
    let server_dir = app_dir.join("server");
    let templates = resource.join("settings");
    for required in [&python, &server_dir.join("app.py"), &app_dir.join("client").join("dist").join("index.html"), &templates.join(".env.example"), &templates.join("configurations.json")] {
        if !required.is_file() { return Err(format!("The desktop package is incomplete. Missing:\n{}", required.display())); }
    }

    let local = app.path().app_local_data_dir().map_err(|e| format!("Cannot locate local application data: {e}"))?;
    let data = local.join("data");
    let logs = local.join("logs");
    let config = local.join("config");
    let cache = local.join("cache");
    for path in [&data, &logs, &config, &cache] { fs::create_dir_all(path).map_err(|e| format!("Cannot create {}: {e}", path.display()))?; }
    copy_if_missing(&templates.join(".env.example"), &config.join(".env"))?;
    copy_if_missing(&templates.join("configurations.json"), &config.join("configurations.json"))?;

    let port = free_port()?;
    let stdout = File::create(logs.join("desktop-backend.log")).map_err(|e| format!("Cannot create backend log: {e}"))?;
    let stderr = File::create(logs.join("desktop-backend.err.log")).map_err(|e| format!("Cannot create backend error log: {e}"))?;
    let bootstrap = format!(
        "import sys;sys.path.insert(0, r'''{}''');from uvicorn.main import main;main()",
        app_dir.display()
    );
    let mut command = Command::new(&python);
    background(&mut command)
        .args(["-c", &bootstrap, "server.app:app", "--host", "127.0.0.1", "--port", &port.to_string(), "--log-level", "info"])
        .current_dir(&server_dir)
        .env("TKBEN_TAURI_MODE", "true")
        .env("TKBEN_DATA_DIR", &data)
        .env("TKBEN_LOG_DIR", &logs)
        .env("TKBEN_CONFIG_DIR", &config)
        .env("HF_HOME", cache.join("huggingface"))
        .env("MPLCONFIGDIR", cache.join("matplotlib"))
        .stdin(Stdio::null()).stdout(stdout).stderr(stderr);
    let child = command.spawn().map_err(|e| format!("Cannot start the bundled backend: {e}\nLogs: {}", logs.display()))?;
    *state.0.lock().map_err(|_| "Cannot lock backend state".to_string())? = Some(child);

    for _ in 0..120 {
        if TcpStream::connect_timeout(&format!("127.0.0.1:{port}").parse().unwrap(), Duration::from_millis(300)).is_ok() {
            if let Some(window) = app.get_webview_window("main") { let _ = window.eval(&format!("window.location.replace('http://127.0.0.1:{port}/');")); }
            return Ok(());
        }
        if let Ok(mut guard) = state.0.lock() {
            if let Some(child) = guard.as_mut() { if let Ok(Some(status)) = child.try_wait() { return Err(format!("The bundled backend exited with {status}.\nLogs: {}", logs.display())); } }
        }
        thread::sleep(Duration::from_millis(500));
    }
    Err(format!("Timed out waiting for the bundled backend.\nLogs: {}", logs.display()))
}

fn main() {
    let state = BackendState(Arc::new(Mutex::new(None)));
    let cleanup_state = state.clone();
    let app = tauri::Builder::default().manage(state.clone()).setup(move |app| {
        render(app.handle(), "Starting TKBEN Desktop", "Preparing the bundled local service...", false);
        let handle = app.handle().clone();
        let worker_state = state.clone();
        thread::spawn(move || if let Err(error) = launch_backend(&handle, &worker_state) { stop_backend(&worker_state); render(&handle, "TKBEN Desktop startup error", &error, true); });
        Ok(())
    }).build(tauri::generate_context!()).expect("failed to build Tauri application");
    app.run(move |_handle, event| if matches!(event, RunEvent::Exit | RunEvent::ExitRequested { .. }) { stop_backend(&cleanup_state); });
}
