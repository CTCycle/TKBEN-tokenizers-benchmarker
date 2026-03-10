@echo off
setlocal enabledelayedexpansion

set "script_dir=%~dp0"
for %%I in ("%script_dir%..\..") do set "repo_root=%%~fI"
set "actual_repo_root=%repo_root%"
set "actual_project_folder=%actual_repo_root%\TKBEN\"
set "actual_client_dir=%actual_project_folder%client"
set "actual_runtime_python_exe=%actual_project_folder%resources\runtimes\python\python.exe"
set "actual_runtime_uv_exe=%actual_project_folder%resources\runtimes\uv\uv.exe"
set "actual_runtime_node_dir=%actual_project_folder%resources\runtimes\nodejs"
set "actual_node_cmd=%actual_runtime_node_dir%\node.exe"
set "actual_npm_cmd=%actual_runtime_node_dir%\npm.cmd"
set "short_drive=Q:"

if exist %short_drive%\ (
  echo [FATAL] Drive letter %short_drive% is already in use. Free it before packaging.
  goto build_error
)
subst %short_drive% "%actual_repo_root%" >nul
if errorlevel 1 (
  echo [FATAL] Failed to create temporary drive mapping %short_drive% for "%actual_repo_root%".
  goto build_error
)

set "repo_root=%short_drive%\"
set "project_folder=%repo_root%TKBEN\"
set "client_dir=%project_folder%client"
set "tauri_dir=%client_dir%\src-tauri"
set "bundle_source_dir=%tauri_dir%\r"
set "bundle_dir=%tauri_dir%\target\release\bundle"
set "release_export_dir=%actual_repo_root%\release\windows"

echo [TAURI] Release build helper

echo [CHECK] Validating bundled runtimes...
call :require_file "%actual_runtime_python_exe%" "embedded Python runtime" || goto build_error
call :require_file "%actual_runtime_uv_exe%" "embedded uv runtime" || goto build_error
call :require_file "%actual_node_cmd%" "embedded Node.js runtime" || goto build_error
call :require_file "%actual_npm_cmd%" "embedded npm runtime" || goto build_error

echo [CHECK] Preparing short Tauri bundle sources...
call :prepare_bundle_sources || goto build_error

echo [CHECK] Resolving Cargo...
set "cargo_cmd="
if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "cargo_cmd=%USERPROFILE%\.cargo\bin\cargo.exe"
if not defined cargo_cmd (
  cargo --version >nul 2>&1
  if not errorlevel 1 set "cargo_cmd=cargo"
)
if not defined cargo_cmd (
  echo [FATAL] Rust/Cargo not found. Install Rust first: https://rustup.rs/
  goto build_error
)
for /f "delims=" %%V in ('"%cargo_cmd%" --version 2^>nul') do set "cargo_version=%%V"
echo [INFO] Cargo command: %cargo_cmd%
if defined cargo_version echo [INFO] !cargo_version!
if /I not "%cargo_cmd%"=="cargo" (
  for %%I in ("%cargo_cmd%") do set "PATH=%%~dpI;%PATH%"
)
set "CARGO=%cargo_cmd%"

if /I not "%actual_node_cmd%"=="node" (
  for %%I in ("%actual_node_cmd%") do set "PATH=%%~dpI;%PATH%"
)

for /f "delims=" %%V in ('"%actual_node_cmd%" --version 2^>nul') do set "node_version=%%V"
for /f "delims=" %%V in ('"%actual_npm_cmd%" --version 2^>nul') do set "npm_version=%%V"

echo [INFO] npm command: %actual_npm_cmd%
echo [INFO] node command: %actual_node_cmd%
if defined node_version echo [INFO] Node.js version: !node_version!
if defined npm_version echo [INFO] npm version: !npm_version!

if not exist "%actual_client_dir%\package.json" (
  echo [FATAL] Missing client package.json at "%actual_client_dir%"
  goto build_error
)

set "RUST_BACKTRACE=1"
set "CARGO_TERM_PROGRESS_WHEN=auto"

echo [STEP 1/3] Installing frontend dependencies
pushd "%actual_client_dir%" >nul
if exist "package-lock.json" (
  echo [CMD] "%actual_npm_cmd%" ci --foreground-scripts
  call "%actual_npm_cmd%" ci --foreground-scripts
) else (
  echo [CMD] "%actual_npm_cmd%" install --foreground-scripts
  call "%actual_npm_cmd%" install --foreground-scripts
)
if errorlevel 1 (
  popd >nul
  echo [FATAL] npm dependency installation failed.
  goto build_error
)

echo [STEP 2/3] Building frontend distribution
echo [CMD] "%actual_npm_cmd%" run build
call "%actual_npm_cmd%" run build
if errorlevel 1 (
  popd >nul
  echo [FATAL] Frontend build failed.
  goto build_error
)
popd >nul

echo [STEP 3/3] Building Tauri application
pushd "%client_dir%" >nul
echo [CMD] "%actual_npm_cmd%" run tauri:build:release
call "%actual_npm_cmd%" run tauri:build:release
if errorlevel 1 (
  popd >nul
  echo [FATAL] Tauri build failed.
  goto build_error
)
popd >nul

call :cleanup_bundle_sources

echo [OK] Build completed successfully.
if exist "%release_export_dir%" (
  echo [INFO] User-facing release artifacts:
  echo        %release_export_dir%
) else if exist "%bundle_dir%" (
  echo [INFO] Release artifacts:
  echo        %bundle_dir%
) else (
  echo [WARN] Build finished but release directories were not found.
  echo        %release_export_dir%
  echo        %bundle_dir%
)

subst %short_drive% /D >nul 2>&1
endlocal & exit /b 0

:require_file
if exist "%~1" (
  echo [OK] %~2 found: %~1
  exit /b 0
)
echo [FATAL] Missing %~2 at "%~1"
echo         Run TKBEN\start_on_windows.bat first to install the portable runtimes.
exit /b 1

:prepare_bundle_sources
call :cleanup_bundle_sources

md "%bundle_source_dir%" >nul 2>&1
if errorlevel 1 (
  echo [FATAL] Failed to create bundle source directory "%bundle_source_dir%".
  exit /b 1
)

copy /y "%repo_root%pyproject.toml" "%bundle_source_dir%\p.toml" >nul
if errorlevel 1 (
  echo [FATAL] Failed to stage pyproject.toml for Tauri bundling.
  exit /b 1
)
copy /y "%repo_root%uv.lock" "%bundle_source_dir%\u.lock" >nul
if errorlevel 1 (
  echo [FATAL] Failed to stage uv.lock for Tauri bundling.
  exit /b 1
)

if exist "%project_folder%resources\database.db" (
  copy /y "%project_folder%resources\database.db" "%bundle_source_dir%\db" >nul
  if errorlevel 1 (
    echo [FATAL] Failed to stage database.db for Tauri bundling.
    exit /b 1
  )
) else (
  type nul > "%bundle_source_dir%\db"
)

call :make_junction "%bundle_source_dir%\srv" "%project_folder%server" || exit /b 1
call :make_junction "%bundle_source_dir%\sc" "%project_folder%scripts" || exit /b 1
call :make_junction "%bundle_source_dir%\set" "%project_folder%settings" || exit /b 1
call :make_junction "%bundle_source_dir%\d" "%project_folder%client\dist" || exit /b 1
call :make_junction "%bundle_source_dir%\tpl" "%project_folder%resources\templates" || exit /b 1
call :make_junction "%bundle_source_dir%\src" "%project_folder%resources\sources" || exit /b 1
call :make_junction "%bundle_source_dir%\py" "%project_folder%resources\runtimes\python" || exit /b 1
call :make_junction "%bundle_source_dir%\uv" "%project_folder%resources\runtimes\uv" || exit /b 1
exit /b 0

:make_junction
cmd /c mklink /J "%~1" "%~2" >nul
if errorlevel 1 (
  echo [FATAL] Failed to create junction "%~1" -> "%~2".
  exit /b 1
)
exit /b 0

:cleanup_bundle_sources
if exist "%bundle_source_dir%" rd /s /q "%bundle_source_dir%" >nul 2>&1
exit /b 0

:build_error
call :cleanup_bundle_sources
subst %short_drive% /D >nul 2>&1
echo.
echo Press any key to close this build script...
pause >nul
endlocal & exit /b 1
