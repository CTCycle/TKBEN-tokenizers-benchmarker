@echo off
setlocal EnableExtensions
set "CI=%CI: =%"
if "%CI%"=="1" set "CI=true"
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
set "CLIENT=%ROOT%\app\client"
set "NPM=%ROOT%\runtimes\nodejs\npm.cmd"
if not exist "%NPM%" set "NPM=npm.cmd"

if not exist "%CLIENT%\package-lock.json" echo [FATAL] Missing frontend lockfile.& exit /b 1
if not exist "%ROOT%\app\server\uv.lock" echo [FATAL] Missing backend lockfile.& exit /b 1

pushd "%CLIENT%"
call "%NPM%" ci || (popd & exit /b 1)
call "%NPM%" run build || (popd & exit /b 1)
popd

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\release\tauri\scripts\prepare-runtime.ps1" || exit /b 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\release\tauri\scripts\clean-tauri-build.ps1" -KeepBundleSource || exit /b 1

pushd "%ROOT%\app"
call "%CLIENT%\node_modules\.bin\tauri.cmd" build --config src-tauri\tauri.conf.json --target x86_64-pc-windows-msvc || (popd & exit /b 1)
popd

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\release\tauri\scripts\export-windows-artifacts.ps1" || exit /b 1
echo [OK] Windows x64 MSI and portable artifacts exported.
exit /b 0
