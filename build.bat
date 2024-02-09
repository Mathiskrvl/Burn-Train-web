@echo off
REM Construire le projet Rust pour la cible wasm32-unknown-unknown
set RUSTFLAGS=--cfg web_sys_unstable_apis && wasm-pack build --release --target web --no-typescript --no-pack --out-name wasm
if %ERRORLEVEL% neq 0 goto error

echo La construction et la liaison ont réussi.
goto end

:error
echo Une erreur s'est produite pendant la construction ou la liaison.

:end
pause
