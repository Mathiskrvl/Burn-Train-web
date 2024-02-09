@echo off
REM Construire le projet Rust pour la cible wasm32-unknown-unknown
set RUSTFLAGS=--cfg web_sys_unstable_apis && cargo build --lib --release --target wasm32-unknown-unknown --manifest-path ./Cargo.toml
if %ERRORLEVEL% neq 0 goto error

REM Utiliser wasm-bindgen pour générer les fichiers nécessaires pour le web
wasm-bindgen --target web --no-typescript --out-dir ./pkg --out-name wasm ./target/wasm32-unknown-unknown/release/test_wasm.wasm
if %ERRORLEVEL% neq 0 goto error

echo La construction et la liaison ont réussi.
goto end

:error
echo Une erreur s'est produite pendant la construction ou la liaison.

:end
pause
