# Add wasm32 target for compiler.
rustup target add wasm32-unknown-unknown

if ! command -v wasm-pack &>/dev/null; then
    echo "wasm-pack could not be found. Installing ..."
    cargo install wasm-pack
    exit
fi

echo "Building your Rust project in wasm for web ..."
export RUSTFLAGS="--cfg web_sys_unstable_apis"
wasm-pack build --release --target web --no-typescript --no-pack --out-name wasm