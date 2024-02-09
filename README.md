# Burn-Train-web
This repo is an example of how we can train a model in the browser with WebAssembly.

[![Live Demo](https://img.shields.io/badge/live-demo-brightgreen)](https://mathiskrvl.github.io/Burn-Train-web/)

## Overview

This demo showcases how to execute an the training process in a web browser using a simple model.
The project utilizes the Burn deep learning framework, WebGPU and WebAssembly.

## Running the Demo

### Step 1: Install all dependecies for node

```bash
npm install
```

### Step 2: Build the WebAssembly Binary and Other Assets

To compile the Rust code into WebAssembly and build other essential files, execute the following
script:


If your are on linux:
```bash
npm run lin-wasm
```
And If your are on Windows:
```bash
npm run win-wasm
```

### Step 3: Launch the dev Web Server

Run the following command to initiate a dev web server on your local machine:

```bash
npm run dev
```

## Backend Compatibility

As of now, the WebGPU backend is compatible only with Chrome browsers running on macOS and Windows.
The application will dynamically detect if WebGPU support is available and proceed accordingly.