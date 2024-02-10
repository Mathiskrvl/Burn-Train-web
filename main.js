import init, { f, create_range, calculate_range, ModelWeb, ModelWebWgpu} from './pkg/wasm.js';
import {initChart, updateChart} from "./chart2d.js"

await init()
const hidden_layer = 32
let model = new ModelWeb(hidden_layer)
const backendDropdown = document.getElementById("backend");
if (!navigator.gpu) {
    backendDropdown.options[1].disabled = true;
    alert("WebGpu not available")
}

backendDropdown.addEventListener("change", handleBackendDropdownChange);

async function handleBackendDropdownChange() {
    const backend = this.value;
    if (backend === "ndarray") model = new ModelWeb(hidden_layer);
    if (backend === "webgpu") model = await new ModelWebWgpu(hidden_layer);
}

const input = document.getElementById("inputX")
document.getElementById("calcul").addEventListener("click", () => {
    let fx = f(input.value)
    document.getElementById("resultatFx").innerText = "f(x) = " + fx;
})

document.getElementById("calculg").addEventListener("click", async () => {
    let gx = await model.g(input.value)
    document.getElementById("resultatGx").innerText = "g(x) = " + gx;
})

const wasm_array = create_range(0, 10, 0.01)
const f_wasm_array = calculate_range(wasm_array)

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

const buttonTrain = document.getElementById("train")
buttonTrain.addEventListener("click", async () => {
    for (let i = 0; i <= 30; i++) {
        await sleep(50)
        await model.train(500, 0.001)
        let output = await model.inference(wasm_array)
        updateChart(myChart, output)
        console.log("update")
    }
})

let myChart = initChart(Array.from(wasm_array), Array.from(f_wasm_array))
let output = await model.inference(wasm_array)
updateChart(myChart, output)