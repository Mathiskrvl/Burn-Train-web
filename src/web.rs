use alloc::vec::Vec;
use crate::model::{Model, ModelConfig};
use wasm_bindgen::prelude::*;
use burn::{
    backend::{ndarray::NdArrayDevice, Autodiff, NdArray}, module::AutodiffModule, nn::loss::{MSELoss, Reduction}, optim::{AdamConfig, GradientsParams, Optimizer}, tensor::{ Data, Shape, Tensor}};
use burn_wgpu::{Wgpu, WgpuDevice, WebGpu, compute::init_async};
use js_sys::Array;
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::default());
}

#[wasm_bindgen]
pub fn f(x: f32) -> f32 {
    x.sin()
}

#[wasm_bindgen]
pub fn create_range(from: f32, to: f32, step: f32) -> Vec<f32> {
    let mut range = Vec::new();
    let mut current = from;
    while current < to {
        range.push(current);
        current += step;
    }
    range
}

#[wasm_bindgen]
pub fn calculate_range(range: Vec<f32>) -> Vec<f32> {
    range.iter().map(|&x| f(x)).collect()
}

// pub enum ModelType {
//     WithNdArrayBackend(Model<Autodiff<NdArray<f32>>>),
//     WithWgpuBackend(Model<Autodiff<Wgpu<AutoGraphicsApi, f32, i32>>>),
// }


#[wasm_bindgen]
pub struct ModelWeb {
    model: Model<Autodiff<NdArray<f32>>>,
    device: NdArrayDevice,
}

#[wasm_bindgen]
impl ModelWeb {
    #[wasm_bindgen(constructor)]
    pub fn new(hidden_size: usize) -> Self {
        log::info!("Initializing the model");
        let device = Default::default();
        Self {
            model: ModelConfig::new(hidden_size).init(&device),
            device,
        }
    }
    pub async fn g(&self, input: f32) -> f32 {
        let temp = [input];
        let output = self.infer(&temp).await;
        output[0]
    }

    async fn infer(&self, input: &[f32]) -> Vec<f32> {
        let result = self.model.valid().forward_web(input).await;
        result
    }

    pub async fn inference(&self, input: &[f32]) -> Array {
        let result = self.infer(input).await;
        let array = Array::new();
        for value in result {
            array.push(&value.into());
        }
        array
    }

    pub async fn train(&mut self, epoch: usize, lr: f64) { 
        let mut optim = AdamConfig::new().init();
        let data = create_range(-10f32, 10f32, 1e-2);
        let data_n = data.len();
        let fdata = calculate_range(data.clone());
        let input = Tensor::from_data(Data::new(data, Shape::new([data_n; 1])), &self.device).unsqueeze_dim(1);
        let example = Tensor::from_data(Data::new(fdata, Shape::new([data_n; 1])), &self.device).unsqueeze_dim(1);
        for i in 0..epoch {
            let output = self.model.forward(input.clone());
            let loss =  MSELoss::new().forward(output, example.clone(), Reduction::Auto);
            if i % 100 == 99 {
                #[cfg(not(target_family = "wasm"))]
                let scalar_loss = loss.clone().into_scalar();
                #[cfg(target_family = "wasm")]
                let scalar_loss = loss.clone().into_scalar().await;
                log::info!("{:?}", scalar_loss);
                // print scalar_loss directly in html
            }
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &self.model);
            self.model = optim.step(lr, self.model.clone(), grads);
        }
    }
}

#[wasm_bindgen]
pub struct ModelWebWgpu {
    model: Model<Autodiff<Wgpu<WebGpu, f32, i32>>>,
    device: WgpuDevice,
}

#[wasm_bindgen]
impl ModelWebWgpu {
    #[wasm_bindgen(constructor)]
    pub async fn new(hidden_size: usize) -> Self {
        log::info!("Initializing the model");
        let device = Default::default();
        init_async::<WebGpu>(&device).await;
        Self {
            model: ModelConfig::new(hidden_size).init(&device),
            device,
        }
    }
    pub async fn g(&self, input: f32) -> f32 {
        let temp = [input];
        let output = self.infer(&temp).await;
        output[0]
    }

    async fn infer(&self, input: &[f32]) -> Vec<f32> {
        let result = self.model.valid().forward_web(input).await;
        result
    }

    pub async fn inference(&self, input: &[f32]) -> Array {
        let result = self.infer(input).await;
        let array = Array::new();
        for value in result {
            array.push(&value.into());
        }
        array
    }

    pub async fn train(&mut self, epoch: usize, lr: f64) { 
        let mut optim = AdamConfig::new().init();
        let data = create_range(-10f32, 10f32, 1e-2);
        let data_n = data.len();
        let fdata = calculate_range(data.clone());
        let input = Tensor::from_data(Data::new(data, Shape::new([data_n; 1])), &self.device).unsqueeze_dim(1);
        let example = Tensor::from_data(Data::new(fdata, Shape::new([data_n; 1])), &self.device).unsqueeze_dim(1);
        for i in 0..epoch {
            let output = self.model.forward(input.clone());
            let loss =  MSELoss::new().forward(output, example.clone(), Reduction::Auto);
            if i % 100 == 99 {
                #[cfg(not(target_family = "wasm"))]
                let scalar_loss = loss.clone().into_scalar();
                #[cfg(target_family = "wasm")]
                let scalar_loss = loss.clone().into_scalar().await;
                log::info!("{:?}", scalar_loss);
                // print scalar_loss directly in html
            }
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &self.model);
            self.model = optim.step(lr, self.model.clone(), grads);
        }
    }
}