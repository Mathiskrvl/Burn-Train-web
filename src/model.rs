use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, ReLU,},
    tensor::{backend::Backend, Tensor},
};
use alloc::vec::Vec;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: ReLU,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear2.forward(self.activation.forward(self.linear1.forward(input)))
    }
    pub async fn forward_web(&self, input: &[f32]) -> Vec<f32> {
        let input: Tensor<B, 2> = Tensor::from_floats(input, &B::Device::default()).unsqueeze_dim(1);
        let output = self.forward(input);
        #[cfg(not(target_family = "wasm"))]
        let result = output.into_data().convert::<f32>().value;
        #[cfg(target_family = "wasm")]
        let result = output.into_data().await.convert::<f32>().value;
        result
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    hidden_size: usize,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(1, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, 1).init(device),
            activation: ReLU::new(),
        }
    }
    pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
        Model {
            linear1: LinearConfig::new(1, self.hidden_size).init_with(record.linear1),
            linear2: LinearConfig::new(self.hidden_size, 1).init_with(record.linear2),
            activation: ReLU::new(),
        }
    }
}