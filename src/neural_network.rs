use ndarray::{Array, Array1, IxDyn};

pub type Weights = Array1<f32>;
pub type Biases = Array1<f32>;
pub type Layer = (Weights, Biases);

// Neural network implementations and traits
pub struct NN<const AMT_LAYERS: usize> {
    layers: Vec<Layer>,
}
// used to create valid neural networks,
// make macro for layers?
pub struct NNBuilder {}
