use ndarray::{Array1, Array2};

use crate::activation_functions::{ActicationFunction, ActivationFunctionEnum};

pub type Weights = Array2<f32>;
pub type Biases = Array1<f32>;
pub type Layer = (Weights, Biases, Box<dyn ActicationFunction>);
// first element is the pre and post activations for all layers.
// the second element in the tuple is the result of the total forward function.
pub type ForwardResult = (Vec<(Array1<f32>, Array1<f32>)>, Array1<f32>);
// list of updates to weights and bias per layer.
pub type BackwardResult = Vec<(Array1<f32>, Array1<f32>)>;

// Neural network implementation
pub struct NN {
    layers: Vec<Layer>,
    alpha: f32,
    batch_size: usize,
    iterations: usize,
}
impl NN {
    pub fn new() -> NNBuilder {
        NNBuilder::new()
    }
    pub fn forward(&self, input: &Array1<f32>) -> ForwardResult {
        let mut pre_and_post_activations: Vec<(Array1<f32>, Array1<f32>)> =
            Vec::with_capacity(self.layers.len());
        pre_and_post_activations.push((input.clone(), Array1::<f32>::zeros(0)));
        let mut current_layer_actiovation: Array1<f32> = input.clone();
        for i in 0..self.layers.len() {
            let current_layer = &self.layers[i];
            let current_weights = &current_layer.0;
            let current_biases = &current_layer.1;
            let pre_activation: Array1<f32> =
                current_weights.dot(&current_layer_actiovation) + current_biases;
            let backup = pre_activation.clone();

            pre_activation.for_each(|item| {
                ActicationFunction::activation(&*current_layer.2, *item);
            });

            current_layer_actiovation = pre_activation;
            pre_and_post_activations.push((backup, current_layer_actiovation.clone()));
        }
        (pre_and_post_activations, current_layer_actiovation)
    }
    pub fn backward(
        &self,
        forward_information: &ForwardResult,
        cost_vector: &Array1<f32>,
    ) -> BackwardResult {
        // derivation of 'last' layer.
        let dz_L = forward_information.1.clone() - cost_vector;
        let mut z_derivatives = vec![dz_L];
        let mut output: BackwardResult = Vec::new();
        for l in (1..self.layers.len()).rev() {
            let al_minus_1: &Array1<f32> = &forward_information.0[l - 1].1;
            // derivation of i'th layer.
            let dz_l: Array1<f32> = self.layers[l + 1].0.dot(
                &z_derivatives
                    .last()
                    .expect("Could not fetch last derivation.") as &Array1<f32>,
            );
            dz_l.for_each(|item| {
                ActicationFunction::derivative(&*self.layers[l].2, *item);
            });
            z_derivatives.push(dz_l.clone());
            // derivation of weights at i'th layer.
            let dw_l: Array1<f32> = (1.0 / self.batch_size as f32) * &dz_l * al_minus_1;
            // derivation of biases at i'th layer.
            let db_l: Array1<f32> = (1.0 / self.batch_size as f32) * dz_l;
            output.push((dw_l, db_l));
        }
        output
    }
    fn sum_batch(&self, total_batch: &mut BackwardResult, new_gradient: &BackwardResult) {
        for i in 0..total_batch.len() {
            let old_batch_weights: &Array1<f32> = &total_batch[i].0;
            let old_batch_biases: &Array1<f32> = &total_batch[i].1.clone();

            let new_weights: &Array1<f32> = &new_gradient[i].0;
            let new_biases: &Array1<f32> = &new_gradient[i].1;

            total_batch[i].0 = old_batch_weights + new_weights;
            total_batch[i].1 = old_batch_biases + new_biases;
        }
    }
    pub fn update_parameters(&mut self, total_batch_gradients: &BackwardResult) {
        for i in 0..self.layers.len() {
            // update weights
            self.layers[0].0 = &self.layers[0].0 - self.alpha * &total_batch_gradients[i].0;
            // update biases
            self.layers[0].1 = &self.layers[0].1 - self.alpha * &total_batch_gradients[i].1;
        }
    }
    pub fn calculate_loss(
        &self,
        forward_information: &ForwardResult,
        expected_probabilities: &Array1<f32>,
    ) -> Array1<f32> {
        &forward_information.1 - expected_probabilities
    }
    pub fn calculate_cost(
        &self,
        forward_information: &ForwardResult,
        expected_probabilities: &Array1<f32>,
    ) -> f32 {
        self.calculate_loss(forward_information, expected_probabilities)
            .sum()
    }
    pub fn mini_batch_gradient_descent(&mut self, input_data: &Vec<(Array1<f32>, Array1<f32>)>) {
        for i in (0..self.iterations).step_by(self.batch_size) {
            let mut costs: Vec<f32> = Vec::with_capacity(self.batch_size);
            let mut total_batch_update: BackwardResult = Vec::new();
            for j in 0..self.batch_size {
                let current_input = &input_data[i + j].0;
                let current_expected_result = &input_data[i + j].1;

                let forward_result = self.forward(current_input);
                let backward_result = self.backward(
                    &forward_result,
                    &self.calculate_loss(&forward_result, current_expected_result),
                );
                self.sum_batch(&mut total_batch_update, &backward_result);
                let cost = self.calculate_cost(&forward_result, &current_expected_result);
                costs.push(cost);
            }
            self.update_parameters(&total_batch_update);
            let average_cost = self.average_cost(&costs);
            println!(
                "Iteration {} ({}% completed) has an average cost of {}",
                i + self.batch_size,
                (i + self.batch_size) as f32 / self.iterations as f32,
                average_cost
            );
        }
    }

    pub fn average_cost(&self, costs: &Vec<f32>) -> f32 {
        let mut output = 0.0;
        for cost in costs.into_iter() {
            output += cost;
        }
        output / costs.len() as f32
    }
}
// used to create valid neural networks,
// make macro for layers?
pub struct NNBuilder {
    layers: Vec<Layer>,
    alpha: f32,
    batch_size: usize,
    iterations: usize,
}

impl NNBuilder {
    pub fn new() -> Self {
        NNBuilder {
            alpha: 0.1,
            layers: Vec::new(),
            batch_size: 1,
            iterations: 0,
        }
    }
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }
    pub fn add_layer(
        mut self,
        layer_size: usize,
        activation_function: ActivationFunctionEnum,
    ) -> Self {
        self.layers.push((Array2::<f32>::zeros((0, 0)), Array1))
    }
}

#[cfg(test)]
pub mod tests {}
