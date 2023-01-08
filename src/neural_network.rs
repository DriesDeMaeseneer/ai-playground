use crate::activation_functions::{self, ActicationFunction, ActivationFunctionEnum};
use ndarray::{Array1, Array2};
use rand::prelude::*;

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
    soft_max: bool,
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
            let mut pre_activation: Array1<f32> =
                current_weights.t().dot(&current_layer_actiovation) + current_biases;
            let backup = pre_activation.clone();

            pre_activation
                .mapv_inplace(|item| ActicationFunction::activation(&*current_layer.2, item));

            current_layer_actiovation = pre_activation;
            pre_and_post_activations.push((backup, current_layer_actiovation.clone()));
        }
        if self.soft_max {
            self.make_soft_max(&mut current_layer_actiovation);
        }
        (pre_and_post_activations, current_layer_actiovation)
    }
    pub fn make_soft_max(&self, forward_result: &mut Array1<f32>) {
        let total: f32 = forward_result.sum();
        if total == 0. {
            return;
        }
        *forward_result = &*forward_result / forward_result.sum();
    }
    pub fn backward(
        &self,
        forward_information: &ForwardResult,
        cost_vector: &Array1<f32>,
    ) -> BackwardResult {
        // derivation of 'last' layer.
        let dz_L = &forward_information.1 - cost_vector;
        let dw_L: Array1<f32> = (1.0 / self.batch_size as f32) * &forward_information.1 * &dz_L;
        let mut z_derivatives = vec![dz_L.clone()];
        let mut output: BackwardResult = vec![(dw_L, dz_L)];
        for l in (0..self.layers.len() - 1).rev() {
            let al_minus_1: &Array1<f32> = &forward_information.0[l + 1].1;
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
        if total_batch.is_empty() {
            *total_batch = new_gradient.clone();
        }
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
            self.layers[i].0 = &self.layers[i].0
                - self.alpha * &total_batch_gradients[self.layers.len() - i - 1].0;
            // update biases
            self.layers[i].1 = &self.layers[i].1
                - self.alpha * &total_batch_gradients[self.layers.len() - i - 1].1;
        }
    }
    pub fn mini_batch_gradient_descent(&mut self, input_data: &Vec<(Array1<f32>, Array1<f32>)>) {
        // TODO: iterations should behave as epochs(doing all intputs once.)
        let mut rng = rand::thread_rng();
        for i in 0..self.iterations {
            let mut batch_costs: Vec<f32> = Vec::with_capacity(self.batch_size);
            let mut current_batch_mean_gradients: BackwardResult = Vec::new();
            let minibatch = input_data
                .into_iter()
                .choose_multiple(&mut rng, self.batch_size);
            for sample in minibatch.into_iter() {
                let current_input = &sample.0;
                let current_expected_result = &sample.1;

                let forward_result = self.forward(current_input);
                let backward_result = self.backward(
                    &forward_result,
                    &self.calculate_loss(&forward_result, current_expected_result),
                );
                self.sum_batch(&mut current_batch_mean_gradients, &backward_result);
                let cost = self.calculate_cost(&forward_result, &current_expected_result);
                batch_costs.push(cost);
            }

            self.update_parameters(&current_batch_mean_gradients);
            let average_cost = self.average_cost(&batch_costs);
            println!(
                "Iteration {} ({}% completed) has an average cost of {}",
                i,
                i as f32 * 100. / self.iterations as f32,
                average_cost
            );
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
    pub fn average_cost(&self, costs: &Vec<f32>) -> f32 {
        let mut output = 0.0;
        for cost in costs.into_iter() {
            output += cost.abs();
        }
        output / costs.len() as f32
    }
    pub fn evaluate(&self, input: &Array1<f32>) -> f32 {
        let forward_pass = self.forward(&input);
        self.argmax(&forward_pass.1)
    }
    fn argmax(&self, input: &Array1<f32>) -> f32 {
        if input.len() == 1 {
            println!("argmax {}", input[0]);
            return input[0];
        }
        let mut max_index = 0;
        let mut max_val: f32 = 0.;
        for (index, value) in input.into_iter().enumerate() {
            if value.abs() > max_val {
                max_val = value.abs();
                max_index = index;
            }
        }
        max_index as f32
    }
}
// used to create valid neural networks,
// make macro for layers?
pub struct NNBuilder {
    layers: Vec<Layer>,
    alpha: f32,
    batch_size: usize,
    iterations: usize,
    input_amt: usize,
    soft_max: bool,
}

impl NNBuilder {
    pub fn new() -> Self {
        NNBuilder {
            alpha: 0.1,
            layers: Vec::new(),
            batch_size: 1,
            iterations: 0,
            input_amt: 0,
            soft_max: false,
        }
    }
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }
    pub fn soft_max(mut self, soft_max: bool) -> Self {
        self.soft_max = soft_max;
        self
    }
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }
    pub fn with_input_nodes(mut self, input_amt: usize) -> Self {
        self.input_amt = input_amt;
        self
    }
    pub fn add_layer(
        mut self,
        layer_size: usize,
        activation_function: ActivationFunctionEnum,
    ) -> Self {
        let activation_fuct: Box<dyn ActicationFunction> = match activation_function {
            ActivationFunctionEnum::Sigmoid => Box::new(activation_functions::Sigmoid {}),
            ActivationFunctionEnum::ReLu => Box::new(activation_functions::ReLu),
            ActivationFunctionEnum::HeavisideStep => Box::new(activation_functions::HeavisideStep),
            ActivationFunctionEnum::Tanh => Box::new(activation_functions::Tanh),
        };
        if self.layers.len() == 0 {
            let new_layer: Layer = (
                Array2::<f32>::zeros((self.input_amt, layer_size)),
                Array1::<f32>::zeros(layer_size),
                activation_fuct,
            );
            self.layers.push(new_layer);
        } else {
            let new_layer: Layer = (
                Array2::<f32>::zeros((self.layers.last().unwrap().1.shape()[0], layer_size)),
                Array1::<f32>::zeros(layer_size),
                activation_fuct,
            );
            self.layers.push(new_layer);
        }
        self
    }
    pub fn build(self) -> NN {
        NN {
            layers: self.layers,
            batch_size: self.batch_size,
            alpha: self.alpha,
            iterations: self.iterations,
            soft_max: self.soft_max,
        }
    }
}

#[cfg(test)]
pub mod tests {
    use ndarray::{arr1, Array1};

    use crate::activation_functions::ActivationFunctionEnum;
    use rand::prelude::*;

    use super::NN;

    #[test]
    fn binary_nn_classification_test() {
        let mut network = NN::new()
            .with_alpha(0.1)
            .with_iterations(1000)
            .with_batch_size(10)
            .with_input_nodes(2)
            .add_layer(1, ActivationFunctionEnum::Sigmoid)
            .add_layer(1, ActivationFunctionEnum::Sigmoid)
            .soft_max(false)
            .build();

        let train_data = vec![
            (arr1(&[1., 2.]), arr1(&[1.])),
            (arr1(&[10.6, 23.5]), arr1(&[1.])),
            (arr1(&[4., 4.]), arr1(&[1.])),
            (arr1(&[3., 6.]), arr1(&[1.])),
            (arr1(&[2., 2.]), arr1(&[1.])),
            (arr1(&[5., 2.]), arr1(&[1.])),
            (arr1(&[3., 2.]), arr1(&[1.])),
            (arr1(&[2., 2.]), arr1(&[1.])),
            (arr1(&[1., 1.]), arr1(&[1.])),
            (arr1(&[-1., -1.]), arr1(&[0.])),
            (arr1(&[-5., -5.]), arr1(&[0.])),
            (arr1(&[-2., -9.]), arr1(&[0.])),
            (arr1(&[-2., 5.]), arr1(&[0.])),
            (arr1(&[-1., -3.]), arr1(&[0.])),
            (arr1(&[-20., -16.]), arr1(&[0.])),
            (arr1(&[-5., -4.]), arr1(&[0.])),
            (arr1(&[-7., -3.]), arr1(&[0.])),
        ];
        network.mini_batch_gradient_descent(&train_data);
        assert!(network.evaluate(&arr1(&[3., 3.])) > 0.8);
        assert!(network.evaluate(&arr1(&[-20., -20.])) < 0.2);
    }

    fn chess_board_data(amt: usize) -> Vec<(Array1<f32>, Array1<f32>)> {
        let mut rng = rand::thread_rng();
        let mut output = Vec::new();
        for _ in 0..amt {
            let x_coord: f32 = rng.gen_range(-8.0..=8.0);
            let y_coord: f32 = rng.gen_range(-8.0..=8.0);
            let class_index: f32;
            if x_coord * y_coord > 0.0 {
                class_index = -1.;
            } else {
                class_index = 1.;
            }
            output.push((arr1(&[x_coord, y_coord]), arr1(&[class_index])));
        }
        output
    }
    #[test]
    fn chess_board_classification_test() {
        let mut network = NN::new()
            .with_alpha(0.01)
            .with_iterations(100)
            .with_batch_size(30)
            .with_input_nodes(2)
            .add_layer(4, ActivationFunctionEnum::ReLu)
            .add_layer(1, ActivationFunctionEnum::ReLu)
            .soft_max(false)
            .build();

        let data_points: usize = 100;
        let train_data = chess_board_data(data_points);
        network.mini_batch_gradient_descent(&train_data);

        assert!(network.evaluate(&arr1(&[3., 3.])) > 0.);
        assert!(network.evaluate(&arr1(&[-4., -4.])) > 0.);
        assert!(network.evaluate(&arr1(&[-3., 3.])) < 0.);
        assert!(network.evaluate(&arr1(&[5., -4.])) < 0.);
    }
}
