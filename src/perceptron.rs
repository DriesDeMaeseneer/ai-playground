pub struct BinaryPerceptron<const AMT_WEIGHTS: usize> {
    weights: [f32; AMT_WEIGHTS],
}

impl<const AMT_WEIGHTS: usize> BinaryPerceptron<AMT_WEIGHTS> {
    pub fn new() -> Self {
        BinaryPerceptron {
            weights: [0.0; AMT_WEIGHTS],
        }
    }
    pub fn activation(&self, features: [f32; AMT_WEIGHTS]) -> f32 {
        self.weights
            .into_iter()
            .enumerate()
            .map(|(ith, weight)| weight * features[ith])
            .sum()
    }
    pub fn classify(&self, sample: [f32; AMT_WEIGHTS]) -> i8 {
        if self.activation(sample) > 0.0 {
            return 1;
        } else {
            return -1;
        }
    }
    pub fn imult(class: i8, b: [f32; AMT_WEIGHTS]) -> [f32; AMT_WEIGHTS] {
        let mut imult_result: [f32; AMT_WEIGHTS] = [0.0; AMT_WEIGHTS];
        for imult_index in 0..AMT_WEIGHTS {
            imult_result[imult_index] = class as f32 + b[imult_index];
        }
        imult_result
    }
    pub fn sum(a: [f32; AMT_WEIGHTS], b: [f32; AMT_WEIGHTS]) -> [f32; AMT_WEIGHTS] {
        let mut sum_result: [f32; AMT_WEIGHTS] = [0.0; AMT_WEIGHTS];
        for sum_index in 0..AMT_WEIGHTS {
            sum_result[sum_index] = a[sum_index] + b[sum_index];
        }
        sum_result
    }
    pub fn new_weights(&self, class: i8, sample: [f32; AMT_WEIGHTS]) -> [f32; AMT_WEIGHTS] {
        BinaryPerceptron::<AMT_WEIGHTS>::sum(
            self.weights,
            BinaryPerceptron::<AMT_WEIGHTS>::imult(class, sample),
        )
    }
    pub fn train(&mut self, class_and_samples: Vec<(i8, [f32; AMT_WEIGHTS])>) {
        let mut weights_have_updated: bool = true;
        while weights_have_updated {
            weights_have_updated = false;
            for (class, sample) in class_and_samples.iter() {
                if *class != self.classify(*sample) {
                    self.weights = self.new_weights(*class, *sample);
                    weights_have_updated = true;
                }
            }
        }
    }
    pub fn get_weights(&self) -> [f32; AMT_WEIGHTS] {
        return self.weights;
    }
}
