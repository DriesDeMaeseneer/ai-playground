use ndarray::Array1;

pub struct BinaryPerceptron<const AMT_WEIGHTS: usize> {
    weights: Array1<f32>,
}

impl<const AMT_WEIGHTS: usize> BinaryPerceptron<AMT_WEIGHTS> {
    pub fn new() -> Self {
        BinaryPerceptron {
            weights: Array1::<f32>::zeros(AMT_WEIGHTS),
        }
    }
    pub fn activation(&self, features: &Array1<f32>) -> f32 {
        self.weights.dot(features)
    }
    pub fn classify(&self, sample: &Array1<f32>) -> i8 {
        if self.activation(&sample) > 0.0 {
            return 1;
        } else {
            return -1;
        }
    }
    pub fn isum(class: i8, b: &Array1<f32>) -> Array1<f32> {
        Array1::<f32>::from_elem(AMT_WEIGHTS, class as f32) + b
    }
    pub fn new_weights(&self, class: i8, sample: &Array1<f32>) -> Array1<f32> {
        self.weights.to_owned() + BinaryPerceptron::<AMT_WEIGHTS>::isum(class, &sample)
    }
    pub fn train(&mut self, class_and_samples: Vec<(i8, Array1<f32>)>) {
        let mut weights_have_updated: bool = true;
        while weights_have_updated {
            weights_have_updated = false;
            for (class, sample) in class_and_samples.iter() {
                if *class != self.classify(&sample) {
                    self.weights = self.new_weights(*class, &sample);
                    weights_have_updated = true;
                }
            }
        }
    }
}

#[cfg(test)]
pub mod test {
    use ndarray::arr1;

    use super::*;
    #[test]
    fn binary_perceptron_classification_test() {
        let mut perceptron = BinaryPerceptron::<2>::new();
        let data = vec![
            (1, arr1(&[1., 2.])),
            (1, arr1(&[2., 2.])),
            (1, arr1(&[2., 0.])),
            (-1, arr1(&[-2., 0.])),
            (-1, arr1(&[0., -2.])),
        ];
        perceptron.train(data);
        assert_eq!(perceptron.classify(&arr1(&[4., 2.])), 1);
        assert_eq!(perceptron.classify(&arr1(&[-1., -1.])), -1);
    }
}
