use std::f32::consts::E;

pub trait ActicationFunction {
    fn activation(value: f32) -> f32;
    fn derivative(value: f32) -> f32;
}

pub struct Sigmoid;
impl ActicationFunction for Sigmoid {
    fn activation(value: f32) -> f32 {
        1.0 / (1.0 + E.powf(-value))
    }
    fn derivative(value: f32) -> f32 {
        Sigmoid::activation(value) * (1.0 - Sigmoid::activation(value))
    }
}
pub struct ReLu;
impl ActicationFunction for ReLu {
    fn activation(value: f32) -> f32 {
        0.0_f32.max(value)
    }
    fn derivative(value: f32) -> f32 {
        if value < 0.0 {
            return 0.0;
        } else if value > 0.0 {
            return 1.0;
        } else {
            // the derivative of the relu in 0 is undefined.
            panic!()
        }
    }
}
pub struct HeavisideStep;
impl ActicationFunction for HeavisideStep {
    fn activation(value: f32) -> f32 {
        if value < 0.0 {
            return 0.0;
        } else if value == 0.0 {
            return 0.5;
        } else {
            return 1.0;
        }
    }
    fn derivative(_value: f32) -> f32 {
        0.0
    }
}
pub struct Tanh;
impl ActicationFunction for Tanh {
    fn activation(value: f32) -> f32 {
        let pos = E.powf(value);
        let neg = E.powf(-value);
        (pos - neg) / (pos + neg)
    }
    fn derivative(value: f32) -> f32 {
        1.0 - (Tanh::activation(value).powi(2))
    }
}
