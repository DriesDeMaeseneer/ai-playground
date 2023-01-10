use std::f32::consts::E;

#[derive(Clone)]
pub enum ActivationFunctionEnum {
    Sigmoid,
    ReLu,
    HeavisideStep,
    Tanh,
    None,
}

pub trait ActicationFunction {
    fn activation(&self, value: f32) -> f32;
    fn derivative(&self, value: f32) -> f32;
}
pub struct ActNone;
impl ActicationFunction for ActNone {
    fn activation(&self, value: f32) -> f32 {
        value
    }
    fn derivative(&self, value: f32) -> f32 {
        value
    }
}

pub struct Sigmoid;
impl ActicationFunction for Sigmoid {
    fn activation(&self, value: f32) -> f32 {
        1.0 / (1.0 + E.powf(-value))
    }
    fn derivative(&self, value: f32) -> f32 {
        self.activation(value) * (1.0 - self.activation(value))
    }
}
pub struct ReLu;
impl ActicationFunction for ReLu {
    fn activation(&self, value: f32) -> f32 {
        0.0_f32.max(value)
    }
    fn derivative(&self, value: f32) -> f32 {
        if value < 0.0 {
            return 0.0;
        } else {
            return 1.0;
        }
    }
}
pub struct HeavisideStep;
impl ActicationFunction for HeavisideStep {
    fn activation(&self, value: f32) -> f32 {
        if value < 0.0 {
            return 0.0;
        } else if value == 0.0 {
            return 0.5;
        } else {
            return 1.0;
        }
    }
    fn derivative(&self, _value: f32) -> f32 {
        0.0
    }
}
pub struct Tanh;
impl ActicationFunction for Tanh {
    fn activation(&self, value: f32) -> f32 {
        let pos = E.powf(value);
        let neg = E.powf(-value);
        (pos - neg) / (pos + neg)
    }
    fn derivative(&self, value: f32) -> f32 {
        1.0 - (self.activation(value).powi(2))
    }
}
