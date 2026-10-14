use crate::{Activation, Layer, Network, Residual};
use omnimd_ml_contract::{Error, Result};

fn check(values: &[f64]) -> Result<()> {
    if values.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error("nonfinite native network intermediate".into()))
    }
}

impl Activation {
    fn value_derivative(self, x: f64) -> (f64, f64) {
        match self {
            Self::Linear => (x, 1.0),
            Self::Tanh => {
                let y = x.tanh();
                (y, 1.0 - y * y)
            }
            Self::Sigmoid => {
                let y = if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let z = x.exp();
                    z / (1.0 + z)
                };
                (y, y * (1.0 - y))
            }
            Self::Softplus => {
                let y = x.max(0.0) + (-x.abs()).exp().ln_1p();
                let sigmoid = if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let z = x.exp();
                    z / (1.0 + z)
                };
                (y, sigmoid)
            }
        }
    }
}

impl Layer {
    fn forward(&self, input: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        check(input)?;
        let mut y = self.bias.clone();
        for (i, x) in input.iter().enumerate() {
            for (j, out) in y.iter_mut().enumerate() {
                *out += x * self.weights[i][j];
            }
        }
        check(&y)?;
        let mut derivative = vec![0.0; y.len()];
        for j in 0..y.len() {
            let (value, slope) = self.activation.value_derivative(y[j]);
            let scale = self.timestep.as_ref().map_or(1.0, |s| s[j]);
            y[j] = value * scale;
            derivative[j] = slope * scale;
            if !matches!(self.residual, Residual::None) {
                y[j] += input[j % input.len()];
            }
        }
        check(&y)?;
        check(&derivative)?;
        Ok((y, derivative))
    }
}

impl Network {
    /// Forward tangent propagation for the scalar radial embedding input.
    pub(crate) fn scalar_with_derivative(&self, input: f64) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut x = vec![input];
        let mut dx = vec![1.0];
        for layer in &self.layers {
            let (y, slopes) = layer.forward(&x)?;
            let mut dy = vec![0.0; y.len()];
            for j in 0..y.len() {
                dy[j] = slopes[j]
                    * dx.iter()
                        .enumerate()
                        .map(|(i, v)| v * layer.weights[i][j])
                        .sum::<f64>();
                if !matches!(layer.residual, Residual::None) {
                    dy[j] += dx[j % dx.len()];
                }
            }
            x = y;
            check(&dy)?;
            dx = dy;
        }
        Ok((x, dx))
    }

    /// Reverse chain rule for the scalar atomic fitting output.
    pub(crate) fn scalar_with_gradient(&self, input: &[f64]) -> Result<(f64, Vec<f64>)> {
        let mut x = input.to_vec();
        let mut slopes = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (y, derivative) = layer.forward(&x)?;
            x = y;
            slopes.push(derivative);
        }
        let value = x[0];
        let mut adjoint = vec![1.0];
        for (layer, slope) in self.layers.iter().zip(slopes).rev() {
            let input_size = layer.weights.len();
            let mut previous = vec![0.0; input_size];
            for i in 0..input_size {
                previous[i] = (0..adjoint.len())
                    .map(|j| adjoint[j] * slope[j] * layer.weights[i][j])
                    .sum();
            }
            if !matches!(layer.residual, Residual::None) {
                for (j, v) in adjoint.iter().enumerate() {
                    previous[j % input_size] += v;
                }
            }
            check(&previous)?;
            adjoint = previous;
        }
        Ok((value, adjoint))
    }
}
