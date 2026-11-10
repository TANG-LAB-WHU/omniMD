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

#[cfg(any(test, feature = "gpu"))]
impl Activation {
    /// Second derivative, for the curvature the embedding table needs.
    fn second_derivative(self, x: f64) -> f64 {
        let (value, slope) = self.value_derivative(x);
        match self {
            Self::Linear => 0.0,
            Self::Tanh => -2.0 * value * slope,
            Self::Sigmoid => slope * (1.0 - 2.0 * value),
            // Softplus' is the sigmoid.
            Self::Softplus => slope * (1.0 - slope),
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

    /// Value, first and second derivative of every output with respect to
    /// the scalar input (second-order forward mode).
    #[cfg(any(test, feature = "gpu"))]
    pub(crate) fn scalar_jet(&self, input: f64) -> Result<[Vec<f64>; 3]> {
        let (mut x, mut dx, mut ddx) = (vec![input], vec![1.0], vec![0.0]);
        for layer in &self.layers {
            let n = layer.bias.len();
            let (mut y, mut dy, mut ddy) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            for j in 0..n {
                let (mut z, mut dz, mut ddz) = (layer.bias[j], 0.0, 0.0);
                for i in 0..x.len() {
                    let w = layer.weights[i][j];
                    z += x[i] * w;
                    dz += dx[i] * w;
                    ddz += ddx[i] * w;
                }
                let (value, slope) = layer.activation.value_derivative(z);
                let curvature = layer.activation.second_derivative(z);
                let scale = layer.timestep.as_ref().map_or(1.0, |s| s[j]);
                y[j] = value * scale;
                dy[j] = slope * scale * dz;
                ddy[j] = scale * (curvature * dz * dz + slope * ddz);
                if !matches!(layer.residual, Residual::None) {
                    y[j] += x[j % x.len()];
                    dy[j] += dx[j % x.len()];
                    ddy[j] += ddx[j % x.len()];
                }
            }
            check(&y)?;
            check(&dy)?;
            check(&ddy)?;
            (x, dx, ddx) = (y, dy, ddy);
        }
        Ok([x, dx, ddx])
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

#[cfg(test)]
mod tests {
    use crate::{Activation, Layer, Network, Residual};

    fn layer(input: usize, output: usize, activation: Activation, residual: Residual) -> Layer {
        // Deterministic, well-conditioned parameters.
        let value =
            |i: usize, j: usize| ((i * 7 + j * 13 + input + output) % 17) as f64 / 17.0 - 0.5;
        Layer {
            weights: (0..input)
                .map(|i| (0..output).map(|j| value(i, j)).collect())
                .collect(),
            bias: (0..output).map(|j| 0.3 * value(j, 5)).collect(),
            activation,
            residual,
            timestep: matches!(residual, Residual::Identity)
                .then(|| (0..output).map(|j| 0.5 + 0.1 * value(3, j)).collect()),
        }
    }

    #[test]
    fn scalar_jet_matches_first_derivative_and_its_finite_difference() {
        let network = Network {
            layers: vec![
                layer(1, 3, Activation::Tanh, Residual::None),
                layer(3, 6, Activation::Sigmoid, Residual::Concat2),
                layer(6, 6, Activation::Softplus, Residual::Identity),
                layer(6, 4, Activation::Linear, Residual::None),
            ],
        };
        let h = 1e-5;
        for x in [-0.8, 0.0, 0.37, 2.5] {
            let [value, first, second] = network.scalar_jet(x).unwrap();
            let (v, d) = network.scalar_with_derivative(x).unwrap();
            let (_, plus) = network.scalar_with_derivative(x + h).unwrap();
            let (_, minus) = network.scalar_with_derivative(x - h).unwrap();
            for j in 0..4 {
                assert!((value[j] - v[j]).abs() < 1e-14 && (first[j] - d[j]).abs() < 1e-14);
                let curvature = (plus[j] - minus[j]) / (2.0 * h);
                assert!(
                    (second[j] - curvature).abs() < 1e-8,
                    "x={x} output {j}: {} vs {curvature}",
                    second[j]
                );
            }
        }
    }
}
