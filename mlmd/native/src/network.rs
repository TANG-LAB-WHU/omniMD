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

    /// Whether the activation may bend somewhere between the arguments `lo`
    /// and `hi`: outside these ranges it is a straight line to the last bit
    /// of a float64 (1 - tanh² of 19 and exp of -37 are below 1e-16).
    fn bends_between(self, lo: f64, hi: f64) -> bool {
        match self {
            Self::Linear => false,
            Self::Tanh => hi > -19.0 && lo < 19.0,
            Self::Sigmoid | Self::Softplus => hi > -37.0 && lo < 37.0,
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

    /// The most that the argument of an activation can move while the scalar
    /// input goes from `lo` to `hi`, among the units of all layers that may
    /// be in the bent part of their activation somewhere on the way. The
    /// ranges of the arguments are bounds by interval arithmetic (every
    /// activation rises with its argument), so a unit moves no more than
    /// this says and none that bends is left out; a range that is not a
    /// number counts as unbounded.
    ///
    /// A piece of a table is only as good as the points it was checked at.
    /// Where no unit moves by more than a couple of units of its argument,
    /// the net has no feature between such points.
    #[cfg(any(test, feature = "gpu"))]
    pub(crate) fn bent_movement(&self, lo: f64, hi: f64) -> f64 {
        let mut x = vec![(lo, hi)];
        let mut most = 0.0_f64;
        for layer in &self.layers {
            let mut y = Vec::with_capacity(layer.bias.len());
            for (j, &bias) in layer.bias.iter().enumerate() {
                let (mut z_lo, mut z_hi) = (bias, bias);
                for (&(x_lo, x_hi), weights) in x.iter().zip(&layer.weights) {
                    let w = weights[j];
                    let (a, b) = (w * x_lo, w * x_hi);
                    z_lo += a.min(b);
                    z_hi += a.max(b);
                }
                if !(z_lo.is_finite() && z_hi.is_finite()) {
                    return f64::INFINITY;
                }
                if layer.activation.bends_between(z_lo, z_hi) {
                    most = most.max(z_hi - z_lo);
                }
                let scale = layer.timestep.as_ref().map_or(1.0, |s| s[j]);
                let a = layer.activation.value_derivative(z_lo).0 * scale;
                let b = layer.activation.value_derivative(z_hi).0 * scale;
                let (mut y_lo, mut y_hi) = (a.min(b), a.max(b));
                if !matches!(layer.residual, Residual::None) {
                    let (x_lo, x_hi) = x[j % x.len()];
                    y_lo += x_lo;
                    y_hi += x_hi;
                }
                y.push((y_lo, y_hi));
            }
            x = y;
        }
        most
    }

    /// Bounds on what rounding does to the value and to the slope of the
    /// outputs at the scalar `input`, the largest over the outputs of each.
    /// A sum is as exact as the sizes of its terms allow, whatever it comes
    /// to (a unit whose weight and bias cancel where it turns is computed
    /// to 2e-16 of the bias, not of the sum), and what a layer gets
    /// inexactly it hands on through its weights and the slope of its
    /// activation. First order, with one rounding counted for a sum.
    #[cfg(any(test, feature = "gpu"))]
    pub(crate) fn scalar_rounding(&self, input: f64) -> Result<(f64, f64)> {
        const ROUND: f64 = f64::EPSILON;
        // Values, slopes and the bounds on the rounding of either.
        let (mut x, mut dx) = (vec![input], vec![1.0]);
        let (mut ex, mut edx) = (vec![0.0], vec![0.0]);
        for layer in &self.layers {
            let n = layer.bias.len();
            let (mut y, mut dy) = (vec![0.0; n], vec![0.0; n]);
            let (mut ey, mut edy) = (vec![0.0; n], vec![0.0; n]);
            for j in 0..n {
                let (mut z, mut dz) = (layer.bias[j], 0.0);
                let (mut ez, mut edz) = (ROUND * layer.bias[j].abs(), 0.0);
                for i in 0..x.len() {
                    let w = layer.weights[i][j];
                    z += x[i] * w;
                    dz += dx[i] * w;
                    ez += w.abs() * ex[i] + ROUND * (x[i] * w).abs();
                    edz += w.abs() * edx[i] + ROUND * (dx[i] * w).abs();
                }
                let (value, slope) = layer.activation.value_derivative(z);
                let curvature = layer.activation.second_derivative(z);
                let scale = layer.timestep.as_ref().map_or(1.0, |s| s[j]);
                y[j] = value * scale;
                dy[j] = slope * scale * dz;
                ey[j] = scale.abs() * slope.abs() * ez + ROUND * y[j].abs();
                edy[j] = scale.abs() * (curvature.abs() * dz.abs() * ez + slope.abs() * edz)
                    + ROUND * dy[j].abs();
                if !matches!(layer.residual, Residual::None) {
                    y[j] += x[j % x.len()];
                    dy[j] += dx[j % x.len()];
                    ey[j] += ex[j % x.len()];
                    edy[j] += edx[j % x.len()];
                }
            }
            check(&y)?;
            check(&dy)?;
            check(&ey)?;
            check(&edy)?;
            (x, dx, ex, edx) = (y, dy, ey, edy);
        }
        let largest = |bounds: &[f64]| bounds.iter().copied().fold(0.0, f64::max);
        Ok((largest(&ex), largest(&edx)))
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
    fn the_bound_on_rounding_covers_what_cancelling_terms_lose() {
        // A unit that turns at x = 1000: the weight of 200 and the bias of
        // -200,000 cancel there, and the sum is exact to 2e-16 of 200,000.
        // The same unit written without the cancellation is the reference.
        let unit = Network {
            layers: vec![Layer {
                weights: vec![vec![200.0]],
                bias: vec![-200_000.0],
                activation: Activation::Tanh,
                residual: Residual::None,
                timestep: None,
            }],
        };
        let (mut worst, mut loosest) = (0.0_f64, 0.0_f64);
        for k in 0..2000 {
            let x = 999.99 + k as f64 * 1e-5;
            let [value, slope, _] = unit.scalar_jet(x).unwrap();
            let exact = (200.0 * (x - 1000.0)).tanh();
            let (bound, slope_bound) = unit.scalar_rounding(x).unwrap();
            let lost = (value[0] - exact).abs();
            assert!(lost <= bound, "x {x}: lost {lost:e}, bound {bound:e}");
            let lost_slope = (slope[0] - 200.0 * (1.0 - exact * exact)).abs();
            assert!(
                lost_slope <= slope_bound,
                "x {x}: {lost_slope:e} {slope_bound:e}"
            );
            worst = worst.max(lost);
            loosest = loosest.max(bound);
        }
        // What is lost is some 1e-11, and the bound is not far above it.
        assert!(worst > 3e-12 && loosest < 1e-10, "{worst:e} {loosest:e}");
        // Where the unit is flat nothing of it shows, and a net of terms of
        // its own size is exact to a few rounding errors.
        let (flat, _) = unit.scalar_rounding(900.0).unwrap();
        assert!(flat <= 4.0 * f64::EPSILON, "{flat:e}");
        let network = Network {
            layers: vec![
                layer(1, 3, Activation::Tanh, Residual::None),
                layer(3, 6, Activation::Sigmoid, Residual::Concat2),
                layer(6, 6, Activation::Softplus, Residual::Identity),
                layer(6, 4, Activation::Linear, Residual::None),
            ],
        };
        for x in [-0.8, 0.0, 0.37, 2.5] {
            let (value, slope) = network.scalar_rounding(x).unwrap();
            assert!(value > 1e-17 && value < 1e-14, "{value:e}");
            assert!(slope > 1e-17 && slope < 1e-14, "{slope:e}");
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
