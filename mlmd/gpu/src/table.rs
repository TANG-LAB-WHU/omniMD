//! Tabulated scalar-input networks (plan I-5, DeePMD `dp compress`).
//!
//! An embedding net is a smooth function of one number, so it can be
//! replaced by a piecewise quintic polynomial per output: on each interval
//! of width `step` the polynomial matches the value, first and second
//! derivative at both ends. With `step = 0.01` the difference from the
//! network is at float64 rounding level, at about a tenth of the arithmetic
//! and none of its intermediate storage.
//!
//! Inputs outside the table are clamped and flagged; the caller decides what
//! to do (the se_e2_a path re-evaluates that frame with the exact network).
use crate::{check_len, cuda_error, Device, Scalar};
use cudarc::driver::{
    CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use omnimd_ml_contract::{Error, Result};

const SOURCE: &str = r#"
// coef: [intervals][width][6], polynomial in u = (x - lower)/step - interval.
template <typename T>
__device__ void scalar_table(const T* x, long long rows, const T* coef, int width, int intervals,
        T lower, T inverse_step, T* out, T* dout, int* flag) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * width) return;
    long long row = idx / width;
    int m = (int)(idx % width);
    T position = (x[row] - lower) * inverse_step;
    // Rounding may leave an input a hair outside; anything more is reported.
    if (!(position >= T(-1e-3) && position <= T(intervals) + T(1e-3))) *flag = 1;
    if (!(position > T(0))) position = T(0);
    if (position > T(intervals)) position = T(intervals);
    int i = (int)position;
    if (i >= intervals) i = intervals - 1;
    T u = position - T(i);
    const T* c = coef + ((long long)i * width + m) * 6;
    out[idx] = c[0] + u * (c[1] + u * (c[2] + u * (c[3] + u * (c[4] + u * c[5]))));
    dout[idx] = inverse_step
        * (c[1] + u * (T(2) * c[2] + u * (T(3) * c[3] + u * (T(4) * c[4] + u * T(5) * c[5]))));
}

#define INSTANTIATE(T, S)                                                                    \
extern "C" __global__ void scalar_table_##S(const T* x, long long rows, const T* coef,       \
        int width, int intervals, T lower, T inverse_step, T* out, T* dout, int* flag) {      \
    scalar_table<T>(x, rows, coef, width, intervals, lower, inverse_step, out, dout, flag);  \
}
INSTANTIATE(float, f32)
INSTANTIATE(double, f64)
"#;

const BLOCK: u32 = 256;

/// Quintic Hermite coefficients, `[intervals][width][6]` in the normalized
/// coordinate `u ∈ [0, 1]` of each interval, from samples at the
/// `intervals + 1` knots: `values`, `first` and `second` derivatives, each
/// `[knots][width]`.
pub fn quintic_coefficients(
    step: f64,
    width: usize,
    values: &[f64],
    first: &[f64],
    second: &[f64],
) -> Result<Vec<f64>> {
    let knots = values.len() / width.max(1);
    if width == 0
        || knots < 2
        || values.len() != knots * width
        || first.len() != values.len()
        || second.len() != values.len()
        || !(step.is_finite() && step > 0.0)
    {
        return Err(Error(format!(
            "table needs a positive step and at least two knots of {width} values, first and second derivatives"
        )));
    }
    let mut coefficients = Vec::with_capacity((knots - 1) * width * 6);
    for i in 0..knots - 1 {
        for m in 0..width {
            let (a, b) = (i * width + m, (i + 1) * width + m);
            let c0 = values[a];
            let c1 = first[a] * step;
            let c2 = second[a] * step * step / 2.0;
            // What the quadratic start still misses at the far end.
            let d0 = values[b] - c0 - c1 - c2;
            let d1 = first[b] * step - c1 - 2.0 * c2;
            let d2 = second[b] * step * step - 2.0 * c2;
            coefficients.extend([
                c0,
                c1,
                c2,
                10.0 * d0 - 4.0 * d1 + d2 / 2.0,
                -15.0 * d0 + 7.0 * d1 - d2,
                6.0 * d0 - 3.0 * d1 + d2 / 2.0,
            ]);
        }
    }
    if coefficients.iter().any(|c| !c.is_finite()) {
        return Err(Error("nonfinite table coefficient".into()));
    }
    Ok(coefficients)
}

/// A piecewise quintic table on `[lower, lower + intervals·step]`.
pub struct ScalarTable<T: Scalar> {
    width: usize,
    intervals: usize,
    lower: T,
    inverse_step: T,
    coefficients: CudaSlice<T>,
    function: CudaFunction,
}

impl<T: Scalar> ScalarTable<T> {
    /// `coefficients` from [`quintic_coefficients`].
    pub fn new(
        device: &Device,
        lower: f64,
        step: f64,
        width: usize,
        coefficients: &[f64],
    ) -> Result<Self> {
        let intervals = coefficients.len() / (6 * width.max(1));
        if width == 0
            || intervals == 0
            || coefficients.len() != intervals * width * 6
            || i32::try_from(intervals).is_err()
            || i32::try_from(width).is_err()
            || !(lower.is_finite() && step.is_finite() && step > 0.0)
        {
            return Err(Error("invalid scalar table shape or range".into()));
        }
        let cast: Vec<T> = coefficients.iter().map(|&c| T::from_f64(c)).collect();
        Ok(Self {
            width,
            intervals,
            lower: T::from_f64(lower),
            inverse_step: T::from_f64(1.0 / step),
            coefficients: device.upload(&cast)?,
            function: device
                .kernels()
                .function(SOURCE, &format!("scalar_table_{}", T::SUFFIX))?,
        })
    }

    pub fn output(&self) -> usize {
        self.width
    }

    pub fn intervals(&self) -> usize {
        self.intervals
    }

    /// `out[rows][width]` and its derivative with respect to `x[rows]`.
    /// `flag[0]` is set to 1 when some input lies outside the table (its
    /// row is then evaluated at the nearest end).
    pub fn forward_tangent_into(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        rows: usize,
        out: &mut impl DevicePtrMut<T>,
        dout: &mut impl DevicePtrMut<T>,
        flag: &mut CudaSlice<i32>,
    ) -> Result<()> {
        check_len("scalar table input", x.len(), rows)?;
        check_len("scalar table output", out.len(), rows * self.width)?;
        check_len("scalar table tangent", dout.len(), rows * self.width)?;
        let total = rows * self.width;
        let blocks = total.div_ceil(BLOCK as usize);
        if total == 0 || blocks > u32::MAX as usize || flag.len() == 0 {
            return Err(Error(format!(
                "scalar table over {rows} rows is out of range"
            )));
        }
        let stream = device.stream();
        let (px, _g0) = x.device_ptr(stream);
        let (pout, _g1) = out.device_ptr_mut(stream);
        let (pdout, _g2) = dout.device_ptr_mut(stream);
        let (rows_i, width_i, intervals_i) =
            (rows as i64, self.width as i32, self.intervals as i32);
        let mut launch = stream.launch_builder(&self.function);
        launch
            .arg(&px)
            .arg(&rows_i)
            .arg(&self.coefficients)
            .arg(&width_i)
            .arg(&intervals_i)
            .arg(&self.lower)
            .arg(&self.inverse_step)
            .arg(&pout)
            .arg(&pdout)
            .arg(flag);
        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(config) }
            .map(drop)
            .map_err(cuda_error("scalar table launch"))
    }
}

#[cfg(test)]
mod tests {
    use super::quintic_coefficients;

    #[test]
    fn quintic_reproduces_a_smooth_function_and_its_slope() {
        // f_m(x) = sin((m + 1) x) on [-1, 2] with step 0.01.
        let (lower, step, width, knots) = (-1.0, 0.01, 3, 301);
        let sample = |order: usize| -> Vec<f64> {
            (0..knots)
                .flat_map(|i| {
                    let x = lower + i as f64 * step;
                    (0..width).map(move |m| {
                        let k = (m + 1) as f64;
                        match order {
                            0 => (k * x).sin(),
                            1 => k * (k * x).cos(),
                            _ => -k * k * (k * x).sin(),
                        }
                    })
                })
                .collect()
        };
        let c = quintic_coefficients(step, width, &sample(0), &sample(1), &sample(2)).unwrap();
        assert_eq!(c.len(), 300 * width * 6);
        let mut worst = (0.0_f64, 0.0_f64);
        for trial in 0..2000 {
            let x = lower + 3.0 * (trial as f64 + 0.37) / 2000.0;
            let position = (x - lower) / step;
            let i = (position as usize).min(299);
            let u = position - i as f64;
            for m in 0..width {
                let p = &c[(i * width + m) * 6..][..6];
                let value = p
                    .iter()
                    .rev()
                    .fold(0.0, |acc, &coefficient| acc * u + coefficient);
                let slope = (1..6).rev().fold(0.0, |acc, k| acc * u + k as f64 * p[k]) / step;
                let k = (m + 1) as f64;
                worst.0 = worst.0.max((value - (k * x).sin()).abs());
                worst.1 = worst.1.max((slope - k * (k * x).cos()).abs());
            }
        }
        // Error ~ step^6 f^(6) / 46080: 1e-12 · 729 / 46080.
        assert!(worst.0 < 1e-13, "value error {}", worst.0);
        assert!(worst.1 < 1e-10, "slope error {}", worst.1);
        assert!(quintic_coefficients(0.0, width, &sample(0), &sample(1), &sample(2)).is_err());
        assert!(quintic_coefficients(
            step,
            width,
            &sample(0)[..3],
            &sample(1)[..3],
            &sample(2)[..3]
        )
        .is_err());
    }
}
