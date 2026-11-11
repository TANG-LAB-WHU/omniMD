//! Device smoke tests against plain CPU references. They skip without a CUDA
//! driver; set OMNIMD_REQUIRE_GPU=1 (GPU CI, local 4090 runs) to make a missing
//! device a failure instead.
use omnimd_gpu::{
    Activation, Cell, DenseLayer, DenseNetwork, Device, MatmulShape, NeighborList, NeighborSearch,
    COINCIDENT, TRUNCATED,
};
use std::sync::OnceLock;

fn device() -> Option<&'static Device> {
    static DEVICE: OnceLock<Option<Device>> = OnceLock::new();
    DEVICE
        .get_or_init(|| match Device::new(0) {
            Ok(device) => Some(device),
            Err(error) if std::env::var_os("OMNIMD_REQUIRE_GPU").is_some() => {
                panic!("OMNIMD_REQUIRE_GPU is set but no device: {error}")
            }
            Err(error) => {
                eprintln!("skipping GPU tests: {error}");
                None
            }
        })
        .as_ref()
}

/// Deterministic values in (-1, 1) (64-bit LCG), so failures reproduce.
fn values(count: usize, seed: u64) -> Vec<f64> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

fn matmul_cpu(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    c
}

/// Max |gpu - cpu| / (1 + |cpu|) over all entries.
fn max_error(gpu: impl IntoIterator<Item = f64>, cpu: &[f64]) -> f64 {
    let gpu: Vec<f64> = gpu.into_iter().collect();
    assert_eq!(gpu.len(), cpu.len());
    gpu.iter()
        .zip(cpu)
        .map(|(g, c)| (g - c).abs() / (1.0 + c.abs()))
        .fold(0.0, f64::max)
}

#[test]
fn roundtrip_is_bit_exact() {
    let Some(device) = device() else { return };
    let host: Vec<f64> = values(1001, 1)
        .into_iter()
        .chain([f64::MIN_POSITIVE, f64::from_bits(1), -0.0, f64::MAX])
        .collect();
    let back = device.download(&device.upload(&host).unwrap()).unwrap();
    assert!(host
        .iter()
        .zip(&back)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    let zeros = device.download(&device.zeros::<f32>(17).unwrap()).unwrap();
    assert_eq!(zeros, vec![0.0f32; 17]);
}

#[test]
fn kernel_cache_hits_memory_then_disk() {
    let Some(device) = device() else { return };
    let dir = std::env::temp_dir().join(format!("omnimd-gpu-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let source = r#"extern "C" __global__ void axpy(double* y, const double* x, double a, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] += a * x[i];
    }"#;
    let first = device.kernels().sibling(Some(dir.clone()));
    let axpy = first.function(source, "axpy").unwrap();
    first.function(source, "axpy").unwrap();
    assert!(first.function(source, "missing").is_err());
    let stats = first.stats();
    assert_eq!(
        (stats.compiled, stats.memory_hits, stats.disk_hits),
        (1, 2, 0)
    );
    let second = device.kernels().sibling(Some(dir.clone()));
    second.function(source, "axpy").unwrap();
    assert_eq!((second.stats().compiled, second.stats().disk_hits), (0, 1));

    use omnimd_gpu::{LaunchConfig, PushKernelArg};
    let x = values(300, 2);
    let mut y_host = values(300, 3);
    let xd = device.upload(&x).unwrap();
    let mut yd = device.upload(&y_host).unwrap();
    let (a, n) = (0.75f64, 300i32);
    let mut launch = device.stream().launch_builder(&axpy);
    launch.arg(&mut yd).arg(&xd).arg(&a).arg(&n);
    unsafe { launch.launch(LaunchConfig::for_num_elems(300)) }.unwrap();
    for (y, x) in y_host.iter_mut().zip(&x) {
        *y += a * x;
    }
    assert!(max_error(device.download(&yd).unwrap(), &y_host) < 1e-15);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn matmul_matches_cpu_in_f64_and_f32() {
    let Some(device) = device() else { return };
    let (m, k, n) = (37, 53, 29);
    let (a, b) = (values(m * k, 4), values(k * n, 5));
    let expected = matmul_cpu(&a, &b, m, k, n);
    let shape = MatmulShape { m, k, n };

    let mut c = device.zeros::<f64>(m * n).unwrap();
    let (ad, bd) = (device.upload(&a).unwrap(), device.upload(&b).unwrap());
    device.matmul(&ad, &bd, &mut c, shape, false).unwrap();
    assert!(max_error(device.download(&c).unwrap(), &expected) < 1e-13);
    // Accumulate: c += a·b doubles it.
    device.matmul(&ad, &bd, &mut c, shape, true).unwrap();
    let doubled: Vec<f64> = expected.iter().map(|v| 2.0 * v).collect();
    assert!(max_error(device.download(&c).unwrap(), &doubled) < 1e-13);

    let a32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
    let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
    let mut c32 = device.zeros::<f32>(m * n).unwrap();
    let (ad, bd) = (device.upload(&a32).unwrap(), device.upload(&b32).unwrap());
    device.matmul(&ad, &bd, &mut c32, shape, false).unwrap();
    let got = device.download(&c32).unwrap().into_iter().map(f64::from);
    assert!(max_error(got, &expected) < 1e-5);

    let mut small = device.zeros::<f64>(m * n - 1).unwrap();
    let (ad, bd) = (device.upload(&a).unwrap(), device.upload(&b).unwrap());
    assert!(device.matmul(&ad, &bd, &mut small, shape, false).is_err());
}

#[test]
fn batched_matmul_matches_each_product() {
    let Some(device) = device() else { return };
    let (m, k, n, batch) = (11, 7, 5, 4);
    let (a, b) = (values(m * k * batch, 6), values(k * n * batch, 7));
    let expected: Vec<f64> = (0..batch)
        .flat_map(|i| matmul_cpu(&a[i * m * k..][..m * k], &b[i * k * n..][..k * n], m, k, n))
        .collect();
    let mut c = device.zeros::<f64>(m * n * batch).unwrap();
    let (ad, bd) = (device.upload(&a).unwrap(), device.upload(&b).unwrap());
    device
        .matmul_batched(&ad, &bd, &mut c, MatmulShape { m, k, n }, batch)
        .unwrap();
    assert!(max_error(device.download(&c).unwrap(), &expected) < 1e-13);
}

/// CPU twin of `Layer::forward` in `native/src/network.rs`, applied per row.
struct CpuLayer {
    input: usize,
    output: usize,
    weights: Vec<f64>,
    bias: Vec<f64>,
    timestep: Option<Vec<f64>>,
    activation: Activation,
    residual: bool,
}

impl CpuLayer {
    fn random(
        input: usize,
        output: usize,
        activation: Activation,
        residual: bool,
        timestep: bool,
        seed: u64,
    ) -> Self {
        Self {
            input,
            output,
            weights: values(input * output, seed),
            bias: values(output, seed + 1),
            timestep: timestep.then(|| {
                values(output, seed + 2)
                    .iter()
                    .map(|v| 0.1 + 0.05 * v)
                    .collect()
            }),
            activation,
            residual,
        }
    }

    fn forward(&self, x: &[f64], rows: usize) -> (Vec<f64>, Vec<f64>) {
        let (mut y, mut slopes) = (vec![0.0; rows * self.output], vec![0.0; rows * self.output]);
        for r in 0..rows {
            let row = &x[r * self.input..][..self.input];
            for j in 0..self.output {
                let z = self.bias[j]
                    + (0..self.input)
                        .map(|i| row[i] * self.weights[i * self.output + j])
                        .sum::<f64>();
                let sigmoid = if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    z.exp() / (1.0 + z.exp())
                };
                let (v, s) = match self.activation {
                    Activation::Linear => (z, 1.0),
                    Activation::Tanh => (z.tanh(), 1.0 - z.tanh() * z.tanh()),
                    Activation::Sigmoid => (sigmoid, sigmoid * (1.0 - sigmoid)),
                    Activation::Softplus => (z.max(0.0) + (-z.abs()).exp().ln_1p(), sigmoid),
                };
                let scale = self.timestep.as_ref().map_or(1.0, |t| t[j]);
                y[r * self.output + j] = v * scale
                    + if self.residual {
                        row[j % self.input]
                    } else {
                        0.0
                    };
                slopes[r * self.output + j] = s * scale;
            }
        }
        (y, slopes)
    }

    fn to_device<T: omnimd_gpu::Scalar>(
        &self,
        device: &Device,
        cast: fn(f64) -> T,
    ) -> DenseLayer<T> {
        let convert = |v: &[f64]| v.iter().map(|&x| cast(x)).collect::<Vec<T>>();
        let timestep = self.timestep.as_deref().map(convert);
        DenseLayer::new(
            device,
            self.input,
            self.output,
            &convert(&self.weights),
            &convert(&self.bias),
            timestep.as_deref(),
            self.activation,
            self.residual,
        )
        .unwrap()
    }
}

#[test]
fn dense_layer_matches_native_semantics() {
    let Some(device) = device() else { return };
    let rows = 333;
    let cases = [
        (8, 8, Activation::Tanh, true, true),
        (8, 16, Activation::Tanh, true, false),
        (8, 5, Activation::Sigmoid, false, false),
        (8, 5, Activation::Softplus, false, true),
        (8, 1, Activation::Linear, false, false),
    ];
    for (seed, &(input, output, activation, residual, timestep)) in cases.iter().enumerate() {
        let layer = CpuLayer::random(
            input,
            output,
            activation,
            residual,
            timestep,
            10 * seed as u64 + 20,
        );
        // Spread pre-activations over both signs and saturation.
        let x: Vec<f64> = values(rows * input, seed as u64 + 90)
            .iter()
            .map(|v| 3.0 * v)
            .collect();
        let (y_cpu, slope_cpu) = layer.forward(&x, rows);
        let label = format!("{input}->{output} {activation:?} residual={residual} dt={timestep}");

        let gpu = layer.to_device(device, |v| v);
        let xd = device.upload(&x).unwrap();
        let mut y = device.zeros::<f64>(rows * output).unwrap();
        let mut slopes = device.zeros::<f64>(rows * output).unwrap();
        gpu.forward(device, &xd, rows, &mut y, Some(&mut slopes))
            .unwrap();
        assert!(
            max_error(device.download(&y).unwrap(), &y_cpu) < 1e-13,
            "f64 y {label}"
        );
        assert!(
            max_error(device.download(&slopes).unwrap(), &slope_cpu) < 1e-13,
            "f64 slope {label}"
        );

        let gpu = layer.to_device(device, |v| v as f32);
        let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let xd = device.upload(&x32).unwrap();
        let mut y = device.zeros::<f32>(rows * output).unwrap();
        gpu.forward(device, &xd, rows, &mut y, None).unwrap();
        let got = device.download(&y).unwrap().into_iter().map(f64::from);
        assert!(max_error(got, &y_cpu) < 1e-5, "f32 y {label}");
    }
}

#[test]
fn embedding_network_matches_cpu_chain() {
    let Some(device) = device() else { return };
    // DeePMD se_e2_a embedding shape: 1->25->50->100, residual where width doubles.
    let layers = [
        CpuLayer::random(1, 25, Activation::Tanh, false, false, 200),
        CpuLayer::random(25, 50, Activation::Tanh, true, false, 210),
        CpuLayer::random(50, 100, Activation::Tanh, true, false, 220),
    ];
    let rows = 4096;
    let x: Vec<f64> = values(rows, 300).iter().map(|v| 0.5 * (v + 1.0)).collect();
    let mut expected = x.clone();
    for layer in &layers {
        expected = layer.forward(&expected, rows).0;
    }
    let network =
        DenseNetwork::new(layers.iter().map(|l| l.to_device(device, |v| v)).collect()).unwrap();
    assert_eq!((network.input(), network.output()), (1, 100));
    let out = network
        .forward(device, &device.upload(&x).unwrap(), rows)
        .unwrap();
    assert!(max_error(device.download(&out).unwrap(), &expected) < 1e-12);

    let mismatched = vec![
        layers[0].to_device(device, |v| v),
        layers[2].to_device(device, |v| v),
    ];
    assert!(DenseNetwork::new(mismatched).is_err());
}

fn scalar_network<T: omnimd_gpu::Scalar>(
    device: &Device,
    layers: &[CpuLayer],
) -> omnimd_gpu::ScalarNetwork<T> {
    let cast = |v: &[f64]| v.iter().map(|&x| T::from_f64(x)).collect::<Vec<T>>();
    let parameters: Vec<_> = layers
        .iter()
        .map(|l| omnimd_gpu::LayerParameters {
            input: l.input,
            output: l.output,
            weights: cast(&l.weights),
            bias: cast(&l.bias),
            timestep: l.timestep.as_deref().map(cast),
            activation: l.activation,
            residual: l.residual,
        })
        .collect();
    omnimd_gpu::ScalarNetwork::new(device, &parameters).unwrap()
}

#[test]
fn fused_scalar_network_matches_cpu_chain_and_its_derivative() {
    let Some(device) = device() else { return };
    // Every layer feature: doubling and same-width residuals, a timestep, all
    // activations, widths that are not multiples of the warp size, and a row
    // count that leaves the last block partly empty.
    let layers = [
        CpuLayer::random(1, 25, Activation::Tanh, false, false, 600),
        CpuLayer::random(25, 50, Activation::Tanh, true, false, 610),
        CpuLayer::random(50, 100, Activation::Softplus, true, false, 620),
        CpuLayer::random(100, 100, Activation::Sigmoid, true, true, 630),
        CpuLayer::random(100, 37, Activation::Linear, false, false, 640),
    ];
    let rows = 1003;
    let x: Vec<f64> = values(rows, 650).iter().map(|v| 0.5 * (v + 1.0)).collect();
    let expected = cpu_chain(&layers, &x, rows);
    let h = 1e-6;
    let shifted = |s: f64| cpu_chain(&layers, &x.iter().map(|v| v + s).collect::<Vec<_>>(), rows);
    let derivative: Vec<f64> = shifted(h)
        .iter()
        .zip(&shifted(-h))
        .map(|(p, m)| (p - m) / (2.0 * h))
        .collect();

    let network = scalar_network::<f64>(device, &layers);
    assert_eq!(network.output(), 37);
    // Poisoned outputs: every element must be written.
    let (mut y, mut dy) = (
        device.upload(&vec![f64::NAN; rows * 37]).unwrap(),
        device.upload(&vec![f64::NAN; rows * 37]).unwrap(),
    );
    network
        .forward_tangent_into(device, &device.upload(&x).unwrap(), rows, &mut y, &mut dy)
        .unwrap();
    assert!(max_error(device.download(&y).unwrap(), &expected) < 1e-12);
    assert!(max_error(device.download(&dy).unwrap(), &derivative) < 1e-6);

    let single = scalar_network::<f32>(device, &layers);
    let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let (mut y, mut dy) = (
        device.zeros::<f32>(rows * 37).unwrap(),
        device.zeros::<f32>(rows * 37).unwrap(),
    );
    single
        .forward_tangent_into(device, &device.upload(&x32).unwrap(), rows, &mut y, &mut dy)
        .unwrap();
    let widen = |v: Vec<f32>| v.into_iter().map(f64::from).collect::<Vec<f64>>();
    assert!(max_error(widen(device.download(&y).unwrap()), &expected) < 2e-4);
    assert!(max_error(widen(device.download(&dy).unwrap()), &derivative) < 2e-3);

    // A network whose first layer does not take one input is rejected.
    let wide = omnimd_gpu::LayerParameters {
        input: 2,
        output: 2,
        weights: vec![0.0f64; 4],
        bias: vec![0.0; 2],
        timestep: None,
        activation: Activation::Tanh,
        residual: false,
    };
    assert!(omnimd_gpu::ScalarNetwork::new(device, &[wide]).is_err());
}

#[test]
fn scalar_table_interpolates_and_flags_inputs_outside_its_range() {
    let Some(device) = device() else { return };
    // f_m(x) = sin((m + 1) x) on [-1, 2], step 0.01.
    let (lower, step, width, knots) = (-1.0, 0.01, 5, 301);
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
    let coefficients =
        omnimd_gpu::quintic_coefficients(step, width, &sample(0), &sample(1), &sample(2)).unwrap();
    let table =
        omnimd_gpu::ScalarTable::<f64>::new(device, lower, step, width, &coefficients).unwrap();
    assert_eq!((table.output(), table.intervals()), (5, 300));

    let rows = 777;
    // Inside the range, including both ends exactly.
    let mut x: Vec<f64> = values(rows, 900).iter().map(|v| 0.5 + 1.5 * v).collect();
    (x[0], x[1]) = (-1.0, 2.0);
    let expected = |x: &[f64], order: usize| -> Vec<f64> {
        x.iter()
            .flat_map(|&x| {
                (0..width).map(move |m| {
                    let k = (m + 1) as f64;
                    if order == 0 {
                        (k * x).sin()
                    } else {
                        k * (k * x).cos()
                    }
                })
            })
            .collect()
    };
    let run = |x: &[f64]| {
        let (mut y, mut dy) = (
            device.upload(&vec![f64::NAN; x.len() * width]).unwrap(),
            device.upload(&vec![f64::NAN; x.len() * width]).unwrap(),
        );
        let mut flag = device.zeros::<i32>(1).unwrap();
        table
            .forward_tangent_into(
                device,
                &device.upload(x).unwrap(),
                x.len(),
                &mut y,
                &mut dy,
                &mut flag,
            )
            .unwrap();
        (
            device.download(&y).unwrap(),
            device.download(&dy).unwrap(),
            device.download(&flag).unwrap()[0],
        )
    };
    let (y, dy, flag) = run(&x);
    assert_eq!(flag, 0);
    // Error ~ step^6 k^6 / 46080 with k up to 5.
    assert!(max_error(y, &expected(&x, 0)) < 1e-11);
    assert!(max_error(dy, &expected(&x, 1)) < 1e-8);

    // One input past either end: flagged, and evaluated at that end.
    for (outside, end) in [(2.5, 2.0), (-1.2, -1.0)] {
        x[5] = outside;
        let (y, _, flag) = run(&x);
        assert_eq!(flag, 1, "input {outside}");
        let at_end = expected(&[end], 0);
        assert!(max_error(y[5 * width..6 * width].to_vec(), &at_end) < 1e-11);
    }

    // The same table continued to 5 by a tail with ten times the step.
    let tail_sample = |order: usize| -> Vec<f64> {
        (0..=30)
            .flat_map(|i| {
                let x = 2.0 + i as f64 * 0.1;
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
    let tail = omnimd_gpu::quintic_coefficients(
        0.1,
        width,
        &tail_sample(0),
        &tail_sample(1),
        &tail_sample(2),
    )
    .unwrap();
    let table = omnimd_gpu::ScalarTable::<f64>::with_tail(
        device,
        lower,
        step,
        width,
        &coefficients,
        0.1,
        &tail,
    )
    .unwrap();
    assert_eq!((table.intervals(), table.tail_intervals()), (300, 30));
    // Main part, the junction, inside the tail and its far end.
    let x = [-0.3, 1.999, 2.0, 2.004, 3.21, 4.95, 5.0];
    let (mut y, mut dy) = (
        device.upload(&vec![f64::NAN; x.len() * width]).unwrap(),
        device.upload(&vec![f64::NAN; x.len() * width]).unwrap(),
    );
    let mut flag = device.zeros::<i32>(1).unwrap();
    let mut evaluate = |x: &[f64]| {
        table
            .forward_tangent_into(
                device,
                &device.upload(x).unwrap(),
                x.len(),
                &mut y,
                &mut dy,
                &mut flag,
            )
            .unwrap();
        (
            device.download(&y).unwrap(),
            device.download(&dy).unwrap(),
            device.download(&flag).unwrap()[0],
        )
    };
    let (values, slopes, flagged) = evaluate(&x);
    assert_eq!(flagged, 0);
    // Tail error ~ 0.1^6 k^6 / 46080 = 3e-7 for k = 5.
    assert!(max_error(values, &expected(&x, 0)) < 1e-6);
    assert!(max_error(slopes, &expected(&x, 1)) < 1e-4);
    // Past the tail: flagged and held at its end.
    let mut beyond = x;
    beyond[3] = 5.5;
    let (values, _, flagged) = evaluate(&beyond);
    assert_eq!(flagged, 1);
    assert!(max_error(values[3 * width..4 * width].to_vec(), &expected(&[5.0], 0)) < 1e-6);
}

fn layer_parameters<T: omnimd_gpu::Scalar>(
    layers: &[CpuLayer],
) -> Vec<omnimd_gpu::LayerParameters<T>> {
    let cast = |v: &[f64]| v.iter().map(|&x| T::from_f64(x)).collect::<Vec<T>>();
    layers
        .iter()
        .map(|l| omnimd_gpu::LayerParameters {
            input: l.input,
            output: l.output,
            weights: cast(&l.weights),
            bias: cast(&l.bias),
            timestep: l.timestep.as_deref().map(cast),
            activation: l.activation,
            residual: l.residual,
        })
        .collect()
}

/// Value and input gradient of a scalar-output chain for one row, on the CPU.
fn cpu_value_and_gradient(layers: &[CpuLayer], x: &[f64]) -> (f64, Vec<f64>) {
    let mut inputs = vec![x.to_vec()];
    let mut slopes = Vec::new();
    for layer in layers {
        let (y, s) = layer.forward(inputs.last().unwrap(), 1);
        inputs.push(y);
        slopes.push(s);
    }
    let mut adjoint = vec![1.0];
    for (layer, slope) in layers.iter().zip(&slopes).rev() {
        let mut previous = vec![0.0; layer.input];
        for (i, p) in previous.iter_mut().enumerate() {
            *p = (0..layer.output)
                .map(|j| adjoint[j] * slope[j] * layer.weights[i * layer.output + j])
                .sum();
        }
        if layer.residual {
            for (j, a) in adjoint.iter().enumerate() {
                previous[j % layer.input] += a;
            }
        }
        adjoint = previous;
    }
    (inputs.last().unwrap()[0], adjoint)
}

#[test]
fn fused_networks_select_one_of_a_set_per_row() {
    let Some(device) = device() else { return };
    // Three embedding nets of one shape with different parameters.
    let nets: Vec<Vec<CpuLayer>> = (0..3)
        .map(|k| {
            vec![
                CpuLayer::random(1, 20, Activation::Tanh, false, false, 700 + 50 * k),
                CpuLayer::random(20, 40, Activation::Tanh, true, false, 710 + 50 * k),
                CpuLayer::random(40, 40, Activation::Tanh, true, true, 720 + 50 * k),
            ]
        })
        .collect();
    let rows = 517;
    let x: Vec<f64> = values(rows, 760).iter().map(|v| 0.5 * (v + 1.0)).collect();
    let ids: Vec<i32> = (0..rows).map(|r| ((r * 7 + r / 5) % 3) as i32).collect();
    let mut expected = vec![0.0; rows * 40];
    for r in 0..rows {
        let y = cpu_chain(&nets[ids[r] as usize], &x[r..r + 1], 1);
        expected[r * 40..(r + 1) * 40].copy_from_slice(&y);
    }
    let parameters: Vec<_> = nets.iter().map(|n| layer_parameters::<f64>(n)).collect();
    let set = omnimd_gpu::ScalarNetwork::new_set(device, &parameters).unwrap();
    assert_eq!((set.networks(), set.output()), (3, 40));
    let (mut y, mut dy) = (
        device.upload(&vec![f64::NAN; rows * 40]).unwrap(),
        device.upload(&vec![f64::NAN; rows * 40]).unwrap(),
    );
    let (xd, idsd) = (device.upload(&x).unwrap(), device.upload(&ids).unwrap());
    set.forward_tangent_by_id(device, &xd, &idsd, rows, &mut y, &mut dy)
        .unwrap();
    assert!(max_error(device.download(&y).unwrap(), &expected) < 1e-12);
    assert!(device.download(&dy).unwrap().iter().all(|v| v.is_finite()));
    // A set needs ids; networks of different shapes do not form a set.
    assert!(set
        .forward_tangent_into(device, &xd, rows, &mut y, &mut dy)
        .is_err());
    let mut odd = parameters.clone();
    odd[1][2].residual = false;
    assert!(omnimd_gpu::ScalarNetwork::new_set(device, &odd).is_err());

    // The same nets as tables: each row is looked up in its own table.
    let (lower, step, knots) = (0.0, 0.01, 101);
    let tables: Vec<omnimd_gpu::TableSegments> = nets
        .iter()
        .map(|net| {
            // Value, slope and curvature at the knots by central differences
            // of the CPU chain; accuracy is not the point of this check.
            let h = 1e-4;
            let at = |x: f64| cpu_chain(net, &[x], 1);
            let (mut v, mut d, mut dd) = (Vec::new(), Vec::new(), Vec::new());
            for k in 0..knots {
                let x = lower + k as f64 * step;
                let (m, c, p) = (at(x - h), at(x), at(x + h));
                for j in 0..40 {
                    v.push(c[j]);
                    d.push((p[j] - m[j]) / (2.0 * h));
                    dd.push((p[j] - 2.0 * c[j] + m[j]) / (h * h));
                }
            }
            omnimd_gpu::TableSegments {
                lower,
                step,
                coefficients: omnimd_gpu::quintic_coefficients(step, 40, &v, &d, &dd).unwrap(),
                tail_step: step,
                tail_coefficients: Vec::new(),
            }
        })
        .collect();
    let table = omnimd_gpu::ScalarTable::<f64>::new_set(device, 40, &tables).unwrap();
    assert_eq!(table.tables(), 3);
    let mut flag = device.zeros::<i32>(1).unwrap();
    table
        .forward_tangent_by_id(device, &xd, &idsd, rows, &mut y, &mut dy, &mut flag)
        .unwrap();
    assert_eq!(device.download(&flag).unwrap()[0], 0);
    // Knot data from finite differences limits this to ~1e-6.
    assert!(max_error(device.download(&y).unwrap(), &expected) < 1e-5);

    // A negative id skips its row: networks and tables leave what was there.
    let skipping: Vec<i32> = ids
        .iter()
        .enumerate()
        .map(|(r, &id)| if r % 3 == 1 { -1 } else { id })
        .collect();
    let skipping_d = device.upload(&skipping).unwrap();
    for (by_table, tolerance) in [(false, 1e-12), (true, 1e-5)] {
        let mut y = device.upload(&vec![7.0f64; rows * 40]).unwrap();
        let mut dy = device.upload(&vec![-7.0f64; rows * 40]).unwrap();
        if by_table {
            table
                .forward_tangent_by_id(device, &xd, &skipping_d, rows, &mut y, &mut dy, &mut flag)
                .unwrap();
        } else {
            set.forward_tangent_by_id(device, &xd, &skipping_d, rows, &mut y, &mut dy)
                .unwrap();
        }
        let (y, dy) = (device.download(&y).unwrap(), device.download(&dy).unwrap());
        for r in 0..rows {
            let (value, tangent) = (&y[r * 40..(r + 1) * 40], &dy[r * 40..(r + 1) * 40]);
            if skipping[r] < 0 {
                assert!(
                    value.iter().all(|&v| v == 7.0) && tangent.iter().all(|&v| v == -7.0),
                    "skipped row {r} was written (tables: {by_table})"
                );
            } else {
                let error = max_error(value.iter().copied(), &expected[r * 40..(r + 1) * 40]);
                assert!(
                    error < tolerance,
                    "row {r} error {error} (tables: {by_table})"
                );
                assert!(tangent.iter().all(|&v| v != -7.0), "row {r} has no tangent");
            }
        }
    }
}

#[test]
fn gradient_network_matches_cpu_value_and_gradient() {
    let Some(device) = device() else { return };
    // Two shapes: every layer feature at small widths, and a fitting-like net
    // wider than a block (so each thread owns several neurons and inputs).
    let small = |seed: u64| {
        vec![
            CpuLayer::random(6, 12, Activation::Tanh, true, false, seed),
            CpuLayer::random(12, 12, Activation::Softplus, true, true, seed + 10),
            CpuLayer::random(12, 3, Activation::Sigmoid, false, false, seed + 20),
            CpuLayer::random(3, 1, Activation::Linear, false, false, seed + 30),
        ]
    };
    let scaled = |mut layer: CpuLayer| {
        // Keep wide layers out of saturation.
        let scale = 1.0 / (layer.input as f64).sqrt();
        layer.weights.iter_mut().for_each(|w| *w *= scale);
        layer
    };
    let wide = |seed: u64| {
        vec![
            scaled(CpuLayer::random(
                600,
                300,
                Activation::Tanh,
                false,
                false,
                seed,
            )),
            scaled(CpuLayer::random(
                300,
                300,
                Activation::Tanh,
                true,
                true,
                seed + 10,
            )),
            scaled(CpuLayer::random(
                300,
                1,
                Activation::Linear,
                false,
                false,
                seed + 20,
            )),
        ]
    };
    for (label, nets, rows) in [
        ("small", vec![small(800), small(900)], 37usize),
        ("wide", vec![wide(1000), wide(1100), wide(1200)], 11),
    ] {
        let input = nets[0][0].input;
        let x = values(rows * input, 1300);
        let ids: Vec<i32> = (0..rows).map(|r| (r % nets.len()) as i32).collect();
        let (mut value, mut gradient) = (Vec::new(), Vec::new());
        for r in 0..rows {
            let (v, g) = cpu_value_and_gradient(&nets[ids[r] as usize], &x[r * input..][..input]);
            value.push(v);
            gradient.extend(g);
        }
        let parameters: Vec<_> = nets.iter().map(|n| layer_parameters::<f64>(n)).collect();
        let network = omnimd_gpu::GradientNetwork::new_set(device, &parameters).unwrap();
        assert_eq!((network.networks(), network.input()), (nets.len(), input));
        let (mut v, mut g) = (
            device.upload(&vec![f64::NAN; rows]).unwrap(),
            device.upload(&vec![f64::NAN; rows * input]).unwrap(),
        );
        let (xd, idsd) = (device.upload(&x).unwrap(), device.upload(&ids).unwrap());
        network
            .forward_gradient_into(device, &xd, Some(&idsd), rows, &mut v, &mut g)
            .unwrap();
        let scale = gradient.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        assert!(
            max_error(device.download(&v).unwrap(), &value) < 1e-12,
            "{label}: value"
        );
        assert!(
            max_error(device.download(&g).unwrap(), &gradient) < 1e-12 * scale,
            "{label}: gradient"
        );

        let single = omnimd_gpu::GradientNetwork::new_set(
            device,
            &nets
                .iter()
                .map(|n| layer_parameters::<f32>(n))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let (mut v, mut g) = (
            device.zeros::<f32>(rows).unwrap(),
            device.zeros::<f32>(rows * input).unwrap(),
        );
        single
            .forward_gradient_into(
                device,
                &device.upload(&x32).unwrap(),
                Some(&idsd),
                rows,
                &mut v,
                &mut g,
            )
            .unwrap();
        let widen = |v: Vec<f32>| v.into_iter().map(f64::from).collect::<Vec<f64>>();
        assert!(
            max_error(widen(device.download(&v).unwrap()), &value) < 1e-4,
            "{label}: f32 value"
        );
        assert!(
            max_error(widen(device.download(&g).unwrap()), &gradient) < 1e-4 * scale,
            "{label}: f32 gradient"
        );
        // A set of several networks needs ids.
        let none: Option<&omnimd_gpu::CudaSlice<i32>> = None;
        assert!(network
            .forward_gradient_into(
                device,
                &xd,
                none,
                rows,
                &mut device.zeros(rows).unwrap(),
                &mut device.zeros(rows * input).unwrap()
            )
            .is_err());
    }
    // More than one output is not a gradient network.
    let vector = vec![CpuLayer::random(4, 2, Activation::Tanh, false, false, 1)];
    assert!(
        omnimd_gpu::GradientNetwork::new_set(device, &[layer_parameters::<f64>(&vector)]).is_err()
    );
}

#[test]
fn transposed_matmul_matches_cpu() {
    let Some(device) = device() else { return };
    let (m, k, n) = (13, 9, 21);
    let (a, bt) = (values(m * k, 40), values(n * k, 41));
    // b[k][n] is the transpose of the stored bt[n][k].
    let b: Vec<f64> = (0..k * n).map(|i| bt[(i % n) * k + i / n]).collect();
    let mut c = device.zeros::<f64>(m * n).unwrap();
    let (ad, btd) = (device.upload(&a).unwrap(), device.upload(&bt).unwrap());
    device
        .matmul_bt(&ad, &btd, &mut c, MatmulShape { m, k, n }, false)
        .unwrap();
    assert!(max_error(device.download(&c).unwrap(), &matmul_cpu(&a, &b, m, k, n)) < 1e-13);
    // Views: the same product on the second half of a larger buffer.
    let mut big = device.zeros::<f64>(2 * m * n).unwrap();
    let mut view = big.slice_mut(m * n..);
    device
        .matmul_bt(&ad, &btd, &mut view, MatmulShape { m, k, n }, false)
        .unwrap();
    let got = device.download(&big).unwrap();
    assert!(got[..m * n].iter().all(|&v| v == 0.0));
    assert!(max_error(got[m * n..].to_vec(), &matmul_cpu(&a, &b, m, k, n)) < 1e-13);
}

fn cpu_chain(layers: &[CpuLayer], x: &[f64], rows: usize) -> Vec<f64> {
    layers
        .iter()
        .fold(x.to_vec(), |h, layer| layer.forward(&h, rows).0)
}

#[test]
fn network_tangent_and_backward_match_finite_differences() {
    let Some(device) = device() else { return };
    let layers = [
        CpuLayer::random(6, 12, Activation::Tanh, true, false, 500),
        CpuLayer::random(12, 12, Activation::Softplus, true, true, 510),
        CpuLayer::random(12, 3, Activation::Sigmoid, false, false, 520),
        CpuLayer::random(3, 1, Activation::Linear, false, false, 530),
    ];
    let rows = 50;
    let x = values(rows * 6, 540);
    let network =
        DenseNetwork::new(layers.iter().map(|l| l.to_device(device, |v| v)).collect()).unwrap();
    let xd = device.upload(&x).unwrap();
    let h = 1e-6;

    // Tangent along a random direction.
    let direction = values(rows * 6, 541);
    let (mut y, mut dy) = (
        device.zeros::<f64>(rows).unwrap(),
        device.zeros::<f64>(rows).unwrap(),
    );
    network
        .forward_tangent_into(
            device,
            &xd,
            &device.upload(&direction).unwrap(),
            rows,
            &mut y,
            &mut dy,
        )
        .unwrap();
    let shifted = |s: f64| -> Vec<f64> {
        let moved: Vec<f64> = x.iter().zip(&direction).map(|(x, d)| x + s * d).collect();
        cpu_chain(&layers, &moved, rows)
    };
    let (plus, minus) = (shifted(h), shifted(-h));
    let fd: Vec<f64> = plus
        .iter()
        .zip(&minus)
        .map(|(p, m)| (p - m) / (2.0 * h))
        .collect();
    assert!(max_error(device.download(&y).unwrap(), &cpu_chain(&layers, &x, rows)) < 1e-13);
    assert!(
        max_error(device.download(&dy).unwrap(), &fd) < 1e-7,
        "tangent"
    );

    // Backward of L = sum_r w_r y_r gives dL/dx row by row.
    let weights = values(rows, 542);
    let tape = network.forward_record(device, &xd, rows).unwrap();
    let mut dx = device.zeros::<f64>(rows * 6).unwrap();
    network
        .backward(device, &tape, &device.upload(&weights).unwrap(), &mut dx)
        .unwrap();
    let loss = |x: &[f64]| -> f64 {
        cpu_chain(&layers, x, rows)
            .iter()
            .zip(&weights)
            .map(|(y, w)| y * w)
            .sum()
    };
    let fd: Vec<f64> = (0..x.len())
        .map(|i| {
            let (mut p, mut m) = (x.clone(), x.clone());
            p[i] += h;
            m[i] -= h;
            (loss(&p) - loss(&m)) / (2.0 * h)
        })
        .collect();
    assert!(
        max_error(
            device.download(&tape.output).unwrap(),
            &cpu_chain(&layers, &x, rows)
        ) < 1e-13
    );
    assert!(
        max_error(device.download(&dx).unwrap(), &fd) < 1e-7,
        "backward"
    );
}

#[test]
fn grouped_rows_take_the_tiled_first_layer() {
    let Some(device) = device() else { return };
    let scaled = |mut layer: CpuLayer| {
        // Keep wide layers out of saturation.
        let scale = 1.0 / (layer.input as f64).sqrt();
        layer.weights.iter_mut().for_each(|w| *w *= scale);
        layer
    };
    // Widths that are no multiple of a warp and inputs that are no multiple
    // of a tile; one net wider than eight warps, one with four layers.
    let wide = |seed: u64| {
        vec![
            scaled(CpuLayer::random(
                600,
                300,
                Activation::Tanh,
                false,
                false,
                seed,
            )),
            scaled(CpuLayer::random(
                300,
                300,
                Activation::Tanh,
                true,
                true,
                seed + 10,
            )),
            scaled(CpuLayer::random(
                300,
                1,
                Activation::Linear,
                false,
                false,
                seed + 20,
            )),
        ]
    };
    let deep = |seed: u64| {
        vec![
            scaled(CpuLayer::random(
                1100,
                70,
                Activation::Tanh,
                false,
                true,
                seed,
            )),
            scaled(CpuLayer::random(
                70,
                70,
                Activation::Softplus,
                true,
                false,
                seed + 10,
            )),
            scaled(CpuLayer::random(
                70,
                35,
                Activation::Tanh,
                false,
                false,
                seed + 20,
            )),
            scaled(CpuLayer::random(
                35,
                1,
                Activation::Linear,
                false,
                false,
                seed + 30,
            )),
        ]
    };
    for (label, nets) in [
        ("wide", vec![wide(2000), wide(2100), wide(2200)]),
        ("deep", vec![deep(2300), deep(2400), deep(2500)]),
    ] {
        let input = nets[0][0].input;
        // Runs longer than a chunk, of odd length, of one row, and not in
        // the order of the networks; network 1 has no rows in the second.
        let groupings: [&[(usize, usize, usize)]; 6] = [
            &[(0, 0, 37), (2, 37, 1), (1, 38, 50)],
            &[(2, 0, 45), (0, 45, 43)],
            &[(1, 0, 88)],
            &[(0, 0, 1), (1, 1, 1), (2, 2, 86)],
            &[(2, 0, 33), (1, 33, 55)],
            &[(0, 0, 37), (2, 37, 1), (1, 38, 50)],
        ];
        let rows = 88;
        let x = values(rows * input, 2600);
        let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let (xd, xd32) = (device.upload(&x).unwrap(), device.upload(&x32).unwrap());
        let double: Vec<_> = nets.iter().map(|n| layer_parameters::<f64>(n)).collect();
        let single: Vec<_> = nets.iter().map(|n| layer_parameters::<f32>(n)).collect();
        let mut network = omnimd_gpu::GradientNetwork::new_set(device, &double).unwrap();
        let mut network32 = omnimd_gpu::GradientNetwork::new_set(device, &single).unwrap();
        // Tiles are the default in float32 for first layers this large, and
        // on request in float64.
        assert!(network32.tiled() && !network.tiled(), "{label}");
        assert!(network.set_tiled(true), "{label}");
        let widen = |v: Vec<f32>| v.into_iter().map(f64::from).collect::<Vec<f64>>();
        for groups in groupings {
            let (mut value, mut gradient) = (Vec::new(), Vec::new());
            for &(net, first, count) in groups {
                for r in first..first + count {
                    let (v, g) = cpu_value_and_gradient(&nets[net], &x[r * input..][..input]);
                    value.push(v);
                    gradient.extend(g);
                }
            }
            let scale = gradient.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            for tiled in [true, false] {
                assert_eq!(network.set_tiled(tiled), tiled);
                let mut v = device.upload(&vec![f64::NAN; rows]).unwrap();
                let mut g = device.upload(&vec![f64::NAN; rows * input]).unwrap();
                network
                    .forward_gradient_grouped(device, &xd, groups, rows, &mut v, &mut g)
                    .unwrap();
                let what = format!("{label} {groups:?} tiled {tiled}");
                let error = max_error(device.download(&v).unwrap(), &value);
                assert!(error < 1e-12, "{what}: value error {error}");
                let error = max_error(device.download(&g).unwrap(), &gradient);
                assert!(error < 1e-12 * scale, "{what}: gradient error {error}");
            }
            let mut v = device.upload(&vec![f32::NAN; rows]).unwrap();
            let mut g = device.upload(&vec![f32::NAN; rows * input]).unwrap();
            network32
                .forward_gradient_grouped(device, &xd32, groups, rows, &mut v, &mut g)
                .unwrap();
            let error = max_error(widen(device.download(&v).unwrap()), &value);
            assert!(
                error < 1e-4,
                "{label} {groups:?}: float32 value error {error}"
            );
            let error = max_error(widen(device.download(&g).unwrap()), &gradient);
            assert!(
                error < 1e-4 * scale,
                "{label} {groups:?}: float32 gradient error {error}"
            );
        }
        // Groups have to be consecutive nonempty runs of known networks
        // that cover the rows.
        let mut v = device.zeros::<f64>(rows).unwrap();
        let mut g = device.zeros::<f64>(rows * input).unwrap();
        let bad: [&[(usize, usize, usize)]; 7] = [
            &[(0, 0, 40), (1, 41, 47)],
            &[(0, 0, 40), (1, 39, 49)],
            // The right number of rows, but the second run starts early.
            &[(0, 0, 40), (1, 39, 48)],
            &[(0, 0, 88), (1, 88, 0)],
            &[(3, 0, 88)],
            &[(0, 0, 87)],
            &[],
        ];
        for groups in bad {
            assert!(
                network
                    .forward_gradient_grouped(device, &xd, groups, rows, &mut v, &mut g)
                    .is_err(),
                "{label}: {groups:?} was accepted"
            );
        }
    }
    // A first layer with a residual keeps the one kernel: it needs its
    // input row again. Grouped rows still work.
    let residual = vec![
        CpuLayer::random(6, 12, Activation::Tanh, true, false, 2700),
        CpuLayer::random(12, 1, Activation::Linear, false, false, 2710),
    ];
    let mut network =
        omnimd_gpu::GradientNetwork::new_set(device, &[layer_parameters::<f64>(&residual)])
            .unwrap();
    assert!(!network.set_tiled(true));
    let x = values(5 * 6, 2720);
    let (mut v, mut g) = (
        device.zeros::<f64>(5).unwrap(),
        device.zeros::<f64>(30).unwrap(),
    );
    network
        .forward_gradient_grouped(
            device,
            &device.upload(&x).unwrap(),
            &[(0, 0, 5)],
            5,
            &mut v,
            &mut g,
        )
        .unwrap();
    let expected: Vec<f64> = (0..5)
        .map(|r| cpu_value_and_gradient(&residual, &x[6 * r..][..6]).0)
        .collect();
    assert!(max_error(device.download(&v).unwrap(), &expected) < 1e-12);
}

/// Brute-force reference: every image within `rcut`, sorted like the GPU
/// list, truncated per type. Returns (index, delta) per slot and full counts.
fn neighbors_cpu(
    positions: &[[f64; 3]],
    types: &[i32],
    cell: &Cell,
    rcut: f64,
    sel: &[usize],
) -> (Vec<i32>, Vec<f64>, Vec<i32>, f64) {
    let images = match cell {
        Cell::Open => 0,
        Cell::Periodic(_) => 3,
    };
    let slots: usize = sel.iter().sum();
    let (mut index, mut delta) = (
        vec![-1; positions.len() * slots],
        vec![0.0; positions.len() * slots * 3],
    );
    let mut counts = vec![0; positions.len() * sel.len()];
    let mut nearest = f64::INFINITY;
    for (i, pi) in positions.iter().enumerate() {
        let mut found: Vec<(f64, usize, [f64; 3])> = Vec::new();
        for (j, pj) in positions.iter().enumerate() {
            for a in -images..=images {
                for b in -images..=images {
                    for c in -images..=images {
                        if i == j && (a, b, c) == (0, 0, 0) {
                            continue;
                        }
                        let shift: [f64; 3] = match cell {
                            Cell::Open => [0.0; 3],
                            Cell::Periodic(l) => std::array::from_fn(|k| {
                                a as f64 * l[0][k] + b as f64 * l[1][k] + c as f64 * l[2][k]
                            }),
                        };
                        let d: [f64; 3] = std::array::from_fn(|k| pj[k] + shift[k] - pi[k]);
                        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        if r < rcut {
                            found.push((r, j, d));
                            counts[i * sel.len() + types[j] as usize] += 1;
                            nearest = nearest.min(r);
                        }
                    }
                }
            }
        }
        // DeePMD-kit 3 selection: the `slots` nearest of any type first,
        // then at most sel[t] of each type; equal distances in the order of
        // the atoms, images of one atom in the order of their deltas.
        found.sort_by(|x, y| {
            x.0.total_cmp(&y.0)
                .then(x.1.cmp(&y.1))
                .then(x.2.partial_cmp(&y.2).unwrap())
        });
        found.truncate(slots);
        let mut slot = i * slots;
        for (t, &room) in sel.iter().enumerate() {
            let of_type = found.iter().filter(|e| types[e.1] as usize == t);
            for (k, (_, j, d)) in of_type.take(room).enumerate() {
                index[slot + k] = *j as i32;
                delta[(slot + k) * 3..][..3].copy_from_slice(d);
            }
            slot += room;
        }
    }
    (index, delta, counts, nearest)
}

fn check_neighbors(
    label: &str,
    positions: &[[f64; 3]],
    types: &[i32],
    cell: Cell,
    rcut: f64,
    sel: &[usize],
) {
    // Default warp path, all-serial fallback, and a mix of both.
    for capacity in [None, Some(0), Some(7)] {
        let label = format!("{label} (capacity {capacity:?})");
        check_neighbors_with(&label, positions, types, cell, rcut, sel, capacity);
    }
}

fn check_neighbors_with(
    label: &str,
    positions: &[[f64; 3]],
    types: &[i32],
    cell: Cell,
    rcut: f64,
    sel: &[usize],
    capacity: Option<usize>,
) {
    let device = device().unwrap();
    let list =
        NeighborList::build_with_capacity(device, positions, types, &cell, rcut, sel, capacity)
            .unwrap();
    compare_list(label, &list, positions, types, &cell, rcut, sel);
}

/// Checks a list against the brute-force reference.
fn compare_list(
    label: &str,
    list: &NeighborList,
    positions: &[[f64; 3]],
    types: &[i32],
    cell: &Cell,
    rcut: f64,
    sel: &[usize],
) {
    let device = device().unwrap();
    let (index, delta, counts, nearest) = neighbors_cpu(positions, types, cell, rcut, sel);
    assert_eq!(
        device.download(&list.types).unwrap(),
        types,
        "{label}: types"
    );
    assert_eq!(
        device.download(&list.count).unwrap(),
        counts,
        "{label}: counts"
    );
    let (got_index, got_delta, got_dist) = (
        device.download(&list.index).unwrap(),
        device.download(&list.delta).unwrap(),
        device.download(&list.dist).unwrap(),
    );
    // Images of one atom at the same distance (small boxes) differ only by
    // rounding, so compare each (center, type) segment as a set and check
    // the GPU order is nondecreasing in distance.
    let slots = list.slots;
    let mut start = 0;
    for center in 0..positions.len() {
        for &width in sel {
            let segment = center * slots + start..center * slots + start + width;
            let entries = |idx: &[i32], del: &[f64]| {
                let mut e: Vec<(i32, [f64; 3])> = segment
                    .clone()
                    .map(|s| (idx[s], [del[3 * s], del[3 * s + 1], del[3 * s + 2]]))
                    .collect();
                e.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
                e
            };
            let (gpu, cpu) = (entries(&got_index, &got_delta), entries(&index, &delta));
            for (g, c) in gpu.iter().zip(&cpu) {
                assert_eq!(g.0, c.0, "{label}: atom {center} neighbor set");
                let error = (0..3).map(|a| (g.1[a] - c.1[a]).abs()).fold(0.0, f64::max);
                assert!(error < 1e-11, "{label}: atom {center} delta error {error}");
            }
            let radii: Vec<f64> = segment
                .clone()
                .filter(|&s| got_index[s] >= 0)
                .map(|s| {
                    (0..3)
                        .map(|a| got_delta[3 * s + a].powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .collect();
            assert!(
                radii.windows(2).all(|w| w[0] <= w[1] + 1e-12),
                "{label}: atom {center} not sorted"
            );
            // The stored distance is the length of the delta; empty slots
            // hold infinity.
            let filled = radii.len();
            for (k, s) in segment.clone().enumerate() {
                let stored = got_dist[s];
                if k < filled {
                    assert!(
                        (stored - radii[k]).abs() <= 1e-12 * radii[k].max(1.0),
                        "{label}: atom {center} distance {stored} against {}",
                        radii[k]
                    );
                } else {
                    assert!(
                        got_index[s] < 0 && stored == f64::INFINITY,
                        "{label}: atom {center} has a hole or a distance in an empty slot"
                    );
                }
            }
            start += width;
        }
        start = 0;
    }
    let stats = list.stats(device).unwrap();
    assert!(
        (stats.min_distance - nearest).abs() < 1e-12,
        "{label}: min distance"
    );
    let truncated = counts
        .iter()
        .enumerate()
        .any(|(k, &c)| c as usize > sel[k % sel.len()]);
    assert_eq!(
        list.truncated(device).unwrap(),
        truncated,
        "{label}: truncation flag"
    );
}
fn random_frame(natoms: usize, lattice: [[f64; 3]; 3], seed: u64) -> (Vec<[f64; 3]>, Vec<i32>) {
    let f = values(natoms * 3, seed);
    let positions = (0..natoms)
        .map(|i| {
            // Fractions in (-0.5, 1.5): some atoms start outside the cell.
            let frac: [f64; 3] = std::array::from_fn(|a| 0.5 + f[3 * i + a]);
            std::array::from_fn(|k| (0..3).map(|a| frac[a] * lattice[a][k]).sum())
        })
        .collect();
    let types = (0..natoms).map(|i| (i % 3 == 0) as i32).collect();
    (positions, types)
}

#[test]
fn neighbor_list_matches_brute_force() {
    if device().is_none() {
        return;
    }
    let cubic = [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]];
    let (p, t) = random_frame(300, cubic, 700);
    check_neighbors("cubic", &p, &t, Cell::Periodic(cubic), 4.0, &[40, 80]);

    let triclinic = [[11.0, 0.0, 0.0], [3.0, 10.0, 0.0], [-2.0, 1.5, 9.5]];
    let (p, t) = random_frame(250, triclinic, 701);
    check_neighbors(
        "triclinic",
        &p,
        &t,
        Cell::Periodic(triclinic),
        4.5,
        &[60, 120],
    );

    // Box narrower than the cutoff in one direction: several images per pair.
    let slab = [[3.5, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]];
    let (p, t) = random_frame(60, slab, 702);
    check_neighbors("small box", &p, &t, Cell::Periodic(slab), 5.0, &[80, 160]);

    let (p, t) = random_frame(200, cubic, 703);
    check_neighbors("open", &p, &t, Cell::Open, 4.0, &[60, 120]);

    // Tight sel: farthest neighbors are dropped, nearest kept, counts still full.
    let (p, t) = random_frame(300, cubic, 704);
    check_neighbors("truncated", &p, &t, Cell::Periodic(cubic), 4.0, &[5, 9]);
}

#[test]
fn neighbor_search_gives_a_fresh_list_for_every_frame() {
    let Some(device) = device() else { return };
    let (rcut, sel) = (4.0, [40usize, 80]);
    let cubic = [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]];
    // 392 cells against the 27 of the cubic box.
    let wide = [[31.0, 0.0, 0.0], [4.0, 29.0, 0.0], [-3.0, 2.0, 33.0]];
    // 1200 cells, as many as 300 atoms can get: the largest input.
    let sparse = [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 48.0]];
    let (first, types) = random_frame(300, cubic, 800);
    let moved: Vec<[f64; 3]> = first
        .iter()
        .zip(values(900, 801).chunks(3))
        .map(|(p, d)| std::array::from_fn(|a| p[a] + 0.3 * d[a]))
        .collect();
    let (spread, _) = random_frame(300, wide, 802);
    let (thin, _) = random_frame(300, sparse, 804);
    let swapped: Vec<i32> = types.iter().map(|t| 1 - t).collect();
    let (fewer, fewer_types) = random_frame(120, cubic, 803);
    let frames: [(&str, &[[f64; 3]], &[i32], Cell); 8] = [
        ("first", &first, &types, Cell::Periodic(cubic)),
        ("moved", &moved, &types, Cell::Periodic(cubic)),
        ("larger cell", &spread, &types, Cell::Periodic(wide)),
        ("most cells", &thin, &types, Cell::Periodic(sparse)),
        ("other types", &first, &swapped, Cell::Periodic(cubic)),
        ("fewer atoms", &fewer, &fewer_types, Cell::Periodic(cubic)),
        ("open", &first, &types, Cell::Open),
        ("first again", &first, &types, Cell::Periodic(cubic)),
    ];
    let mut search = NeighborSearch::new(rcut, &sel).unwrap();
    assert!(search.list().is_none());
    for (label, positions, types, cell) in &frames {
        let list = search.run(device, positions, types, cell).unwrap();
        let label = format!("reused search, {label}");
        compare_list(&label, list, positions, types, cell, rcut, &sel);
    }
    // The serial path writes through the same buffers.
    let cell = Cell::Periodic(cubic);
    let list = search
        .run_with_capacity(device, &moved, &types, &cell, Some(0))
        .unwrap();
    compare_list(
        "reused search, serial",
        list,
        &moved,
        &types,
        &cell,
        rcut,
        &sel,
    );
    // A frame the search refuses leaves the last list as it was.
    assert!(search.run(device, &first, &types[..10], &cell).is_err());
    assert!(search.run(device, &first, &vec![2; 300], &cell).is_err());
    let mut bad = first.clone();
    bad[7][1] = f64::NAN;
    assert!(search.run(device, &bad, &types, &cell).is_err());
    let list = search.list().unwrap();
    compare_list("after refusals", list, &moved, &types, &cell, rcut, &sel);
    assert_eq!(search.into_list().unwrap().natoms, 300);
    assert!(NeighborSearch::new(0.0, &sel).is_err());
    assert!(NeighborSearch::new(f64::INFINITY, &sel).is_err());
    assert!(NeighborSearch::new(4.0, &[0, 0]).is_err());
}

#[test]
fn neighbor_status_is_mirrored_per_center() {
    let Some(device) = device() else { return };
    // Four atoms in a row 1 Å apart and one on its own, room for two
    // neighbors each: the inner two atoms have three within 2.5 Å.
    let mut positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ];
    let types = [0; 5];
    let mut search = NeighborSearch::new(2.5, &[2]).unwrap();
    // The mirror may be part of a larger buffer of the caller.
    let mut buffer = device.upload(&[-1i32; 9]).unwrap();
    let list = search
        .run_mirrored(
            device,
            &positions,
            &types,
            &Cell::Open,
            &mut buffer.slice_mut(2..7),
        )
        .unwrap();
    assert!(list.check(device).unwrap(), "truncation is reported");
    assert_eq!(
        device.download(&buffer).unwrap(),
        [-1, -1, 0, TRUNCATED, TRUNCATED, 0, 0, -1, -1]
    );
    // The last atom onto the first: both see an atom at no distance.
    positions[4] = positions[0];
    let list = search
        .run_mirrored(
            device,
            &positions,
            &types,
            &Cell::Open,
            &mut buffer.slice_mut(2..7),
        )
        .unwrap();
    assert!(list.check(device).is_err(), "coincident atoms are an error");
    let status = device.download(&buffer).unwrap();
    assert_eq!(status[2] & COINCIDENT, COINCIDENT);
    assert_eq!(status[6] & COINCIDENT, COINCIDENT);
    assert_eq!(status[3..6], [TRUNCATED, TRUNCATED, 0]);
    // Reading the bits: any coincidence is an error, else truncation.
    assert!(!NeighborList::status(0).unwrap());
    assert!(NeighborList::status(TRUNCATED).unwrap());
    assert!(NeighborList::status(COINCIDENT).is_err());
    assert!(NeighborList::status(COINCIDENT | TRUNCATED).is_err());
    // A mirror of another length is refused.
    assert!(search
        .run_mirrored(device, &positions, &types, &Cell::Open, &mut buffer)
        .is_err());
}

#[test]
fn neighbors_at_the_cutoff_survive_the_single_precision_scan() {
    if device().is_none() {
        return;
    }
    // Pairs a part in 1e9 inside and outside the cutoff, at coordinates
    // where single precision resolves no better than 1e-5 to 1e-2 Å: the
    // first pass of the search cannot tell them apart and has to hand all
    // of them to the pass in double precision.
    let rcut = 5.0;
    let pairs = |origin: [f64; 3], spacing: f64, count: usize, seed: u64| {
        let v = values(3 * count, seed);
        let mut positions = Vec::new();
        for pair in 0..count {
            let direction: [f64; 3] = std::array::from_fn(|a| v[3 * pair + a]);
            let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
            let sign = if pair % 2 == 0 { -1.0 } else { 1.0 };
            let length = rcut * (1.0 + sign * 1e-9);
            let first: [f64; 3] =
                std::array::from_fn(|k| origin[k] + f64::from(k == 0) * spacing * pair as f64);
            positions.push(first);
            positions.push(std::array::from_fn(|k| {
                first[k] + direction[k] / norm * length
            }));
        }
        let types = vec![0; positions.len()];
        (positions, types)
    };
    // Far from the origin in an open cell.
    let (positions, types) = pairs([3.0e5, -2.0e5, 1.0e5], 20.0, 40, 900);
    let label = "pairs at the cutoff, far away";
    check_neighbors(label, &positions, &types, Cell::Open, rcut, &[4]);
    // In a long periodic cell, some pairs across its faces.
    let long = [[400.0, 0.0, 0.0], [7.0, 31.0, 0.0], [0.0, -3.0, 29.0]];
    let (positions, types) = pairs([1.0, 2.0, 3.0], 20.0, 20, 901);
    let label = "pairs at the cutoff, periodic";
    check_neighbors(label, &positions, &types, Cell::Periodic(long), rcut, &[4]);
}

#[test]
fn equidistant_neighbors_are_ranked_by_atom_then_delta() {
    if device().is_none() {
        return;
    }
    // A simple cubic lattice in exact coordinates: six neighbors at exactly
    // 2 Å and room for four. Which four are kept is decided by the atom
    // index alone.
    let lattice = [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]];
    let positions: Vec<[f64; 3]> = (0..27)
        .map(|i| {
            [
                2.0 * (i % 3) as f64,
                2.0 * (i / 3 % 3) as f64,
                2.0 * (i / 9) as f64,
            ]
        })
        .collect();
    let types = vec![0; 27];
    check_neighbors(
        "cubic lattice",
        &positions,
        &types,
        Cell::Periodic(lattice),
        2.5,
        &[4],
    );
    // Room for exactly the six: full, and nothing dropped.
    check_neighbors(
        "full shell",
        &positions,
        &types,
        Cell::Periodic(lattice),
        2.5,
        &[6],
    );
    // Two kinds of atoms in layers: ties within each type, and a type with
    // room for all its neighbors next to one without.
    let layered: Vec<i32> = (0..27).map(|i| (i / 9 % 2) as i32).collect();
    check_neighbors(
        "layered lattice",
        &positions,
        &layered,
        Cell::Periodic(lattice),
        2.5,
        &[3, 6],
    );
    // One atom in a 2 Å box: the six are images of the atom itself and are
    // told apart by their deltas. Twelve more at 2 sqrt(2) Å, room for ten.
    let single = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
    let atom = [[0.5, 0.25, 1.0]];
    check_neighbors(
        "one atom and its images",
        &atom,
        &[0],
        Cell::Periodic(single),
        2.5,
        &[4],
    );
    check_neighbors(
        "two shells of images",
        &atom,
        &[0],
        Cell::Periodic(single),
        3.0,
        &[10],
    );
}

/// Packs CPU layers into a flat parameter vector and their training layout.
fn training_layout(layers: &[CpuLayer]) -> (Vec<f64>, Vec<omnimd_gpu::TrainLayer>) {
    let mut theta = Vec::new();
    let mut layout = Vec::new();
    for layer in layers {
        let weights = theta.len();
        theta.extend_from_slice(&layer.weights);
        let bias = theta.len();
        theta.extend_from_slice(&layer.bias);
        let timestep = layer.timestep.as_ref().map(|t| {
            let offset = theta.len();
            theta.extend_from_slice(t);
            offset
        });
        layout.push(omnimd_gpu::TrainLayer {
            input: layer.input,
            output: layer.output,
            activation: layer.activation,
            residual: layer.residual,
            weights,
            bias,
            timestep,
        });
    }
    (theta, layout)
}

#[test]
fn training_network_gradients_match_finite_differences() {
    let Some(device) = device() else { return };
    let layers = [
        CpuLayer::random(3, 6, Activation::Tanh, true, false, 1400),
        CpuLayer::random(6, 6, Activation::Softplus, true, true, 1410),
        CpuLayer::random(6, 4, Activation::Sigmoid, false, false, 1420),
        CpuLayer::random(4, 2, Activation::Tanh, false, true, 1430),
        CpuLayer::random(2, 1, Activation::Linear, false, false, 1440),
    ];
    let (theta, layout) = training_layout(&layers);
    let network = omnimd_gpu::TrainNetwork::<f64>::new(device, layout.clone()).unwrap();
    assert_eq!((network.input(), network.output()), (3, 1));
    let rows = 9;
    let (x, xdot) = (values(rows * 3, 1450), values(rows * 3, 1451));
    // Phi = sum_r (wy_r y_r + wd_r ydot_r): depends on values and tangents.
    let (wy, wd) = (values(rows, 1452), values(rows, 1453));
    let phi = |theta: &[f64], x: &[f64], xdot: &[f64]| -> f64 {
        let tape = network
            .forward(
                device,
                &device.upload(theta).unwrap(),
                &device.upload(x).unwrap(),
                &device.upload(xdot).unwrap(),
                rows,
            )
            .unwrap();
        let y = device.download(tape.output()).unwrap();
        let ydot = device.download(tape.tangent()).unwrap();
        (0..rows).map(|r| wy[r] * y[r] + wd[r] * ydot[r]).sum()
    };

    // The forward values agree with the CPU chain, and the tangent with its
    // directional finite difference.
    let theta_d = device.upload(&theta).unwrap();
    let (xd, xdotd) = (device.upload(&x).unwrap(), device.upload(&xdot).unwrap());
    let tape = network
        .forward(device, &theta_d, &xd, &xdotd, rows)
        .unwrap();
    assert!(
        max_error(
            device.download(tape.output()).unwrap(),
            &cpu_chain(&layers, &x, rows)
        ) < 1e-13
    );
    let h = 1e-6;
    let moved = |s: f64| -> Vec<f64> { x.iter().zip(&xdot).map(|(a, b)| a + s * b).collect() };
    let directional: Vec<f64> = cpu_chain(&layers, &moved(h), rows)
        .iter()
        .zip(&cpu_chain(&layers, &moved(-h), rows))
        .map(|(p, m)| (p - m) / (2.0 * h))
        .collect();
    assert!(max_error(device.download(tape.tangent()).unwrap(), &directional) < 1e-8);

    // Reverse pass: parameter gradient and input adjoints.
    let mut gradient = device.zeros::<f64>(theta.len()).unwrap();
    let (mut xbar, mut xdbar) = (
        device.zeros::<f64>(rows * 3).unwrap(),
        device.zeros::<f64>(rows * 3).unwrap(),
    );
    let ones = device.upload(&vec![1.0f64; rows]).unwrap();
    network
        .backward_to_input(
            device,
            &theta_d,
            &mut gradient,
            true,
            &tape,
            &xd,
            &xdotd,
            &device.upload(&wy).unwrap(),
            &device.upload(&wd).unwrap(),
            &ones,
            &mut xbar,
            &mut xdbar,
        )
        .unwrap();
    let central = |f: &dyn Fn(f64) -> f64| (f(h) - f(-h)) / (2.0 * h);
    let analytic = device.download(&gradient).unwrap();
    for k in 0..theta.len() {
        let numeric = central(&|s| {
            let mut t = theta.clone();
            t[k] += s;
            phi(&t, &x, &xdot)
        });
        assert!(
            (analytic[k] - numeric).abs() < 1e-7 * numeric.abs().max(1.0),
            "parameter {k}: {} vs {numeric}",
            analytic[k]
        );
    }
    let (got_x, got_xdot) = (
        device.download(&xbar).unwrap(),
        device.download(&xdbar).unwrap(),
    );
    for k in 0..rows * 3 {
        let numeric = central(&|s| {
            let mut moved = x.clone();
            moved[k] += s;
            phi(&theta, &moved, &xdot)
        });
        assert!(
            (got_x[k] - numeric).abs() < 1e-7 * numeric.abs().max(1.0),
            "x {k}"
        );
        let numeric = central(&|s| {
            let mut moved = xdot.clone();
            moved[k] += s;
            phi(&theta, &x, &moved)
        });
        assert!(
            (got_xdot[k] - numeric).abs() < 1e-7 * numeric.abs().max(1.0),
            "xdot {k}"
        );
    }

    // Without input adjoints the parameter gradient is the same, and it
    // accumulates into what the buffer already holds.
    network
        .backward(
            device,
            &theta_d,
            &mut gradient,
            &tape,
            &xd,
            &xdotd,
            &device.upload(&wy).unwrap(),
            &device.upload(&wd).unwrap(),
            &ones,
        )
        .unwrap();
    let doubled: Vec<f64> = analytic.iter().map(|g| 2.0 * g).collect();
    assert!(max_error(device.download(&gradient).unwrap(), &doubled) < 1e-12);

    // Layers that do not chain are rejected.
    let mut broken = layout;
    broken[1].input = 5;
    assert!(omnimd_gpu::TrainNetwork::<f64>::new(device, broken).is_err());
}

/// A set of same-shaped networks against its networks evaluated one by one:
/// outputs, tangents, parameter gradient and input adjoints, for inputs
/// `x[rows][input]` with tangents `xdot`.
fn compare_training_set<T: omnimd_gpu::Scalar>(
    device: &Device,
    theta: &[f64],
    layouts: &[Vec<omnimd_gpu::TrainLayer>],
    x: &[f64],
    xdot: &[f64],
    blocks: &[(usize, usize, usize)],
    tolerance: f64,
) {
    let cast = |values: &[f64]| -> Vec<T> { values.iter().map(|&v| T::from_f64(v)).collect() };
    let widen = |values: Vec<T>| -> Vec<f64> { values.into_iter().map(T::to_f64).collect() };
    let (input, width) = (layouts[0][0].input, layouts[0].last().unwrap().output);
    let rows = x.len() / input;
    let (ybar, ydotbar) = (values(rows * width, 2501), values(rows * width, 2502));
    let theta_d = device.upload(&cast(theta)).unwrap();
    let ones = device.upload(&vec![T::ONE; rows]).unwrap();
    let mut ids = vec![0i32; rows];
    let (mut expected, mut expected_tangent) = (vec![0.0; rows * width], vec![0.0; rows * width]);
    let (mut expected_xbar, mut expected_xdbar) =
        (vec![0.0; rows * input], vec![0.0; rows * input]);
    let mut reference = device.zeros::<T>(theta.len()).unwrap();
    for &(net, first, count) in blocks {
        ids[first..first + count].fill(net as i32);
        let network = omnimd_gpu::TrainNetwork::<T>::new(device, layouts[net].clone()).unwrap();
        let (inputs, outputs) = (
            first * input..(first + count) * input,
            first * width..(first + count) * width,
        );
        let block = device.upload(&cast(&x[inputs.clone()])).unwrap();
        let block_dot = device.upload(&cast(&xdot[inputs.clone()])).unwrap();
        let tape = network
            .forward(device, &theta_d, &block, &block_dot, count)
            .unwrap();
        expected[outputs.clone()].copy_from_slice(&widen(device.download(tape.output()).unwrap()));
        expected_tangent[outputs.clone()]
            .copy_from_slice(&widen(device.download(tape.tangent()).unwrap()));
        let (mut xbar, mut xdbar) = (
            device.zeros::<T>(count * input).unwrap(),
            device.zeros::<T>(count * input).unwrap(),
        );
        network
            .backward_to_input(
                device,
                &theta_d,
                &mut reference,
                true,
                &tape,
                &block,
                &block_dot,
                &device.upload(&cast(&ybar[outputs.clone()])).unwrap(),
                &device.upload(&cast(&ydotbar[outputs])).unwrap(),
                &ones,
                &mut xbar,
                &mut xdbar,
            )
            .unwrap();
        expected_xbar[inputs.clone()].copy_from_slice(&widen(device.download(&xbar).unwrap()));
        expected_xdbar[inputs].copy_from_slice(&widen(device.download(&xdbar).unwrap()));
    }
    let reference = widen(device.download(&reference).unwrap());
    let largest = reference.iter().fold(0.0f64, |m, g| m.max(g.abs()));
    assert!(largest > 0.1, "the gradient should not vanish");

    let set = omnimd_gpu::TrainSet::<T>::new(device, layouts.to_vec()).unwrap();
    assert_eq!((set.input(), set.output()), (input, width));
    let (xd, xdotd, idsd) = (
        device.upload(&cast(x)).unwrap(),
        device.upload(&cast(xdot)).unwrap(),
        device.upload(&ids).unwrap(),
    );
    let tape = set
        .forward_with_tangent(device, &theta_d, &xd, &xdotd, &idsd, rows)
        .unwrap();
    let close = |name: &str, got: Vec<T>, expected: &[f64], scale: f64| {
        let error = max_error(widen(got), expected);
        assert!(error < tolerance * scale, "set {name} differs by {error}");
    };
    close(
        "outputs",
        device.download(tape.output()).unwrap(),
        &expected,
        1.0,
    );
    close(
        "tangents",
        device.download(tape.tangent()).unwrap(),
        &expected_tangent,
        1.0,
    );
    let (ybar_d, ydotbar_d) = (
        device.upload(&cast(&ybar)).unwrap(),
        device.upload(&cast(&ydotbar)).unwrap(),
    );
    let mut gradient = device.zeros::<T>(theta.len()).unwrap();
    let (mut xbar, mut xdbar) = (
        device.zeros::<T>(rows * input).unwrap(),
        device.zeros::<T>(rows * input).unwrap(),
    );
    for pass in 1..=2 {
        // The gradient accumulates: a second pass doubles it.
        set.backward_to_input(
            device,
            &theta_d,
            &mut gradient,
            true,
            &tape,
            &idsd,
            &ybar_d,
            &ydotbar_d,
            blocks,
            &mut xbar,
            &mut xdbar,
        )
        .unwrap();
        let scaled: Vec<f64> = reference.iter().map(|g| g * f64::from(pass)).collect();
        close(
            "gradient",
            device.download(&gradient).unwrap(),
            &scaled,
            1.0 + largest,
        );
        close(
            "input adjoints",
            device.download(&xbar).unwrap(),
            &expected_xbar,
            1.0,
        );
        close(
            "input tangent adjoints",
            device.download(&xdbar).unwrap(),
            &expected_xdbar,
            1.0,
        );
    }
    // Without `parameters` only the input adjoints are produced, and
    // `backward` alone gives the same gradient.
    set.backward_to_input(
        device,
        &theta_d,
        &mut gradient,
        false,
        &tape,
        &idsd,
        &ybar_d,
        &ydotbar_d,
        blocks,
        &mut xbar,
        &mut xdbar,
    )
    .unwrap();
    let doubled: Vec<f64> = reference.iter().map(|g| 2.0 * g).collect();
    close(
        "untouched gradient",
        device.download(&gradient).unwrap(),
        &doubled,
        1.0 + largest,
    );
    let mut alone = device.zeros::<T>(theta.len()).unwrap();
    set.backward(
        device, &theta_d, &mut alone, &tape, &idsd, &ybar_d, &ydotbar_d, blocks,
    )
    .unwrap();
    close(
        "gradient alone",
        device.download(&alone).unwrap(),
        &reference,
        1.0 + largest,
    );
    // Blocks outside the rows are refused.
    let mut bad = blocks.to_vec();
    bad[0].2 += rows;
    assert!(set
        .backward(device, &theta_d, &mut alone, &tape, &idsd, &ybar_d, &ydotbar_d, &bad)
        .is_err());
    // So is a gradient buffer that does not reach the last parameter.
    let mut short = device.zeros::<T>(theta.len() - 1).unwrap();
    assert!(set
        .backward(device, &theta_d, &mut short, &tape, &idsd, &ybar_d, &ydotbar_d, blocks)
        .is_err());
}

/// Lays `count` networks of the shape `shapes(seed)` out back to back.
fn set_layout(
    count: u64,
    shapes: impl Fn(u64) -> Vec<CpuLayer>,
) -> (Vec<f64>, Vec<Vec<omnimd_gpu::TrainLayer>>) {
    let (mut theta, mut layouts) = (Vec::new(), Vec::new());
    for k in 0..count {
        let (part, mut layout) = training_layout(&shapes(2000 + 100 * k));
        for layer in &mut layout {
            layer.weights += theta.len();
            layer.bias += theta.len();
            layer.timestep = layer.timestep.map(|t| t + theta.len());
        }
        theta.extend(part);
        layouts.push(layout);
    }
    (theta, layouts)
}

#[test]
fn training_set_matches_its_networks_one_by_one() {
    let Some(device) = device() else { return };
    // Blocks of 13, 3 and 21 rows (kernel blocks hold eight) in the network
    // order 2, 0, 1.
    let blocks = [(2, 0, 13), (0, 13, 3), (1, 16, 21)];
    let rows = 37;

    // One input, as the embedding nets: both residual kinds, timesteps and
    // three activations.
    let (theta, layouts) = set_layout(3, |seed| {
        vec![
            CpuLayer::random(1, 6, Activation::Tanh, false, false, seed),
            CpuLayer::random(6, 12, Activation::Tanh, true, true, seed + 10),
            CpuLayer::random(12, 12, Activation::Softplus, true, true, seed + 20),
            CpuLayer::random(12, 5, Activation::Sigmoid, false, false, seed + 30),
        ]
    });
    let (x, xdot) = (values(rows, 2500), values(rows, 2503));
    compare_training_set::<f64>(device, &theta, &layouts, &x, &xdot, &blocks, 1e-12);
    compare_training_set::<f32>(device, &theta, &layouts, &x, &xdot, &blocks, 2e-5);
    // `forward` is the forward pass with unit tangents.
    let set = omnimd_gpu::TrainSet::<f64>::new(device, layouts.clone()).unwrap();
    let theta_d = device.upload(&theta).unwrap();
    let ids: Vec<i32> = (0..rows as i32).map(|r| r % 3).collect();
    let (xd, idsd) = (device.upload(&x).unwrap(), device.upload(&ids).unwrap());
    let unit = set.forward(device, &theta_d, &xd, &idsd, rows).unwrap();
    let ones = device.upload(&vec![1.0f64; rows]).unwrap();
    let explicit = set
        .forward_with_tangent(device, &theta_d, &xd, &ones, &idsd, rows)
        .unwrap();
    assert_eq!(
        device.download(unit.tangent()).unwrap(),
        device.download(explicit.tangent()).unwrap()
    );

    // Several inputs and one linear output, as the fitting nets, with a
    // residual first layer.
    let (theta, layouts) = set_layout(3, |seed| {
        vec![
            CpuLayer::random(7, 7, Activation::Tanh, true, false, seed),
            CpuLayer::random(7, 6, Activation::Tanh, false, false, seed + 10),
            CpuLayer::random(6, 6, Activation::Tanh, true, true, seed + 20),
            CpuLayer::random(6, 1, Activation::Linear, false, false, seed + 30),
        ]
    });
    let (x, xdot) = (values(rows * 7, 2510), values(rows * 7, 2511));
    compare_training_set::<f64>(device, &theta, &layouts, &x, &xdot, &blocks, 1e-12);
    compare_training_set::<f32>(device, &theta, &layouts, &x, &xdot, &blocks, 2e-5);

    // A layer wider than a warp can serve: these rows get a whole thread
    // block each.
    let (wide_theta, wide_layouts) = set_layout(3, |seed| {
        vec![
            CpuLayer::random(140, 9, Activation::Tanh, false, false, seed),
            CpuLayer::random(9, 9, Activation::Tanh, true, true, seed + 10),
            CpuLayer::random(9, 1, Activation::Linear, false, false, seed + 20),
        ]
    });
    let (wide_x, wide_xdot) = (values(rows * 140, 2520), values(rows * 140, 2521));
    for (theta, layouts, x, xdot) in [(&wide_theta, &wide_layouts, &wide_x, &wide_xdot)] {
        compare_training_set::<f64>(device, theta, layouts, x, xdot, &blocks, 1e-11);
        compare_training_set::<f32>(device, theta, layouts, x, xdot, &blocks, 1e-4);
    }

    // Layers larger than one piece of the gradient kernel (16 inputs by 64
    // outputs, four outputs per thread): the widths of the se_e2_a embedding
    // nets, with a last piece of 36 outputs and of 3 inputs and the constant.
    let (tiled_theta, tiled_layouts) = set_layout(3, |seed| {
        vec![
            CpuLayer::random(1, 25, Activation::Tanh, false, false, seed),
            CpuLayer::random(25, 50, Activation::Tanh, true, true, seed + 10),
            CpuLayer::random(50, 100, Activation::Tanh, true, false, seed + 20),
        ]
    });
    let (x, xdot) = (values(rows, 2530), values(rows, 2531));
    compare_training_set::<f64>(
        device,
        &tiled_theta,
        &tiled_layouts,
        &x,
        &xdot,
        &blocks,
        1e-11,
    );
    compare_training_set::<f32>(
        device,
        &tiled_theta,
        &tiled_layouts,
        &x,
        &xdot,
        &blocks,
        1e-4,
    );
    // More rows of one network than a piece of the gradient kernel holds
    // (128): the sums of its pieces meet in the gradient.
    let long = [(1, 0, 290), (0, 290, 10)];
    let (x, xdot) = (values(300, 2550), values(300, 2551));
    compare_training_set::<f64>(
        device,
        &tiled_theta,
        &tiled_layouts,
        &x,
        &xdot,
        &long,
        1e-11,
    );
    compare_training_set::<f32>(device, &tiled_theta, &tiled_layouts, &x, &xdot, &long, 1e-4);
    // The same with a whole thread block per row, timesteps wider than one
    // warp, and output counts that end inside every position of a thread's
    // four outputs (70, 140, 37, 1).
    let (tiled_theta, tiled_layouts) = set_layout(3, |seed| {
        vec![
            CpuLayer::random(20, 70, Activation::Tanh, false, false, seed),
            CpuLayer::random(70, 140, Activation::Tanh, true, true, seed + 10),
            CpuLayer::random(140, 37, Activation::Tanh, false, false, seed + 20),
            CpuLayer::random(37, 1, Activation::Linear, false, false, seed + 30),
        ]
    });
    let (x, xdot) = (values(rows * 20, 2540), values(rows * 20, 2541));
    compare_training_set::<f64>(
        device,
        &tiled_theta,
        &tiled_layouts,
        &x,
        &xdot,
        &blocks,
        1e-11,
    );
    compare_training_set::<f32>(
        device,
        &tiled_theta,
        &tiled_layouts,
        &x,
        &xdot,
        &blocks,
        1e-4,
    );

    // As many inputs as a piece holds, like the 1600 of an se_e2_a fitting
    // net: the bias, the row after the inputs, is a piece of its own.
    let (edge_theta, edge_layouts) = set_layout(3, |seed| {
        vec![
            CpuLayer::random(64, 9, Activation::Tanh, false, false, seed),
            CpuLayer::random(9, 1, Activation::Linear, false, false, seed + 10),
        ]
    });
    let (x, xdot) = (values(rows * 64, 2560), values(rows * 64, 2561));
    compare_training_set::<f64>(
        device,
        &edge_theta,
        &edge_layouts,
        &x,
        &xdot,
        &blocks,
        1e-11,
    );
    compare_training_set::<f32>(device, &edge_theta, &edge_layouts, &x, &xdot, &blocks, 1e-4);

    // Refused: no network, different shapes, a bias away from its weights.
    let new = |networks: Vec<Vec<omnimd_gpu::TrainLayer>>| {
        omnimd_gpu::TrainSet::<f64>::new(device, networks)
    };
    let layout = layouts[0].clone();
    assert!(new(vec![]).is_err());
    let mut other = layout.clone();
    other[2].residual = false;
    assert!(new(vec![layout.clone(), other]).is_err());
    let mut apart = layout.clone();
    apart[1].bias += 1;
    assert!(new(vec![apart]).is_err());
    let mut unchained = layout.clone();
    unchained[1].input = 5;
    assert!(new(vec![unchained]).is_err());
    assert!(new(vec![layout]).is_ok());
}

#[test]
fn sibling_handles_work_at_the_same_time() {
    let Some(device) = device() else { return };
    // Three threads, each with a handle of its own on the same context,
    // multiply their own matrices while the others do.
    let handles: Vec<Device> = (0..3).map(|_| device.sibling().unwrap()).collect();
    assert_eq!(handles[0].name(), device.name());
    std::thread::scope(|scope| {
        for (index, handle) in handles.into_iter().enumerate() {
            scope.spawn(move || {
                let shape = MatmulShape {
                    m: 40 + index,
                    k: 30,
                    n: 20,
                };
                for round in 0..25 {
                    let seed = 3000 + 100 * index as u64 + round;
                    let (a, b) = (
                        values(shape.m * shape.k, seed),
                        values(shape.k * shape.n, seed + 50),
                    );
                    let mut c = handle.zeros::<f64>(shape.m * shape.n).unwrap();
                    handle
                        .matmul(
                            &handle.upload(&a).unwrap(),
                            &handle.upload(&b).unwrap(),
                            &mut c,
                            shape,
                            false,
                        )
                        .unwrap();
                    let expected = matmul_cpu(&a, &b, shape.m, shape.k, shape.n);
                    assert!(max_error(handle.download(&c).unwrap(), &expected) < 1e-12);
                }
            });
        }
    });
}
