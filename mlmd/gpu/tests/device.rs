//! Device smoke tests against plain CPU references. They skip without a CUDA
//! driver; set OMNIMD_REQUIRE_GPU=1 (GPU CI, local 4090 runs) to make a missing
//! device a failure instead.
use omnimd_gpu::{Activation, DenseLayer, DenseNetwork, Device, MatmulShape};
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
