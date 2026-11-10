//! Device smoke tests against plain CPU references. They skip without a CUDA
//! driver; set OMNIMD_REQUIRE_GPU=1 (GPU CI, local 4090 runs) to make a missing
//! device a failure instead.
use omnimd_gpu::{Activation, Cell, DenseLayer, DenseNetwork, Device, MatmulShape, NeighborList};
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
        let mut groups: Vec<Vec<(f64, usize, [f64; 3])>> = vec![Vec::new(); sel.len()];
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
                            groups[types[j] as usize].push((r, j, d));
                            nearest = nearest.min(r);
                        }
                    }
                }
            }
        }
        let mut slot = i * slots;
        for (t, group) in groups.iter_mut().enumerate() {
            counts[i * sel.len() + t] = group.len() as i32;
            group.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
            for (k, (_, j, d)) in group.iter().take(sel[t]).enumerate() {
                index[slot + k] = *j as i32;
                delta[(slot + k) * 3..][..3].copy_from_slice(d);
            }
            slot += sel[t];
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
    let (index, delta, counts, nearest) = neighbors_cpu(positions, types, &cell, rcut, sel);
    assert_eq!(
        device.download(&list.count).unwrap(),
        counts,
        "{label}: counts"
    );
    let (got_index, got_delta) = (
        device.download(&list.index).unwrap(),
        device.download(&list.delta).unwrap(),
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
