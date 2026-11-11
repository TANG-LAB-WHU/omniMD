//! Probes of what device calls and arithmetic cost on this machine. They
//! print timings and check nothing, so they are ignored by default:
//!
//! ```text
//! cargo test --release -p omnimd-gpu --test costs -- --ignored --nocapture
//! ```
//!
//! The crate's README records what they gave on an RTX 4090 and what the
//! kernels do about it.
use omnimd_gpu::Device;
use omnimd_gpu::{LaunchConfig, PushKernelArg};
use std::time::Instant;

const SOURCE: &str = r#"
extern "C" __global__ void touch(double* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0;
}
"#;

fn micros(label: &str, rounds: usize, device: &Device, mut body: impl FnMut()) {
    for _ in 0..200 {
        body();
    }
    device.synchronize().unwrap();
    let start = Instant::now();
    for _ in 0..rounds {
        body();
    }
    let issued = start.elapsed().as_secs_f64() * 1e6 / rounds as f64;
    device.synchronize().unwrap();
    let done = start.elapsed().as_secs_f64() * 1e6 / rounds as f64;
    println!("{label:<44} {issued:8.2} us issued {done:8.2} us with final sync");
}

#[test]
#[ignore]
fn device_call_costs() {
    let device = Device::new(0).unwrap();
    let stream = device.stream().clone();
    let function = device.kernels().function(SOURCE, "touch").unwrap();
    for n in [1024usize, 16 * 1024] {
        println!("--- {n} doubles");
        let host = vec![1.0f64; n];
        let mut held = device.upload(&host).unwrap();
        let mut back = vec![0.0f64; n];
        let n_i = n as i32;
        let config = LaunchConfig {
            grid_dim: (n.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        micros("alloc_uninit + free", 2000, &device, || {
            drop(unsafe { device.alloc_uninit::<f64>(n) }.unwrap());
        });
        micros("zeros + free", 2000, &device, || {
            drop(device.zeros::<f64>(n).unwrap());
        });
        micros("upload (alloc + copy) + free", 2000, &device, || {
            drop(device.upload(&host).unwrap());
        });
        micros("copy into a held buffer", 2000, &device, || {
            stream.memcpy_htod(&host, &mut held).unwrap();
        });
        micros("memset of a held buffer", 2000, &device, || {
            stream.memset_zeros(&mut held).unwrap();
        });
        micros("kernel launch", 2000, &device, || {
            let mut launch = stream.launch_builder(&function);
            launch.arg(&mut held).arg(&n_i);
            unsafe { launch.launch(config) }.map(drop).unwrap();
        });
        micros("synchronize alone", 2000, &device, || {
            device.synchronize().unwrap();
        });
        micros("download to a new vector", 2000, &device, || {
            drop(device.download(&held).unwrap());
        });
        micros("download into a held vector", 2000, &device, || {
            stream.memcpy_dtoh(&held, &mut back).unwrap();
        });
        micros("launch + download", 2000, &device, || {
            let mut launch = stream.launch_builder(&function);
            launch.arg(&mut held).arg(&n_i);
            unsafe { launch.launch(config) }.map(drop).unwrap();
            stream.memcpy_dtoh(&held, &mut back).unwrap();
        });
        micros("5 launches + download", 2000, &device, || {
            for _ in 0..5 {
                let mut launch = stream.launch_builder(&function);
                launch.arg(&mut held).arg(&n_i);
                unsafe { launch.launch(config) }.map(drop).unwrap();
            }
            stream.memcpy_dtoh(&held, &mut back).unwrap();
        });
        micros("copy in + 5 launches + download", 2000, &device, || {
            stream.memcpy_htod(&host, &mut held).unwrap();
            for _ in 0..5 {
                let mut launch = stream.launch_builder(&function);
                launch.arg(&mut held).arg(&n_i);
                unsafe { launch.launch(config) }.map(drop).unwrap();
            }
            stream.memcpy_dtoh(&held, &mut back).unwrap();
        });
        micros("copy in + 5 launches + 4 downloads", 2000, &device, || {
            stream.memcpy_htod(&host, &mut held).unwrap();
            for _ in 0..5 {
                let mut launch = stream.launch_builder(&function);
                launch.arg(&mut held).arg(&n_i);
                unsafe { launch.launch(config) }.map(drop).unwrap();
            }
            for _ in 0..4 {
                stream.memcpy_dtoh(&held, &mut back).unwrap();
            }
        });
        let context = stream.context().clone();
        let mut pin_up = unsafe { context.alloc_pinned::<f64>(n) }.unwrap();
        pin_up.as_mut_slice().unwrap().copy_from_slice(&host);
        let mut pin_down = unsafe { context.alloc_pinned_with_flags::<f64>(n, 0) }.unwrap();
        micros("pinned copy into a held buffer", 2000, &device, || {
            stream.memcpy_htod(&pin_up, &mut held).unwrap();
        });
        let mut sum = 0.0;
        micros("pinned download + read", 2000, &device, || {
            stream.memcpy_dtoh(&held, &mut pin_down).unwrap();
            sum += pin_down.as_slice().unwrap()[0];
        });
        micros("launch + 4 pinned downloads + read", 2000, &device, || {
            let mut launch = stream.launch_builder(&function);
            launch.arg(&mut held).arg(&n_i);
            unsafe { launch.launch(config) }.map(drop).unwrap();
            for _ in 0..4 {
                stream.memcpy_dtoh(&held, &mut pin_down).unwrap();
            }
            sum += pin_down.as_slice().unwrap()[0];
        });
        micros(
            "launch + 4 downloads into held vectors",
            2000,
            &device,
            || {
                let mut launch = stream.launch_builder(&function);
                launch.arg(&mut held).arg(&n_i);
                unsafe { launch.launch(config) }.map(drop).unwrap();
                for _ in 0..4 {
                    stream.memcpy_dtoh(&held, &mut back).unwrap();
                }
            },
        );
        micros(
            "launch + 1 download into a held vector",
            2000,
            &device,
            || {
                let mut launch = stream.launch_builder(&function);
                launch.arg(&mut held).arg(&n_i);
                unsafe { launch.launch(config) }.map(drop).unwrap();
                stream.memcpy_dtoh(&held, &mut back).unwrap();
            },
        );
        assert!(sum.is_finite());
        micros(
            "8 uploads + 6 zeros + launch + frees",
            2000,
            &device,
            || {
                let ups: Vec<_> = (0..8)
                    .map(|_| device.upload(&host[..n.min(4096)]).unwrap())
                    .collect();
                let zs: Vec<_> = (0..6).map(|_| device.zeros::<f64>(n).unwrap()).collect();
                let mut launch = stream.launch_builder(&function);
                launch.arg(&mut held).arg(&n_i);
                unsafe { launch.launch(config) }.map(drop).unwrap();
                drop((ups, zs));
            },
        );
    }
}

const BURN: &str = r#"
extern "C" __global__ void burn(double* x, int iters, int active, int mode) {
    int lane = threadIdx.x & 31;
    if (lane >= active) return;
    long long at = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    double a = x[at], acc = 0.0;
    if (mode == 0) for (int k = 0; k < iters; ++k) acc = acc * 0.999999 + a;
    if (mode == 1) {
        float af = (float)a, accf = 0.0f;
        for (int k = 0; k < iters; ++k) accf = accf * 0.999999f + af;
        acc = accf;
    }
    if (mode == 2) for (int k = 0; k < iters; ++k) acc = a / (acc + 1.5);
    if (mode == 3) for (int k = 0; k < iters; ++k) acc = sqrt(acc + a + 1.0);
    if (mode == 4) {
        double b = 0.0, c = 0.0, d = 0.0;
        for (int k = 0; k < iters; k += 4) {
            acc = acc * 0.999999 + a;
            b = b * 0.999998 + a;
            c = c * 0.999997 + a;
            d = d * 0.999996 + a;
        }
        acc += b + c + d;
    }
    if (mode == 5) {
        float af = (float)a, p = 0.0f, q = 0.0f, r = 0.0f, s = 0.0f;
        for (int k = 0; k < iters; k += 4) {
            p = p * 0.999999f + af;
            q = q * 0.999998f + af;
            r = r * 0.999997f + af;
            s = s * 0.999996f + af;
        }
        acc = p + q + r + s;
    }
    if (mode == 6) for (int k = 0; k < iters; ++k) acc += (double)(float)(a + k);
    if (mode == 7) {
        long long u = (long long)(a * 1000.0), v = 0;
        for (int k = 0; k < iters; ++k) v += (u + k < v) ? 3 : 5;
        acc = (double)v;
    }
    x[at] = acc;
}
"#;

#[test]
#[ignore]
fn arithmetic_costs() {
    let device = Device::new(0).unwrap();
    let stream = device.stream().clone();
    let function = device.kernels().function(BURN, "burn").unwrap();
    let iters = 40_000i32;
    let names = [
        "double fma chain",
        "float fma chain",
        "double division chain",
        "double sqrt chain",
        "double fma, 4 chains",
        "float fma, 4 chains",
        "double->float->double",
        "64-bit integer compare-add",
    ];
    println!("ns per operation and warp ({iters} operations per thread)");
    for (blocks, threads, active) in [
        (1usize, 32u32, 32i32),
        (88, 32, 32),
        (88, 32, 1),
        (88, 128, 32),
        (88, 256, 32),
        (88, 1024, 32),
        (22, 128, 32),
        (128, 256, 32),
        (1024, 256, 32),
    ] {
        let mut held = device
            .upload(&vec![0.5f64; blocks * threads as usize])
            .unwrap();
        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut line = format!("{blocks:>5} blocks x {threads:>4} threads, {active:>2} lanes:");
        for mode in 0..names.len() as i32 {
            let mut run = || {
                let mut launch = stream.launch_builder(&function);
                launch.arg(&mut held).arg(&iters).arg(&active).arg(&mode);
                unsafe { launch.launch(config) }.map(drop).unwrap();
                device.synchronize().unwrap();
            };
            run();
            let start = Instant::now();
            for _ in 0..5 {
                run();
            }
            let ns = start.elapsed().as_secs_f64() * 1e9 / 5.0 / iters as f64;
            line.push_str(&format!(" {ns:7.2}"));
        }
        println!("{line}");
    }
    for (k, name) in names.iter().enumerate() {
        println!("column {k}: {name}");
    }
}

#[test]
#[ignore]
fn neighbor_search_costs() {
    use omnimd_gpu::{Cell, NeighborSearch};
    let device = Device::new(0).unwrap();
    // 88 atoms of four types at the density of hydroxyapatite (two of its
    // hexagonal cells), then the same replicated.
    let cell = [[9.42, 0.0, 0.0], [-4.71, 8.158, 0.0], [0.0, 0.0, 13.76]];
    let mut state = 12345u64;
    let mut uniform = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for (natoms, repeat) in [(88usize, 1usize), (704, 2), (5632, 4)] {
        let mut positions: Vec<[f64; 3]> = Vec::new();
        let scaled = cell.map(|row| row.map(|v| v * repeat as f64));
        while positions.len() < natoms {
            let f = [uniform(), uniform(), uniform()];
            positions.push(std::array::from_fn(|a| {
                (0..3).map(|b| f[b] * scaled[b][a]).sum()
            }));
        }
        let types: Vec<i32> = (0..natoms).map(|i| (i % 4) as i32).collect();
        let sel = [44usize, 16, 60, 72, 20];
        let cell = Cell::Periodic(scaled);
        let mut search = NeighborSearch::new(6.0, &sel).unwrap();
        let rounds = 3000;
        for _ in 0..100 {
            search.run(&device, &positions, &types, &cell).unwrap();
        }
        device.synchronize().unwrap();
        let start = Instant::now();
        for _ in 0..rounds {
            search.run(&device, &positions, &types, &cell).unwrap();
        }
        let issued = start.elapsed().as_secs_f64() * 1e6 / rounds as f64;
        device.synchronize().unwrap();
        let start = Instant::now();
        for _ in 0..rounds {
            search.run(&device, &positions, &types, &cell).unwrap();
            device.synchronize().unwrap();
        }
        let synced = start.elapsed().as_secs_f64() * 1e6 / rounds as f64;
        let stats = search.list().unwrap().stats(&device).unwrap();
        println!(
            "{natoms:>6} atoms: {issued:7.1} us to issue (back to back), {synced:7.1} us with a sync each; \
             most neighbors per type {:?}",
            stats.max_count
        );
    }
}

const LOADS: &str = r#"
// The inner loop of a dense layer in several shapes: `inputs` values `s[k]`
// from shared memory times weights `w[k][n]` from global memory.
extern "C" __global__ void loads(const float* w, float* out, int inputs, int stride, int repeats,
        int mode) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    for (int k = tid; k < inputs; k += blockDim.x) s[k] = 1.0f / (1 + k);
    __syncthreads();
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int r = 0; r < repeats; ++r) {
        if (mode == 0)        // shared loads only
            for (int k = 0; k + 8 <= inputs; k += 8)
                for (int u = 0; u < 8; ++u) acc[u] += s[k + u] * 0.999f;
        if (mode == 1)        // weight loads only
            for (int k = 0; k + 8 <= inputs; k += 8)
                for (int u = 0; u < 8; ++u) acc[u] += w[(long long)(k + u) * stride + tid] * 0.999f;
        if (mode == 2)        // one output per thread: a shared and a weight load per product
            for (int k = 0; k + 8 <= inputs; k += 8)
                for (int u = 0; u < 8; ++u) acc[u] += s[k + u] * w[(long long)(k + u) * stride + tid];
        if (mode == 3)        // four outputs per thread
            for (int k = 0; k < inputs; ++k) {
                float a = s[k];
                const float* row = w + (long long)k * stride + tid;
                for (int c = 0; c < 4; ++c) acc[c] += a * row[c * blockDim.x];
            }
        if (mode == 4)        // eight outputs per thread
            for (int k = 0; k < inputs; ++k) {
                float a = s[k];
                const float* row = w + (long long)k * stride + tid;
                for (int c = 0; c < 8; ++c) acc[c] += a * row[c * blockDim.x];
            }
        if (mode == 6)        // one output per thread, its weights contiguous
            for (int k = 0; k + 8 <= inputs; k += 8)
                for (int u = 0; u < 8; ++u) acc[u] += s[k + u] * w[(long long)tid * inputs + k + u];
        if (mode == 7)        // the same weights for every thread: a broadcast load
            for (int k = 0; k + 8 <= inputs; k += 8)
                for (int u = 0; u < 8; ++u) acc[u] += s[k + u] * w[k + u];
        if (mode == 8)        // coalesced as mode 2, but over 16 inputs again and again
            for (int k = 0; k + 8 <= inputs; k += 8)
                for (int u = 0; u < 8; ++u)
                    acc[u] += s[k + u] * w[(long long)((k + u) & 15) * stride + tid];
        if (mode == 5)        // four outputs per thread, inputs two at a time
            for (int k = 0; k + 2 <= inputs; k += 2) {
                float a = s[k], b = s[k + 1];
                const float* row = w + (long long)k * stride + tid;
                for (int c = 0; c < 4; ++c)
                    acc[c] += a * row[c * blockDim.x] + b * row[stride + c * blockDim.x];
            }
    }
    float total = 0;
    for (int u = 0; u < 8; ++u) total += acc[u];
    out[(long long)blockIdx.x * blockDim.x + tid] = total;
}
"#;

#[test]
#[ignore]
fn load_costs() {
    let device = Device::new(0).unwrap();
    let stream = device.stream().clone();
    let function = device.kernels().function(LOADS, "loads").unwrap();
    let (inputs, stride) = (1600i32, 256i32);
    let weights = device
        .upload(&vec![0.001f32; (inputs * stride) as usize])
        .unwrap();
    let names = [
        "shared loads",
        "weight loads",
        "shared x weight, 1 output per thread",
        "4 outputs per thread",
        "8 outputs per thread",
        "4 outputs per thread, 2 inputs at a time",
        "1 output per thread, weights contiguous per thread",
        "broadcast weight load",
        "coalesced weights of 16 inputs, reused",
    ];
    println!("ns per product and warp, {inputs} inputs, weights [{inputs}][{stride}]");
    for (blocks, threads) in [
        (1usize, 32u32),
        (88, 32),
        (88, 64),
        (88, 256),
        (22, 256),
        (128, 256),
        (224, 64),
        (224, 256),
        (1024, 256),
    ] {
        let mut out = device.zeros::<f32>(blocks * threads as usize).unwrap();
        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: (inputs * 4) as u32,
        };
        let mut line = format!("{blocks:>4} blocks x {threads:>3} threads:");
        for mode in 0..names.len() as i32 {
            let outputs = match mode {
                3 | 5 => 4,
                4 => 8,
                _ => 1,
            };
            if threads as i32 * outputs > stride {
                line.push_str("       -");
                continue;
            }
            let repeats = 40i32;
            let mut run = || {
                let mut launch = stream.launch_builder(&function);
                launch
                    .arg(&weights)
                    .arg(&mut out)
                    .arg(&inputs)
                    .arg(&stride)
                    .arg(&repeats)
                    .arg(&mode);
                unsafe { launch.launch(config) }.map(drop).unwrap();
                device.synchronize().unwrap();
            };
            run();
            let start = Instant::now();
            for _ in 0..5 {
                run();
            }
            let products = (inputs * repeats * outputs) as f64;
            let ns = start.elapsed().as_secs_f64() * 1e9 / 5.0 / products;
            line.push_str(&format!(" {ns:7.3}"));
        }
        println!("{line}");
    }
    for (k, name) in names.iter().enumerate() {
        println!("column {k}: {name}");
    }
}

/// The fitting nets of a five-type model for the type runs of two real
/// systems and of batches of one of them: the one kernel a row against the
/// first layer in tiles with the other layers two rows a block.
#[test]
#[ignore]
fn fitting_costs() {
    use omnimd_gpu::{Activation, GradientNetwork, LayerParameters};
    let device = Device::new(0).unwrap();
    let (input, width) = (1600usize, 240usize);
    let layer = |i: usize, o: usize, activation, residual, seed: usize| LayerParameters::<f32> {
        input: i,
        output: o,
        weights: (0..i * o)
            .map(|e| (((e + seed) * 37 % 101) as f32 - 50.0) / (50.0 * (i as f32).sqrt()))
            .collect(),
        bias: vec![0.01; o],
        timestep: None,
        activation,
        residual,
    };
    let nets: Vec<_> = (0..5)
        .map(|t| {
            vec![
                layer(input, width, Activation::Tanh, false, t),
                layer(width, width, Activation::Tanh, true, t + 7),
                layer(width, width, Activation::Tanh, true, t + 13),
                layer(width, 1, Activation::Linear, false, t + 19),
            ]
        })
        .collect();
    let mut network = GradientNetwork::new_set(&device, &nets).unwrap();
    println!("microseconds per call, fitting nets {input} -> {width} -> {width} -> {width} -> 1");
    // The type runs of the HAP bulk frame, for so many copies of it in a batch.
    let bulk = |copies: usize| {
        let mut first = 0;
        [20usize, 12, 52, 4]
            .iter()
            .enumerate()
            .map(|(net, atoms)| {
                let group = (net, first, atoms * copies);
                first += atoms * copies;
                group
            })
            .collect::<Vec<_>>()
    };
    let cases: [(&str, Vec<(usize, usize, usize)>); 8] = [
        ("HAP bulk, 88", bulk(1)),
        ("2 of it, 176", bulk(2)),
        ("8 of it, 704", bulk(8)),
        ("32 of it, 2816", bulk(32)),
        (
            "monetite, 224",
            vec![(0, 0, 32), (1, 32, 32), (2, 64, 128), (3, 192, 32)],
        ),
        ("4 of it, 896", vec![(0, 0, 128), (1, 128, 128), (2, 256, 512), (3, 768, 128)]),
        ("one type, 88", vec![(2, 0, 88)]),
        ("one type, 1024", vec![(2, 0, 1024)]),
    ];
    for (label, groups) in &cases {
        let groups = groups.as_slice();
        let rows: usize = groups.iter().map(|g| g.2).sum();
        let x = device.upload(&vec![0.01f32; rows * input]).unwrap();
        let mut value = device.zeros::<f32>(rows).unwrap();
        let mut gradient = device.zeros::<f32>(rows * input).unwrap();
        let mut line = format!("{label:<16}");
        // Alternate the two layouts so that drift of the machine cancels.
        let mut sums = [0.0f64; 2];
        let rounds = 6;
        for round in 0..rounds {
            for (slot, tiled) in [(0usize, false), (1, true)] {
                assert_eq!(network.set_tiled(tiled), tiled);
                let mut run = || {
                    network
                        .forward_gradient_grouped(
                            &device,
                            &x,
                            groups,
                            rows,
                            &mut value,
                            &mut gradient,
                        )
                        .unwrap();
                };
                for _ in 0..20 {
                    run();
                }
                device.synchronize().unwrap();
                let start = Instant::now();
                for _ in 0..300 {
                    run();
                }
                device.synchronize().unwrap();
                if round > 0 {
                    sums[slot] += start.elapsed().as_secs_f64() * 1e6 / 300.0;
                }
            }
        }
        let [plain, tiled] = sums.map(|s| s / (rounds - 1) as f64);
        line.push_str(&format!(
            " one kernel {plain:7.1}   tiles {tiled:7.1}   ratio {:.2}",
            plain / tiled
        ));
        println!("{line}");
    }
}
