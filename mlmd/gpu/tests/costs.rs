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
    if (mode == 8) {
        // By a number the compiler does not know (4 here): a real division.
        int n = (int)(a * 8.0), v = 0;
        for (int k = 0; k < iters; ++k) v = (v + 7 * k + 11) / n;
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
        "32-bit integer division chain",
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

/// What marks cost the host, and what the device takes over single calls
/// by them. The device is idle before each call here, so a time is from the
/// call to the end of its work: for a kernel that does nothing, the call
/// itself and the start of a kernel.
#[test]
#[ignore]
fn device_times_by_marks() {
    let device = Device::new(0).unwrap();
    let stream = device.stream().clone();
    let function = device.kernels().function(SOURCE, "touch").unwrap();
    let n = 1024usize;
    let host = vec![1.0f64; n];
    let mut held = device.upload(&host).unwrap();
    let mut back = vec![0.0f64; n];
    micros("a new mark", 2000, &device, || {
        drop(device.mark().unwrap());
    });
    let by_marks = |label: &str, body: &mut dyn FnMut()| {
        for _ in 0..200 {
            body();
        }
        let mut sum = 0.0;
        for _ in 0..2000 {
            device.synchronize().unwrap();
            let before = device.mark().unwrap();
            body();
            let after = device.mark().unwrap();
            sum += before.ms_until(&after).unwrap() * 1e3;
        }
        println!("{label:<44} {:8.2} us between marks", sum / 2000.0);
    };
    by_marks("nothing", &mut || {});
    for (label, blocks, threads, count) in [
        ("a kernel that does nothing, 1 block", 1u32, 32u32, 0i32),
        ("a kernel that does nothing, 88 blocks", 88, 32, 0),
        ("a kernel that does nothing, 88 x 256", 88, 256, 0),
        ("a kernel that does nothing, 5632 blocks", 5632, 32, 0),
        ("a kernel over 1024 doubles", 4, 256, 1024),
    ] {
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        by_marks(label, &mut || {
            let mut launch = stream.launch_builder(&function);
            launch.arg(&mut held).arg(&count);
            unsafe { launch.launch(config) }.map(drop).unwrap();
        });
    }
    by_marks("copy of 1024 doubles to the device", &mut || {
        stream.memcpy_htod(&host, &mut held).unwrap();
    });
    by_marks("memset of 1024 doubles", &mut || {
        stream.memset_zeros(&mut held).unwrap();
    });
    by_marks("copy of 1024 doubles to the host", &mut || {
        stream.memcpy_dtoh(&held, &mut back).unwrap();
    });
    // Behind a kernel that keeps the device busy, the calls are issued
    // before the device gets to them: what is left is the device's own time.
    let busy = LaunchConfig {
        grid_dim: (8192, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut wide = device.zeros::<f64>(8192 * 256).unwrap();
    let wide_n: i32 = 8192 * 256;
    let mut sums = [0.0; 5];
    let nothing = LaunchConfig {
        grid_dim: (88, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let zero = 0i32;
    for round in 0..1200 {
        device.synchronize().unwrap();
        let mut launch = stream.launch_builder(&function);
        launch.arg(&mut wide).arg(&wide_n);
        unsafe { launch.launch(busy) }.map(drop).unwrap();
        let mut marks = vec![device.mark().unwrap()];
        for _ in 0..5 {
            let mut launch = stream.launch_builder(&function);
            launch.arg(&mut held).arg(&zero);
            unsafe { launch.launch(nothing) }.map(drop).unwrap();
            marks.push(device.mark().unwrap());
        }
        if round >= 200 {
            for (k, sum) in sums.iter_mut().enumerate() {
                *sum += marks[k].ms_until(&marks[k + 1]).unwrap() * 1e3;
            }
        }
    }
    println!(
        "kernels that do nothing behind a busy device  {:?} us each, in turn",
        sums.map(|sum| (sum / 10.0).round() / 100.0)
    );
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
        (
            "4 of it, 896",
            vec![(0, 0, 128), (1, 128, 128), (2, 256, 512), (3, 768, 128)],
        ),
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

/// What an allocation costs by its size: the work buffers of an evaluation
/// come from the stream-ordered pool and go back to it, four at a time here
/// as for the rows of an se_e3 descriptor, with a launch between and a
/// download at the end as an evaluation has them.
#[test]
#[ignore]
fn allocation_costs_by_size() {
    let device = Device::new(0).unwrap();
    let stream = device.stream().clone();
    let function = device.kernels().function(SOURCE, "touch").unwrap();
    let mut small = device.upload(&vec![1.0f64; 1024]).unwrap();
    let mut back = vec![0.0f64; 1024];
    let n_i = 1024i32;
    let config = LaunchConfig {
        grid_dim: (4, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    for megabytes in [0.008, 0.125, 1.0, 8.0, 64.0, 256.0] {
        let n = (megabytes * 1024.0 * 1024.0 / 8.0) as usize;
        println!("--- buffers of {megabytes} MiB");
        micros("alloc_uninit + free", 2000, &device, || {
            drop(unsafe { device.alloc_uninit::<f64>(n) }.unwrap());
        });
        micros("4 x alloc_uninit, then free", 2000, &device, || {
            let held: Vec<_> = (0..4)
                .map(|_| unsafe { device.alloc_uninit::<f64>(n) }.unwrap())
                .collect();
            drop(held);
        });
        micros("launch + download", 2000, &device, || {
            let mut launch = stream.launch_builder(&function);
            launch.arg(&mut small).arg(&n_i);
            unsafe { launch.launch(config) }.map(drop).unwrap();
            stream.memcpy_dtoh(&small, &mut back).unwrap();
        });
        micros("4 x alloc, launch + download, free", 2000, &device, || {
            let held: Vec<_> = (0..4)
                .map(|_| unsafe { device.alloc_uninit::<f64>(n) }.unwrap())
                .collect();
            let mut launch = stream.launch_builder(&function);
            launch.arg(&mut small).arg(&n_i);
            unsafe { launch.launch(config) }.map(drop).unwrap();
            stream.memcpy_dtoh(&small, &mut back).unwrap();
            drop(held);
        });
    }
}

const SPIN: &str = r#"
extern "C" __global__ void spin(long long cycles) {
    long long start = clock64();
    while (clock64() - start < cycles) {}
}
"#;

/// Whether a copy from pageable host memory into a device buffer waits for
/// a stream that is still busy. The grouped products upload the addresses
/// of their operands that way when the layout of a chunk changes, between
/// launches on that stream. Timed on the host, on an idle stream and right
/// behind a kernel that spins for some milliseconds; the copy must still
/// deliver what the host buffer held, though it is gone by then.
#[test]
#[ignore]
fn upload_behind_a_busy_stream() {
    let device = Device::new(0).unwrap();
    let stream = device.stream().clone();
    let spin = device.kernels().function(SPIN, "spin").unwrap();
    let one = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let cycles: i64 = 10_000_000;
    let mut spun = Vec::new();
    for _ in 0..5 {
        device.synchronize().unwrap();
        let start = Instant::now();
        let mut launch = stream.launch_builder(&spin);
        launch.arg(&cycles);
        unsafe { launch.launch(one) }.map(drop).unwrap();
        device.synchronize().unwrap();
        spun.push(start.elapsed().as_secs_f64() * 1e3);
    }
    println!("the spin alone: {spun:.3?} ms");
    let rounds = 30;
    for len in [420usize, 4096, 65536, 1 << 20] {
        let mut held = device.zeros::<u64>(len).unwrap();
        let (mut idle, mut busy) = (Vec::new(), Vec::new());
        for round in 0..rounds as u64 {
            let host: Vec<u64> = (0..len as u64).map(|v| v * 3 + round).collect();
            device.synchronize().unwrap();
            let start = Instant::now();
            stream.memcpy_htod(&host, &mut held).unwrap();
            idle.push(start.elapsed().as_secs_f64() * 1e6);
            device.synchronize().unwrap();
            let mut launch = stream.launch_builder(&spin);
            launch.arg(&cycles);
            unsafe { launch.launch(one) }.map(drop).unwrap();
            let start = Instant::now();
            stream.memcpy_htod(&host, &mut held).unwrap();
            busy.push(start.elapsed().as_secs_f64() * 1e6);
            drop(host);
            device.synchronize().unwrap();
        }
        idle.sort_by(f64::total_cmp);
        busy.sort_by(f64::total_cmp);
        let half = rounds / 2;
        println!(
            "{:>8} KB from pageable memory, us on the host: idle stream {:.1} (least {:.1}), behind the spin {:.1} (least {:.1}, most {:.1})",
            len * 8 / 1024,
            idle[half],
            idle[0],
            busy[half],
            busy[0],
            busy[rounds - 1]
        );
        let back = stream.clone_dtoh(&held).unwrap();
        assert_eq!(back[len - 1], (len as u64 - 1) * 3 + rounds as u64 - 1);
    }
    // Copies one after another behind the spin, each from a buffer of its
    // own: how many the host issues before one waits for the stream.
    for len in [420usize, 5120, 65536] {
        let mut held = device.zeros::<u64>(len).unwrap();
        let copies = 64u64;
        let hosts: Vec<Vec<u64>> = (0..copies)
            .map(|copy| (0..len as u64).map(|v| v * 5 + copy).collect())
            .collect();
        device.synchronize().unwrap();
        let mut launch = stream.launch_builder(&spin);
        launch.arg(&cycles);
        unsafe { launch.launch(one) }.map(drop).unwrap();
        let mut times = Vec::new();
        for host in &hosts {
            let start = Instant::now();
            stream.memcpy_htod(host, &mut held).unwrap();
            times.push(start.elapsed().as_secs_f64() * 1e6);
            if times[times.len() - 1] > 1000.0 {
                break;
            }
        }
        device.synchronize().unwrap();
        let issued = times.len() as u64;
        let kb = len * 8 / 1024;
        match times.iter().position(|&t| t > 1000.0) {
            Some(at) => println!(
                "{kb:>8} KB copies behind the spin: {at} issued without waiting ({} KB), the next waited {:.0} us",
                at * kb,
                times[at]
            ),
            None => println!(
                "{kb:>8} KB copies behind the spin: all {issued} issued without waiting ({} KB, the slowest {:.1} us)",
                issued as usize * kb,
                times.iter().cloned().fold(0.0, f64::max)
            ),
        }
        let back = stream.clone_dtoh(&held).unwrap();
        assert_eq!(back[len - 1], (len as u64 - 1) * 5 + issued - 1);
    }
}

const TABLE_PROBE: &str = r#"
template <typename T>
__device__ T warp_total(T v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// One warp a row, the lanes over the outputs, as the kernels of the tables
// and of the forces go: every row ends in one number, so that nothing is
// left out by the compiler.
template <typename T, bool TANGENT>
__device__ void lookup(const int* pieces, const T* us, const T* coef, int width, long long rows,
        T* out) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long row = (long long)blockIdx.x * warps + warp;
    if (row >= rows) return;
    int piece = pieces[row];
    T u = us[row], acc = 0;
    const T* of_piece = coef + (long long)piece * width * 6;
    for (int m = lane; m < width; m += 32) {
        const T* c = of_piece + m * 6;
        acc += c[0] + u * (c[1] + u * (c[2] + u * (c[3] + u * (c[4] + u * c[5]))));
        if (TANGENT)
            acc += T(0.5)
                * (c[1] + u * (T(2) * c[2] + u * (T(3) * c[3] + u * (T(4) * c[4] + u * T(5) * c[5]))));
    }
    acc = warp_total(acc);
    if (lane == 0) out[row] = acc;
}

template <typename T, bool TWO>
__device__ void stored(const T* g, const T* dg, int width, long long rows, T* out) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long row = (long long)blockIdx.x * warps + warp;
    if (row >= rows) return;
    T acc = 0;
    for (int m = lane; m < width; m += 32) {
        acc += g[row * width + m];
        if (TWO) acc += T(0.5) * dg[row * width + m];
    }
    acc = warp_total(acc);
    if (lane == 0) out[row] = acc;
}

// The values and tangents of every row and output written out, as the
// kernel of the tables does it.
template <typename T>
__device__ void written(const int* pieces, const T* us, const T* coef, int width, long long rows,
        T* g, T* dg) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long row = (long long)blockIdx.x * warps + warp;
    if (row >= rows) return;
    int piece = pieces[row];
    T u = us[row];
    const T* of_piece = coef + (long long)piece * width * 6;
    for (int m = lane; m < width; m += 32) {
        const T* c = of_piece + m * 6;
        long long idx = row * width + m;
        g[idx] = c[0] + u * (c[1] + u * (c[2] + u * (c[3] + u * (c[4] + u * c[5]))));
        dg[idx] = T(0.5)
            * (c[1] + u * (T(2) * c[2] + u * (T(3) * c[3] + u * (T(4) * c[4] + u * T(5) * c[5]))));
    }
}

#define PROBE(T, S)                                                                           \
extern "C" __global__ void value_##S(const int* p, const T* u, const T* c, int w,              \
        long long r, T* o) { lookup<T, false>(p, u, c, w, r, o); }                             \
extern "C" __global__ void value_tangent_##S(const int* p, const T* u, const T* c, int w,      \
        long long r, T* o) { lookup<T, true>(p, u, c, w, r, o); }                              \
extern "C" __global__ void one_load_##S(const T* g, const T* dg, int w, long long r, T* o) {   \
    stored<T, false>(g, dg, w, r, o); }                                                        \
extern "C" __global__ void two_loads_##S(const T* g, const T* dg, int w, long long r, T* o) {  \
    stored<T, true>(g, dg, w, r, o); }                                                         \
extern "C" __global__ void written_##S(const int* p, const T* u, const T* c, int w,            \
        long long r, T* g, T* dg) { written<T>(p, u, c, w, r, g, dg); }
PROBE(float, f32)
PROBE(double, f64)
"#;

/// What the kernels of an se_e2_a descriptor with tables would pay to look
/// their embedding values up themselves instead of reading them stored: a
/// warp a row and the lanes over the outputs, as the kernel of the tables
/// and the force kernel go, over a million rows of 100 outputs. Per row:
/// the quintic of every output looked up (value, and value and tangent),
/// one or two stored values read, and the values and tangents written as
/// the kernel of the tables does. Tables of 178 pieces (the water model) and
/// of 4,576 (a trained HAP model), the pieces of the rows at random or in
/// runs of 64 rows.
#[test]
#[ignore]
fn embedding_table_costs() {
    let device = Device::new(0).unwrap();
    let stream = device.stream().clone();
    let rows: usize = 1 << 20;
    let width: usize = 100;
    let warps_per_block = 8usize;
    let config = LaunchConfig {
        grid_dim: (rows.div_ceil(warps_per_block) as u32, 1, 1),
        block_dim: (32 * warps_per_block as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let rows_i = rows as i64;
    let width_i = width as i32;
    macro_rules! run {
        ($t:ty, $s:literal) => {{
            let f = |name: &str| {
                device
                    .kernels()
                    .function(TABLE_PROBE, &format!("{name}_{}", $s))
                    .unwrap()
            };
            let (value, value_tangent, one_load, two_loads, written) = (
                f("value"),
                f("value_tangent"),
                f("one_load"),
                f("two_loads"),
                f("written"),
            );
            let us: Vec<$t> = (0..rows).map(|_| (next() >> 11) as $t / (1u64 << 53) as $t).collect();
            let us = device.upload(&us).unwrap();
            let mut g = device.zeros::<$t>(rows * width).unwrap();
            let mut dg = device.zeros::<$t>(rows * width).unwrap();
            let mut out = device.zeros::<$t>(rows).unwrap();
            for pieces_n in [178usize, 4576] {
                let coef: Vec<$t> = (0..pieces_n * width * 6)
                    .map(|_| (next() >> 11) as $t / (1u64 << 53) as $t - 0.5)
                    .collect();
                let coef = device.upload(&coef).unwrap();
                for runs in [1usize, 64] {
                    let pieces: Vec<i32> = (0..rows / runs)
                        .flat_map(|_| {
                            let p = (next() % pieces_n as u64) as i32;
                            std::iter::repeat(p).take(runs)
                        })
                        .collect();
                    let pieces = device.upload(&pieces).unwrap();
                    let mut time = |label: &str, launch: &mut dyn FnMut()| {
                        for _ in 0..3 {
                            launch();
                        }
                        device.synchronize().unwrap();
                        let start = Instant::now();
                        let n = 20;
                        for _ in 0..n {
                            launch();
                        }
                        device.synchronize().unwrap();
                        let us_each = start.elapsed().as_secs_f64() * 1e6 / n as f64;
                        println!(
                            "{:>4} {:>5} pieces, runs of {:>2}: {label:<24} {us_each:8.1} us, {:6.3} ns a row",
                            $s,
                            pieces_n,
                            runs,
                            us_each * 1e3 / rows as f64
                        );
                    };
                    time("value", &mut || {
                        let mut l = stream.launch_builder(&value);
                        l.arg(&pieces).arg(&us).arg(&coef).arg(&width_i).arg(&rows_i).arg(&mut out);
                        unsafe { l.launch(config) }.map(drop).unwrap();
                    });
                    time("value and tangent", &mut || {
                        let mut l = stream.launch_builder(&value_tangent);
                        l.arg(&pieces).arg(&us).arg(&coef).arg(&width_i).arg(&rows_i).arg(&mut out);
                        unsafe { l.launch(config) }.map(drop).unwrap();
                    });
                    time("written (table kernel)", &mut || {
                        let mut l = stream.launch_builder(&written);
                        l.arg(&pieces)
                            .arg(&us)
                            .arg(&coef)
                            .arg(&width_i)
                            .arg(&rows_i)
                            .arg(&mut g)
                            .arg(&mut dg);
                        unsafe { l.launch(config) }.map(drop).unwrap();
                    });
                }
            }
            let mut time = |label: &str, launch: &mut dyn FnMut()| {
                for _ in 0..3 {
                    launch();
                }
                device.synchronize().unwrap();
                let start = Instant::now();
                let n = 20;
                for _ in 0..n {
                    launch();
                }
                device.synchronize().unwrap();
                let us_each = start.elapsed().as_secs_f64() * 1e6 / n as f64;
                println!(
                    "{:>4} stored: {label:<38} {us_each:8.1} us, {:6.3} ns a row",
                    $s,
                    us_each * 1e3 / rows as f64
                );
            };
            time("one load", &mut || {
                let mut l = stream.launch_builder(&one_load);
                l.arg(&g).arg(&dg).arg(&width_i).arg(&rows_i).arg(&mut out);
                unsafe { l.launch(config) }.map(drop).unwrap();
            });
            time("two loads", &mut || {
                let mut l = stream.launch_builder(&two_loads);
                l.arg(&g).arg(&dg).arg(&width_i).arg(&rows_i).arg(&mut out);
                unsafe { l.launch(config) }.map(drop).unwrap();
            });
        }};
    }
    run!(f32, "f32");
    run!(f64, "f64");
}
