//! Same-shaped networks trained together: the se_e2_a embedding nets (one per
//! pair of types, one input) and the fitting nets (one per type).
//!
//! [`crate::TrainNetwork`] needs about ten device calls per layer for the
//! forward and reverse pass of one network, so a model with twenty-five
//! embedding and five fitting nets costs several milliseconds of call
//! overhead per training step, far more than its arithmetic in float32.
//! [`TrainSet`] takes rows that each name their network and does the forward
//! pass with tangent in one kernel, the reverse pass in a second, and the
//! parameter gradients in one matrix product per network and layer.
//!
//! The kernels keep everything a row needs in two "strips". For row `r` of
//! the value strip, row `2r` holds `[x | 1 | y_1 | 1 | ... | y_L]` and row
//! `2r + 1` the tangents `[dx | 0 | dy_1 | 0 | ... | dy_L]`: the input block of
//! every layer is followed by a constant that is one for values and zero for
//! tangents. The adjoint strip holds the adjoints of the pre-activations and
//! of their tangents the same way. With that layout the gradient of a layer's
//! weights and bias, `sum_r (a^T zbar + da^T zdbar)` and `sum_r zbar`, is the
//! product of two column blocks of the strips over the rows of one network,
//! and it lands directly on the layer's `[input + 1][output]` block of the
//! flat parameter vector (weights followed by bias).
use crate::train::{TrainLayer, ACTIVATE2};
use crate::{check_len, cuda_error, Device, MatmulShape, Scalar};
use cudarc::driver::{
    CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use omnimd_ml_contract::{Error, Result};

const SOURCE: &str = r#"
// shape: 4 ints per layer {in, out, activation, residual}. cols: 3 ints per
// layer {input column and output column in the value strip, column in the
// record and adjoint strips}. offsets: per network and layer 3 ints {weights,
// bias, timestep or -1} into theta; weights are [in][out].
//
// A group of threads per row: one warp for narrow networks (eight rows per
// block), the whole block for wide ones. Each thread owns the outputs
// j = lane, lane + group, ... of a layer, so the weight loads of the loop
// over the inputs are coalesced across the group.
struct RowGroup {
    long long row;
    int lane, stride;
    bool wide;
};

__device__ RowGroup row_group(int wide) {
    RowGroup g;
    g.wide = wide != 0;
    if (g.wide) {
        g.row = blockIdx.x;
        g.lane = threadIdx.x;
        g.stride = blockDim.x;
    } else {
        g.row = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
        g.lane = threadIdx.x & 31;
        g.stride = 32;
    }
    return g;
}

// Orders the writes of one phase before the reads of the next within a row.
__device__ void row_sync(const RowGroup& g) {
    if (g.wide) __syncthreads(); else __syncwarp();
}

template <typename T>
__device__ void set_forward(const T* x, const T* xdot, const int* ids, long long rows,
        const int* shape, const int* cols, const int* offsets, int layers, const T* theta,
        int strip_width, int record_width, T* strip, T* p1, T* p2, T* zd, T* phi, T* out,
        T* dout, int wide) {
    RowGroup group = row_group(wide);
    long long row = group.row;
    int lane = group.lane, stride = group.stride;
    if (row >= rows) return;
    T* value = strip + 2 * row * strip_width;
    T* tangent = value + strip_width;
    const int* off = offsets + 3 * layers * ids[row];
    // Inputs and their tangents (unit tangents without xdot), then the constant.
    int in0 = shape[0];
    for (int i = lane; i < in0; i += stride) {
        value[i] = x[row * in0 + i];
        tangent[i] = xdot ? xdot[row * in0 + i] : T(1);
    }
    if (lane == 0) {
        value[in0] = T(1);
        tangent[in0] = T(0);
    }
    row_sync(group);
    for (int l = 0; l < layers; ++l) {
        int in = shape[4 * l], on = shape[4 * l + 1], act = shape[4 * l + 2];
        int residual = shape[4 * l + 3];
        const T* w = theta + off[3 * l];
        const T* b = theta + off[3 * l + 1];
        const T* dt = off[3 * l + 2] >= 0 ? theta + off[3 * l + 2] : nullptr;
        const T* a = value + cols[3 * l];
        const T* ad = tangent + cols[3 * l];
        int oc = cols[3 * l + 1];
        long long rec = row * record_width + cols[3 * l + 2];
        for (int j = lane; j < on; j += stride) {
            T z = b[j], zdot = T(0);
            for (int i = 0; i < in; ++i) {
                T weight = w[(long long)i * on + j];
                z += a[i] * weight;
                zdot += ad[i] * weight;
            }
            T v, d1, d2;
            activate2(act, z, &v, &d1, &d2);
            T scale = dt ? dt[j] : T(1);
            p1[rec + j] = d1;
            p2[rec + j] = d2;
            zd[rec + j] = zdot;
            if (phi) phi[rec + j] = v;
            T y = v * scale + (residual ? a[j % in] : T(0));
            T yd = d1 * scale * zdot + (residual ? ad[j % in] : T(0));
            value[oc + j] = y;
            tangent[oc + j] = yd;
            if (l == layers - 1) {
                out[row * on + j] = y;
                dout[row * on + j] = yd;
            }
        }
        if (l < layers - 1 && lane == 0) {
            value[oc + on] = T(1);
            tangent[oc + on] = T(0);
        }
        row_sync(group);
    }
}

// Adjoints of every pre-activation (row 2r of zz) and of its tangent (row
// 2r + 1) from the adjoints of the outputs and of their tangents; `adj` is
// scratch of the same shape for the adjoints of the hidden outputs. With
// xbar, also the adjoints of the inputs and of their tangents.
template <typename T>
__device__ void set_backward(const int* ids, long long rows, const int* shape, const int* cols,
        const int* offsets, int layers, const T* theta, int record_width, const T* p1,
        const T* p2, const T* zd, const T* phi, const T* ybar, const T* ydotbar, T* zz, T* adj,
        T* dtacc, T* xbar, T* xdbar, int wide) {
    RowGroup group = row_group(wide);
    long long row = group.row;
    int lane = group.lane, stride = group.stride;
    if (row >= rows) return;
    const int* off = offsets + 3 * layers * ids[row];
    T* zb = zz + 2 * row * record_width;
    T* zdb = zb + record_width;
    T* ab = adj + 2 * row * record_width;
    T* adb = ab + record_width;
    for (int l = layers - 1; l >= 0; --l) {
        int in = shape[4 * l], on = shape[4 * l + 1], residual = shape[4 * l + 3];
        int rc = cols[3 * l + 2];
        long long rec = row * record_width + rc;
        const T* dt = off[3 * l + 2] >= 0 ? theta + off[3 * l + 2] : nullptr;
        const T* yb = l == layers - 1 ? ybar + row * on : ab + rc;
        const T* ydb = l == layers - 1 ? ydotbar + row * on : adb + rc;
        for (int j = lane; j < on; j += stride) {
            T scale = dt ? dt[j] : T(1);
            T bar = yb[j], dbar = ydb[j];
            zb[rc + j] = (bar * p1[rec + j] + dbar * p2[rec + j] * zd[rec + j]) * scale;
            zdb[rc + j] = dbar * p1[rec + j] * scale;
            if (dt) dtacc[rec + j] = bar * phi[rec + j] + dbar * p1[rec + j] * zd[rec + j];
        }
        row_sync(group);
        if (l == 0 && !xbar) break;
        const T* w = theta + off[3 * l];
        int pc = l > 0 ? cols[3 * (l - 1) + 2] : 0;
        for (int i = lane; i < in; i += stride) {
            const T* wi = w + (long long)i * on;
            T s = T(0), sd = T(0);
            for (int j = 0; j < on; ++j) {
                s += zb[rc + j] * wi[j];
                sd += zdb[rc + j] * wi[j];
            }
            if (residual) {
                for (int j = i; j < on; j += in) {
                    s += yb[j];
                    sd += ydb[j];
                }
            }
            if (l > 0) {
                ab[pc + i] = s;
                adb[pc + i] = sd;
            } else {
                xbar[row * in + i] = s;
                xdbar[row * in + i] = sd;
            }
        }
        row_sync(group);
    }
}

#define INSTANTIATE(T, S)                                                                     \
extern "C" __global__ void set_forward_##S(const T* x, const T* xdot, const int* ids,         \
        long long rows, const int* shape, const int* cols, const int* offsets, int layers,    \
        const T* theta, int strip_width, int record_width, T* strip, T* p1, T* p2, T* zd,     \
        T* phi, T* out, T* dout, int wide) {                                                  \
    set_forward<T>(x, xdot, ids, rows, shape, cols, offsets, layers, theta, strip_width,      \
                   record_width, strip, p1, p2, zd, phi, out, dout, wide);                    \
}                                                                                             \
extern "C" __global__ void set_backward_##S(const int* ids, long long rows, const int* shape, \
        const int* cols, const int* offsets, int layers, const T* theta, int record_width,    \
        const T* p1, const T* p2, const T* zd, const T* phi, const T* ybar,                   \
        const T* ydotbar, T* zz, T* adj, T* dtacc, T* xbar, T* xdbar, int wide) {             \
    set_backward<T>(ids, rows, shape, cols, offsets, layers, theta, record_width, p1, p2, zd, \
                    phi, ybar, ydotbar, zz, adj, dtacc, xbar, xdbar, wide);                   \
}
INSTANTIATE(float, f32)
INSTANTIATE(double, f64)
"#;

/// Warps per block of the row kernels.
const WARPS: usize = 8;

/// What [`TrainSet::forward`] recorded for [`TrainSet::backward`].
pub struct SetTape<T: Scalar> {
    rows: usize,
    strip: CudaSlice<T>,
    p1: CudaSlice<T>,
    p2: CudaSlice<T>,
    zd: CudaSlice<T>,
    phi: Option<CudaSlice<T>>,
    output: CudaSlice<T>,
    tangent: CudaSlice<T>,
}

impl<T: Scalar> SetTape<T> {
    /// The network outputs `[rows][output]`.
    pub fn output(&self) -> &CudaSlice<T> {
        &self.output
    }

    /// Their tangents along the input directions given to the forward pass.
    pub fn tangent(&self) -> &CudaSlice<T> {
        &self.tangent
    }
}

/// Networks of one shape over a shared flat parameter vector; every row is
/// evaluated by the network it names.
pub struct TrainSet<T: Scalar> {
    networks: Vec<Vec<TrainLayer>>,
    /// Per layer: input and output columns of the value strip and the
    /// column of the record and adjoint strips.
    columns: Vec<[usize; 3]>,
    strip_width: usize,
    record_width: usize,
    timesteps: bool,
    /// A whole thread block per row instead of one warp.
    wide: bool,
    shape: CudaSlice<i32>,
    cols: CudaSlice<i32>,
    offsets: CudaSlice<i32>,
    forward: CudaFunction,
    backward: CudaFunction,
    _scalar: std::marker::PhantomData<T>,
}

/// Layers wider than this get a whole thread block per row.
const NARROW_LIMIT: usize = 128;

fn launch_config(rows: usize, wide: bool) -> Result<LaunchConfig> {
    let blocks = if wide { rows } else { rows.div_ceil(WARPS) };
    if rows == 0 || blocks > u32::MAX as usize {
        return Err(Error(format!(
            "row kernel over {rows} rows is out of range"
        )));
    }
    Ok(LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (32 * WARPS as u32, 1, 1),
        shared_mem_bytes: 0,
    })
}

impl<T: Scalar> TrainSet<T> {
    /// `networks[k]` is the layout of network `k`. All must have the same
    /// layer shapes, and each layer's bias directly after its weights in the
    /// parameter vector.
    pub fn new(device: &Device, networks: Vec<Vec<TrainLayer>>) -> Result<Self> {
        let first = networks
            .first()
            .filter(|layers| !layers.is_empty())
            .ok_or_else(|| Error("a network set needs a network with a layer".into()))?
            .clone();
        let int = |value: usize| {
            i32::try_from(value).map_err(|_| Error("network set offset exceeds i32".into()))
        };
        let mut offsets = Vec::with_capacity(networks.len() * first.len() * 3);
        for layers in &networks {
            if layers.len() != first.len() {
                return Err(Error("networks of a set need the same depth".into()));
            }
            for (index, (layer, like)) in layers.iter().zip(&first).enumerate() {
                let chained = index == 0 || layers[index - 1].output == layer.input;
                let residual_ok = !layer.residual
                    || layer.output == layer.input
                    || layer.output == 2 * layer.input;
                if (layer.input, layer.output, layer.activation, layer.residual)
                    != (like.input, like.output, like.activation, like.residual)
                    || layer.timestep.is_some() != like.timestep.is_some()
                    || !chained
                    || !residual_ok
                    || layer.input == 0
                    || layer.output == 0
                {
                    return Err(Error(
                        "networks of a set need the same chained layer shapes".into(),
                    ));
                }
                if layer.bias != layer.weights + layer.input * layer.output {
                    return Err(Error(
                        "a network set needs each bias directly after its weights".into(),
                    ));
                }
                offsets.extend([
                    int(layer.weights)?,
                    int(layer.bias)?,
                    layer.timestep.map_or(Ok(-1), int)?,
                ]);
            }
        }
        // Value strip: [x | c | y_1 | c | ... | y_L]; records: [1 | ... | L].
        let (mut columns, mut shape, mut cols) = (Vec::new(), Vec::new(), Vec::new());
        let (mut input_column, mut record_column) = (0, 0);
        for layer in &first {
            let output_column = input_column + layer.input + 1;
            columns.push([input_column, output_column, record_column]);
            shape.extend([
                int(layer.input)?,
                int(layer.output)?,
                layer.activation.code(),
                i32::from(layer.residual),
            ]);
            cols.extend([int(input_column)?, int(output_column)?, int(record_column)?]);
            input_column = output_column;
            record_column += layer.output;
        }
        let strip_width = input_column + first.last().unwrap().output;
        int(2 * strip_width.max(record_column))?;
        let source = format!("{ACTIVATE2}{SOURCE}");
        let function = |name: &str| {
            device
                .kernels()
                .function(&source, &format!("{name}_{}", T::SUFFIX))
        };
        Ok(Self {
            timesteps: first.iter().any(|layer| layer.timestep.is_some()),
            wide: first
                .iter()
                .any(|layer| layer.input.max(layer.output) > NARROW_LIMIT),
            columns,
            strip_width,
            record_width: record_column,
            shape: device.upload(&shape)?,
            cols: device.upload(&cols)?,
            offsets: device.upload(&offsets)?,
            forward: function("set_forward")?,
            backward: function("set_backward")?,
            networks,
            _scalar: std::marker::PhantomData,
        })
    }

    pub fn networks(&self) -> &[Vec<TrainLayer>] {
        &self.networks
    }

    /// Width of the inputs.
    pub fn input(&self) -> usize {
        self.networks[0][0].input
    }

    /// Width of the outputs.
    pub fn output(&self) -> usize {
        self.networks[0].last().unwrap().output
    }

    /// Outputs `[rows][output]` and their tangents for inputs `x[rows][input]`
    /// with unit input tangents (for one input: the derivative), and the
    /// network index `ids[rows]` of every row, recording what `backward` needs.
    pub fn forward(
        &self,
        device: &Device,
        theta: &CudaSlice<T>,
        x: &impl DevicePtr<T>,
        ids: &impl DevicePtr<i32>,
        rows: usize,
    ) -> Result<SetTape<T>> {
        self.run_forward(device, theta, x, None::<&CudaSlice<T>>, ids, rows)
    }

    /// As [`forward`](Self::forward), with the input tangents `xdot[rows][input]`.
    pub fn forward_with_tangent(
        &self,
        device: &Device,
        theta: &CudaSlice<T>,
        x: &impl DevicePtr<T>,
        xdot: &impl DevicePtr<T>,
        ids: &impl DevicePtr<i32>,
        rows: usize,
    ) -> Result<SetTape<T>> {
        check_len("network set input tangent", xdot.len(), rows * self.input())?;
        self.run_forward(device, theta, x, Some(xdot), ids, rows)
    }

    fn run_forward(
        &self,
        device: &Device,
        theta: &CudaSlice<T>,
        x: &impl DevicePtr<T>,
        xdot: Option<&impl DevicePtr<T>>,
        ids: &impl DevicePtr<i32>,
        rows: usize,
    ) -> Result<SetTape<T>> {
        check_len("network set input", x.len(), rows * self.input())?;
        check_len("network set ids", ids.len(), rows)?;
        let (width, record) = (self.output(), rows * self.record_width);
        // SAFETY: the kernel writes every element of these buffers: both
        // strip rows of every row, and every record, output and tangent.
        let (mut strip, mut p1, mut p2, mut zd, mut output, mut tangent) = unsafe {
            (
                device.alloc_uninit::<T>(2 * rows * self.strip_width)?,
                device.alloc_uninit::<T>(record)?,
                device.alloc_uninit::<T>(record)?,
                device.alloc_uninit::<T>(record)?,
                device.alloc_uninit::<T>(rows * width)?,
                device.alloc_uninit::<T>(rows * width)?,
            )
        };
        let mut phi = match self.timesteps {
            true => Some(unsafe { device.alloc_uninit::<T>(record)? }),
            false => None,
        };
        let (rows_d, layers) = (rows as i64, self.networks[0].len() as i32);
        let (strip_width, record_width) = (self.strip_width as i32, self.record_width as i32);
        let null = 0u64;
        let stream = device.stream();
        // Views are passed by address; the guards keep them alive.
        let ((x, _x), (ids, _ids)) = (x.device_ptr(stream), ids.device_ptr(stream));
        let xdot = xdot.map(|xdot| xdot.device_ptr(stream));
        let mut launch = stream.launch_builder(&self.forward);
        launch.arg(&x);
        match xdot.as_ref() {
            Some((xdot, _)) => launch.arg(xdot),
            None => launch.arg(&null),
        };
        launch
            .arg(&ids)
            .arg(&rows_d)
            .arg(&self.shape)
            .arg(&self.cols)
            .arg(&self.offsets)
            .arg(&layers)
            .arg(theta)
            .arg(&strip_width)
            .arg(&record_width)
            .arg(&mut strip)
            .arg(&mut p1)
            .arg(&mut p2)
            .arg(&mut zd);
        match phi.as_mut() {
            Some(phi) => launch.arg(phi),
            None => launch.arg(&null),
        };
        let wide = i32::from(self.wide);
        launch.arg(&mut output).arg(&mut tangent).arg(&wide);
        unsafe { launch.launch(launch_config(rows, self.wide)?) }
            .map(drop)
            .map_err(cuda_error("network set forward kernel"))?;
        Ok(SetTape {
            rows,
            strip,
            p1,
            p2,
            zd,
            phi,
            output,
            tangent,
        })
    }

    /// Adds to `gradient` the derivative of a scalar with respect to every
    /// parameter, given its derivatives `ybar` with respect to the outputs
    /// and `ydotbar` with respect to their tangents (both `[rows][output]`).
    /// `blocks` lists the rows of each network as `(network, first row,
    /// rows)`; `ones` holds at least as many ones as the longest block and is
    /// read only for layers with a timestep.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        device: &Device,
        theta: &CudaSlice<T>,
        gradient: &mut CudaSlice<T>,
        tape: &SetTape<T>,
        ids: &impl DevicePtr<i32>,
        ybar: &impl DevicePtr<T>,
        ydotbar: &impl DevicePtr<T>,
        blocks: &[(usize, usize, usize)],
        ones: &CudaSlice<T>,
    ) -> Result<()> {
        let inputs = None::<(&mut CudaSlice<T>, &mut CudaSlice<T>)>;
        self.reverse(
            device, theta, gradient, true, tape, ids, ybar, ydotbar, blocks, ones, inputs,
        )
    }

    /// As [`backward`](Self::backward), also writing the derivatives with
    /// respect to the inputs (`xbar`) and to their tangents (`xdbar`), both
    /// `[rows][input]`; without `parameters` the gradient is left alone.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_to_input(
        &self,
        device: &Device,
        theta: &CudaSlice<T>,
        gradient: &mut CudaSlice<T>,
        parameters: bool,
        tape: &SetTape<T>,
        ids: &impl DevicePtr<i32>,
        ybar: &impl DevicePtr<T>,
        ydotbar: &impl DevicePtr<T>,
        blocks: &[(usize, usize, usize)],
        ones: &CudaSlice<T>,
        xbar: &mut impl DevicePtrMut<T>,
        xdbar: &mut impl DevicePtrMut<T>,
    ) -> Result<()> {
        let needed = tape.rows * self.input();
        check_len("network set input adjoint", xbar.len(), needed)?;
        check_len("network set input tangent adjoint", xdbar.len(), needed)?;
        self.reverse(
            device,
            theta,
            gradient,
            parameters,
            tape,
            ids,
            ybar,
            ydotbar,
            blocks,
            ones,
            Some((xbar, xdbar)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn reverse<A: DevicePtrMut<T>, B: DevicePtrMut<T>>(
        &self,
        device: &Device,
        theta: &CudaSlice<T>,
        gradient: &mut CudaSlice<T>,
        parameters: bool,
        tape: &SetTape<T>,
        ids: &impl DevicePtr<i32>,
        ybar: &impl DevicePtr<T>,
        ydotbar: &impl DevicePtr<T>,
        blocks: &[(usize, usize, usize)],
        ones: &CudaSlice<T>,
        inputs: Option<(&mut A, &mut B)>,
    ) -> Result<()> {
        let rows = tape.rows;
        let width = self.output();
        check_len("network set ids", ids.len(), rows)?;
        check_len("network set output adjoint", ybar.len(), rows * width)?;
        check_len("network set tangent adjoint", ydotbar.len(), rows * width)?;
        let record = rows * self.record_width;
        // SAFETY: the kernel writes both adjoint rows of every row for every
        // layer, the scratch before it reads it, and every timestep term of
        // the layers that have a timestep (the only ones read below).
        let (mut zz, mut scratch) = unsafe {
            (
                device.alloc_uninit::<T>(2 * record)?,
                device.alloc_uninit::<T>(2 * record)?,
            )
        };
        let mut dtacc = match self.timesteps {
            true => Some(unsafe { device.alloc_uninit::<T>(record)? }),
            false => None,
        };
        let (rows_d, layers) = (rows as i64, self.networks[0].len() as i32);
        let record_width = self.record_width as i32;
        let null = 0u64;
        {
            let stream = device.stream();
            let (ids, _ids) = ids.device_ptr(stream);
            let ((ybar, _ybar), (ydotbar, _ydotbar)) =
                (ybar.device_ptr(stream), ydotbar.device_ptr(stream));
            let inputs = inputs
                .map(|(xbar, xdbar)| (xbar.device_ptr_mut(stream), xdbar.device_ptr_mut(stream)));
            let mut launch = stream.launch_builder(&self.backward);
            launch
                .arg(&ids)
                .arg(&rows_d)
                .arg(&self.shape)
                .arg(&self.cols)
                .arg(&self.offsets)
                .arg(&layers)
                .arg(theta)
                .arg(&record_width)
                .arg(&tape.p1)
                .arg(&tape.p2)
                .arg(&tape.zd);
            match tape.phi.as_ref() {
                Some(phi) => launch.arg(phi),
                None => launch.arg(&null),
            };
            launch
                .arg(&ybar)
                .arg(&ydotbar)
                .arg(&mut zz)
                .arg(&mut scratch);
            match dtacc.as_mut() {
                Some(dtacc) => launch.arg(dtacc),
                None => launch.arg(&null),
            };
            match inputs.as_ref() {
                Some(((xbar, _), (xdbar, _))) => launch.arg(xbar).arg(xdbar),
                None => launch.arg(&null).arg(&null),
            };
            let wide = i32::from(self.wide);
            launch.arg(&wide);
            unsafe { launch.launch(launch_config(rows, self.wide)?) }
                .map(drop)
                .map_err(cuda_error("network set reverse kernel"))?;
        }
        if !parameters {
            return Ok(());
        }
        // One product per network and layer: its rows of the value strip
        // (inputs and the constant) against its rows of the adjoint strip.
        for &(network, first, count) in blocks {
            if count == 0 {
                continue;
            }
            if first + count > rows || network >= self.networks.len() {
                return Err(Error(format!(
                    "network set block ({network}, {first}, {count}) is out of range"
                )));
            }
            for (layer, columns) in self.networks[network].iter().zip(&self.columns) {
                let inputs = tape
                    .strip
                    .slice(2 * first * self.strip_width + columns[0]..);
                let adjoints = zz.slice(2 * first * self.record_width + columns[2]..);
                let block = (layer.input + 1) * layer.output;
                device.matmul_at_strided(
                    &inputs,
                    self.strip_width,
                    &adjoints,
                    self.record_width,
                    &mut gradient.slice_mut(layer.weights..layer.weights + block),
                    MatmulShape {
                        m: 2 * count,
                        k: layer.input + 1,
                        n: layer.output,
                    },
                    true,
                )?;
                if let (Some(timestep), Some(dtacc)) = (layer.timestep, dtacc.as_ref()) {
                    let terms = dtacc.slice(first * self.record_width + columns[2]..);
                    device.matmul_at_strided(
                        &ones.slice(0..count),
                        1,
                        &terms,
                        self.record_width,
                        &mut gradient.slice_mut(timestep..timestep + layer.output),
                        MatmulShape {
                            m: count,
                            k: 1,
                            n: layer.output,
                        },
                        true,
                    )?;
                }
            }
        }
        Ok(())
    }
}
