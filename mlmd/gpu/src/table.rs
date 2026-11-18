//! Tabulated scalar-input networks (plan I-5, DeePMD `dp compress`).
//!
//! An embedding net is a smooth function of one number, so it can be
//! replaced by a piecewise quintic polynomial per output: on each interval
//! of width `step` the polynomial matches the value, first and second
//! derivative at both ends. With `step = 0.01` the difference from a smooth
//! network is near rounding level, at about a tenth of the arithmetic and
//! none of its intermediate storage.
//!
//! A table may continue past its upper end with a coarser tail (DeePMD's
//! extrapolation region), so rare large inputs stay tabulated without paying
//! for fine knots there. Several tables of the same width form one
//! [`ScalarTable`]; each row names the table it uses, so one launch serves
//! every embedding net.
//!
//! Inputs outside a table are clamped and flagged; the caller decides what
//! to do (the se_e2_a path re-evaluates that frame with the exact network).
//!
//! A [`CellTable`] is for a net whose input spans a range far wider than the
//! finest of its features (the product of two directions that se_e3 gives
//! its nets): its pieces are of uneven length, and the piece of an input is
//! found from the bits of the input, with nothing rounded on the way.
use crate::{check_len, cuda_error, Device, Scalar};
use cudarc::driver::{
    CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use omnimd_ml_contract::{Error, Result};

const SOURCE: &str = r#"
// coef: [intervals][width][6] for every table in turn, polynomial in the
// position within an interval, u in [0, 1]. Table t: shape[3t..] = {intervals,
// tail intervals, first interval in coef}, range[3t..] = {lower, 1/step,
// step/tail step}. The tail intervals follow the main ones.
//
// One warp per row: the row's table and interval are found once per lane,
// then the lanes stride over the outputs, so coefficients are read and
// results written coalesced. A row with a negative id is skipped: nothing is
// written for it.
template <typename T>
__device__ void scalar_table(const T* x, const int* ids, long long rows, const T* coef,
        int width, const int* shape, const T* range, T* out, T* dout, int* flag) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long row = (long long)blockIdx.x * warps + warp;
    if (row >= rows || (ids && ids[row] < 0)) return;
    int table = ids ? ids[row] : 0;
    int intervals = shape[3 * table], tail = shape[3 * table + 1];
    long long first = shape[3 * table + 2];
    T inverse_step = range[3 * table + 1];
    T position = (x[row] - range[3 * table]) * inverse_step;
    T slope_scale = inverse_step;
    int count = intervals;
    if (tail > 0 && position > T(intervals)) {
        position = (position - T(intervals)) * range[3 * table + 2];
        slope_scale = inverse_step * range[3 * table + 2];
        count = tail;
        first += intervals;
    }
    // Rounding may leave an input a hair outside; anything more is reported.
    if (!(position >= T(-1e-3) && position <= T(count) + T(1e-3))) *flag = 1;
    if (!(position > T(0))) position = T(0);
    if (position > T(count)) position = T(count);
    int i = (int)position;
    if (i >= count) i = count - 1;
    T u = position - T(i);
    const T* interval = coef + (first + i) * width * 6;
    for (int m = lane; m < width; m += 32) {
        const T* c = interval + m * 6;
        long long idx = row * width + m;
        out[idx] = c[0] + u * (c[1] + u * (c[2] + u * (c[3] + u * (c[4] + u * c[5]))));
        dout[idx] = slope_scale
            * (c[1] + u * (T(2) * c[2] + u * (T(3) * c[3] + u * (T(4) * c[4] + u * T(5) * c[5]))));
    }
}

#define INSTANTIATE(T, S)                                                                    \
extern "C" __global__ void scalar_table_##S(const T* x, const int* ids, long long rows,      \
        const T* coef, int width, const int* shape, const T* range, T* out, T* dout,         \
        int* flag) {                                                                         \
    scalar_table<T>(x, ids, rows, coef, width, shape, range, out, dout, flag);               \
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

/// One table of a [`ScalarTable`]: `coefficients` from
/// [`quintic_coefficients`] for knots every `step` from `lower`, continued
/// by `tail_coefficients` (possibly empty) for knots every `tail_step` from
/// the last main knot.
#[derive(Clone, Debug)]
pub struct TableSegments {
    pub lower: f64,
    pub step: f64,
    pub coefficients: Vec<f64>,
    pub tail_step: f64,
    pub tail_coefficients: Vec<f64>,
}

/// Piecewise quintic tables of one output width; rows select a table by
/// index.
pub struct ScalarTable<T: Scalar> {
    width: usize,
    /// Per table: (main intervals, tail intervals).
    intervals: Vec<(usize, usize)>,
    shape: CudaSlice<i32>,
    range: CudaSlice<T>,
    coefficients: CudaSlice<T>,
    function: CudaFunction,
}

impl<T: Scalar> ScalarTable<T> {
    /// A single table without a tail; `coefficients` from
    /// [`quintic_coefficients`].
    pub fn new(
        device: &Device,
        lower: f64,
        step: f64,
        width: usize,
        coefficients: &[f64],
    ) -> Result<Self> {
        Self::with_tail(device, lower, step, width, coefficients, step, &[])
    }

    /// A single table whose last knot is the first knot of a tail with
    /// intervals of width `tail_step`.
    pub fn with_tail(
        device: &Device,
        lower: f64,
        step: f64,
        width: usize,
        coefficients: &[f64],
        tail_step: f64,
        tail_coefficients: &[f64],
    ) -> Result<Self> {
        Self::new_set(
            device,
            width,
            &[TableSegments {
                lower,
                step,
                coefficients: coefficients.to_vec(),
                tail_step,
                tail_coefficients: tail_coefficients.to_vec(),
            }],
        )
    }

    /// Several tables with `width` outputs each.
    pub fn new_set(device: &Device, width: usize, tables: &[TableSegments]) -> Result<Self> {
        let per_interval = 6 * width;
        if width == 0 || tables.is_empty() || i32::try_from(width).is_err() {
            return Err(Error("a scalar table set needs tables and a width".into()));
        }
        let (mut shape, mut range) = (Vec::new(), Vec::new());
        let (mut intervals, mut values) = (Vec::new(), Vec::<T>::new());
        for table in tables {
            let main = table.coefficients.len() / per_interval;
            let tail = table.tail_coefficients.len() / per_interval;
            let first = values.len() / per_interval;
            if main == 0
                || table.coefficients.len() != main * per_interval
                || table.tail_coefficients.len() != tail * per_interval
                || i32::try_from(first + main + tail).is_err()
                || !(table.lower.is_finite() && table.step.is_finite() && table.step > 0.0)
                || !(table.tail_step.is_finite() && table.tail_step > 0.0)
            {
                return Err(Error("invalid scalar table shape or range".into()));
            }
            shape.extend([main as i32, tail as i32, first as i32]);
            range.extend([
                T::from_f64(table.lower),
                T::from_f64(1.0 / table.step),
                T::from_f64(table.step / table.tail_step),
            ]);
            values.extend(
                table
                    .coefficients
                    .iter()
                    .chain(&table.tail_coefficients)
                    .map(|&c| T::from_f64(c)),
            );
            intervals.push((main, tail));
        }
        Ok(Self {
            width,
            intervals,
            shape: device.upload(&shape)?,
            range: device.upload(&range)?,
            coefficients: device.upload(&values)?,
            function: device
                .kernels()
                .function(SOURCE, &format!("scalar_table_{}", T::SUFFIX))?,
        })
    }

    pub fn output(&self) -> usize {
        self.width
    }

    pub fn tables(&self) -> usize {
        self.intervals.len()
    }

    /// Intervals of the first table's main part (its tail is not counted).
    pub fn intervals(&self) -> usize {
        self.intervals[0].0
    }

    pub fn tail_intervals(&self) -> usize {
        self.intervals[0].1
    }

    /// Device bytes held by the coefficients.
    pub fn bytes(&self) -> usize {
        self.coefficients.len() * std::mem::size_of::<T>()
    }

    /// `out[rows][width]` and its derivative with respect to `x[rows]`, for a
    /// single table. `flag[0]` is set to 1 when some input lies outside the
    /// table (its row is then evaluated at the nearest end).
    pub fn forward_tangent_into(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        rows: usize,
        out: &mut impl DevicePtrMut<T>,
        dout: &mut impl DevicePtrMut<T>,
        flag: &mut impl DevicePtrMut<i32>,
    ) -> Result<()> {
        self.launch(device, x, None::<&CudaSlice<i32>>, rows, out, dout, flag)
    }

    /// Like [`Self::forward_tangent_into`], with row `r` looked up in table
    /// `ids[r]` (which must be below [`Self::tables`]). A row with a negative
    /// id is skipped: its outputs are left as they were.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tangent_by_id(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        ids: &impl DevicePtr<i32>,
        rows: usize,
        out: &mut impl DevicePtrMut<T>,
        dout: &mut impl DevicePtrMut<T>,
        flag: &mut impl DevicePtrMut<i32>,
    ) -> Result<()> {
        self.launch(device, x, Some(ids), rows, out, dout, flag)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        ids: Option<&impl DevicePtr<i32>>,
        rows: usize,
        out: &mut impl DevicePtrMut<T>,
        dout: &mut impl DevicePtrMut<T>,
        flag: &mut impl DevicePtrMut<i32>,
    ) -> Result<()> {
        check_len("scalar table input", x.len(), rows)?;
        check_len("scalar table output", out.len(), rows * self.width)?;
        check_len("scalar table tangent", dout.len(), rows * self.width)?;
        match ids {
            Some(ids) => check_len("table ids", ids.len(), rows)?,
            None if self.tables() == 1 => {}
            None => {
                return Err(Error(format!(
                    "a set of {} tables needs one table id per row",
                    self.tables()
                )))
            }
        }
        // One warp per row.
        let blocks = rows.div_ceil(BLOCK as usize / 32);
        if rows == 0 || blocks > u32::MAX as usize || flag.len() == 0 {
            return Err(Error(format!(
                "scalar table over {rows} rows is out of range"
            )));
        }
        let stream = device.stream();
        let (px, _g0) = x.device_ptr(stream);
        let ids = ids.map(|i| i.device_ptr(stream));
        let pids = ids.as_ref().map_or(0, |(pointer, _)| *pointer);
        let (pout, _g1) = out.device_ptr_mut(stream);
        let (pdout, _g2) = dout.device_ptr_mut(stream);
        let (pflag, _g3) = flag.device_ptr_mut(stream);
        let (rows_i, width_i) = (rows as i64, self.width as i32);
        let mut launch = stream.launch_builder(&self.function);
        launch
            .arg(&px)
            .arg(&pids)
            .arg(&rows_i)
            .arg(&self.coefficients)
            .arg(&width_i)
            .arg(&self.shape)
            .arg(&self.range)
            .arg(&pout)
            .arg(&pdout)
            .arg(&pflag);
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

/// CUDA source of the lookup of a [`CellTable`], for kernels that go through
/// the table themselves on the buffers of [`CellTable::buffers`].
pub const CELL_LOOKUP: &str = r#"
// A table in quintic pieces of uneven length, all of whose ends are whole
// multiples of powers of two. The piece of an input is found from the input
// itself: its sign, its exponent and its first `bits` bits after that name a
// cell, and the pieces of a cell are of one width, 1 / scale[cell], counted
// from zero: piece = base[cell] + floor(|x| scale[cell]), and the fraction
// that is left is the place in the piece. A float holds the ends of the
// pieces exactly and the product with a power of two too, so nothing is
// rounded on the way, in either precision and for an input of any size.
//
// Table t: shape[3t..] = {its first cell, cells of one sign, the key of the
// cell before its cell 1}, range[2t..] = {the end of the table, the largest
// number below it}. The cells of x >= 0 come first, then those of x < 0,
// whose pieces are laid out along |x| as well.
//
// The key of |x|: its exponent and first bits as a float32 has them (a
// float64 has its exponent 896 higher).
__device__ int cell_key(float a, int bits) { return __float_as_int(a) >> (23 - bits); }
__device__ int cell_key(double a, int bits) {
    return (__double2hiint(a) >> (20 - bits)) - (896 << bits);
}

// The piece of `x` in table `table`, its place `u` there (0 to 1, rising with
// |x|) and du/dx. An input outside the table (or not a number) sets *flag,
// if there is one, and is looked up at the end of the table.
template <typename T>
__device__ int cell_piece(T x, int table, int bits, const int* shape, const T* range,
        const int* base, const T* scale, T* u, T* du, int* flag) {
    const int* s = shape + 3 * table;
    T a = fabs(x);
    if (!(a < range[2 * table])) {
        if (flag) *flag = 1;
        a = range[2 * table + 1];
    }
    int cells = s[1];
    int c = cell_key(a, bits) - s[2];
    c = c < 0 ? 0 : (c < cells ? c : cells - 1);
    int cell = s[0] + (x < T(0) ? cells + c : c);
    T inverse = scale[cell];
    T t = a * inverse, whole = floor(t);
    *u = t - whole;
    *du = x < T(0) ? -inverse : inverse;
    return base[cell] + (int)whole;
}
"#;

const CELL_SOURCE: &str = r#"
// coef: [pieces][width][6], the quintic of every output in the place within
// the piece. One warp per row, as for the tables of one step: the lanes
// stride over the outputs. A row with a negative id is skipped.
template <typename T>
__device__ void cell_table(const T* x, const int* ids, long long rows, const T* coef, int width,
        int bits, const int* shape, const T* range, const int* base, const T* scale, T* out,
        T* dout, int* flag) {
    int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    long long row = (long long)blockIdx.x * warps + warp;
    if (row >= rows || (ids && ids[row] < 0)) return;
    T u, du;
    int piece = cell_piece(x[row], ids ? ids[row] : 0, bits, shape, range, base, scale, &u, &du,
                           flag);
    const T* of_piece = coef + (long long)piece * width * 6;
    for (int m = lane; m < width; m += 32) {
        const T* c = of_piece + m * 6;
        long long idx = row * width + m;
        out[idx] = c[0] + u * (c[1] + u * (c[2] + u * (c[3] + u * (c[4] + u * c[5]))));
        dout[idx] = du
            * (c[1] + u * (T(2) * c[2] + u * (T(3) * c[3] + u * (T(4) * c[4] + u * T(5) * c[5]))));
    }
}

#define INSTANTIATE(T, S)                                                                    \
extern "C" __global__ void cell_table_##S(const T* x, const int* ids, long long rows,        \
        const T* coef, int width, int bits, const int* shape, const T* range,                \
        const int* base, const T* scale, T* out, T* dout, int* flag) {                       \
    cell_table<T>(x, ids, rows, coef, width, bits, shape, range, base, scale, out, dout,     \
                  flag);                                                                     \
}
INSTANTIATE(float, f32)
INSTANTIATE(double, f64)
"#;

/// One table of a [`CellTable`]: quintic pieces of uneven length over
/// `|x| < 2^top`, and the cells that tell the piece of an input.
#[derive(Clone, Debug)]
pub struct CellSegments {
    pub top: i32,
    /// Cell 0 of either sign holds the inputs below `2^lowest`; from there
    /// on every exponent has `2^key_bits` cells, one for each value of the
    /// first `key_bits` bits of the input after its exponent.
    pub lowest: i32,
    /// `(base, e)` of every cell, those of `x >= 0` first, then as many of
    /// `x < 0`: the pieces of the cell are `2^e` wide, and an input is in
    /// piece `base + floor(|x| / 2^e)` of this table.
    pub cells: Vec<(i64, i32)>,
    /// `[pieces][width][6]`: the quintic of every output in the place within
    /// the piece, 0 to 1, which rises with `|x|` on either side of zero.
    pub coefficients: Vec<f64>,
}

/// The device buffers of a [`CellTable`], for a kernel that looks inputs up
/// itself with [`CELL_LOOKUP`].
pub struct CellBuffers<'a, T: Scalar> {
    pub key_bits: i32,
    pub shape: &'a CudaSlice<i32>,
    pub range: &'a CudaSlice<T>,
    pub base: &'a CudaSlice<i32>,
    pub scale: &'a CudaSlice<T>,
    pub coefficients: &'a CudaSlice<T>,
}

/// Tables in pieces of uneven length, all of one output width; rows select
/// a table by index. For nets whose input spans a range far wider than the
/// finest of their features (see [`CELL_LOOKUP`] for the layout).
pub struct CellTable<T: Scalar> {
    width: usize,
    key_bits: u32,
    tables: usize,
    pieces: usize,
    shape: CudaSlice<i32>,
    range: CudaSlice<T>,
    base: CudaSlice<i32>,
    scale: CudaSlice<T>,
    coefficients: CudaSlice<T>,
    function: CudaFunction,
}

impl<T: Scalar> CellTable<T> {
    /// Tables of `width` outputs whose cells go by `key_bits` bits. Table
    /// `t` of the set is `segments[of[t]]`, so several tables can be one.
    pub fn new_set(
        device: &Device,
        width: usize,
        key_bits: u32,
        segments: &[CellSegments],
        of: &[usize],
    ) -> Result<Self> {
        let per_piece = 6 * width;
        let single = std::mem::size_of::<T>() == 4;
        if width == 0
            || i32::try_from(width).is_err()
            || !(1..=12).contains(&key_bits)
            || segments.is_empty()
            || of.is_empty()
            || of.iter().any(|&segment| segment >= segments.len())
        {
            return Err(Error(
                "a cell table set needs tables, a width and 1 to 12 key bits".into(),
            ));
        }
        let invalid = |what: &str| Error(format!("invalid cell table: {what}"));
        let per_exponent = 1i64 << key_bits;
        let (mut first_cell, mut base, mut scale) = (Vec::new(), Vec::<i32>::new(), Vec::new());
        let mut values = Vec::<T>::new();
        for segment in segments {
            let pieces = (segment.coefficients.len() / per_piece) as i64;
            let exponents = i64::from(segment.top) - i64::from(segment.lowest);
            // What a float32 holds as a power of two, with room to spare.
            if !(-100..=100).contains(&segment.lowest)
                || !(0..=200).contains(&exponents)
                || segment.top > 100
            {
                return Err(invalid("the range"));
            }
            let cells = 1 + exponents * per_exponent;
            if pieces == 0
                || segment.coefficients.len() != pieces as usize * per_piece
                || segment.cells.len() as i64 != 2 * cells
                || segment.coefficients.iter().any(|c| !c.is_finite())
            {
                return Err(invalid("the shape"));
            }
            let first_piece = (values.len() / per_piece) as i64;
            first_cell.push(base.len());
            for (index, &(cell_base, level)) in segment.cells.iter().enumerate() {
                // The cell as widths of 2^own from zero, and the pieces its
                // first and its last input are in: both must be the table's.
                let cell = index as i64 % cells;
                let (own, number) = match cell {
                    0 => (i64::from(segment.lowest), 0),
                    _ => (
                        i64::from(segment.lowest) + (cell - 1) / per_exponent - i64::from(key_bits),
                        per_exponent + (cell - 1) % per_exponent,
                    ),
                };
                let shift = own - i64::from(level);
                if !(-100..=100).contains(&level) || !(-60..=40).contains(&shift) {
                    return Err(invalid("the width of a piece"));
                }
                let (first, last) = match shift >= 0 {
                    true => (number << shift, ((number + 1) << shift) - 1),
                    false => (number >> -shift, number >> -shift),
                };
                let global = first_piece + cell_base;
                if cell_base + first < 0
                    || cell_base + last >= pieces
                    || i32::try_from(global).is_err()
                    || i32::try_from(global + last).is_err()
                {
                    return Err(invalid("a cell names a piece the table does not have"));
                }
                base.push(global as i32);
                scale.push(T::from_f64(2f64.powi(-level)));
            }
            values.extend(segment.coefficients.iter().map(|&c| T::from_f64(c)));
        }
        if i32::try_from(base.len()).is_err() {
            return Err(invalid("too many cells"));
        }
        let (mut shape, mut range) = (Vec::new(), Vec::new());
        for &segment in of {
            let data = &segments[segment];
            let cells = data.cells.len() / 2;
            let end = 2f64.powi(data.top);
            // The largest number below the end that the precision holds.
            let below = end
                * (1.0
                    - if single {
                        2f64.powi(-24)
                    } else {
                        2f64.powi(-53)
                    });
            shape.extend([
                first_cell[segment] as i32,
                cells as i32,
                ((127 + data.lowest) << key_bits) - 1,
            ]);
            range.extend([T::from_f64(end), T::from_f64(below)]);
        }
        Ok(Self {
            width,
            key_bits,
            tables: of.len(),
            pieces: values.len() / per_piece,
            shape: device.upload(&shape)?,
            range: device.upload(&range)?,
            base: device.upload(&base)?,
            scale: device.upload(&scale)?,
            coefficients: device.upload(&values)?,
            function: device.kernels().function(
                &format!("{CELL_LOOKUP}{CELL_SOURCE}"),
                &format!("cell_table_{}", T::SUFFIX),
            )?,
        })
    }

    pub fn output(&self) -> usize {
        self.width
    }

    pub fn tables(&self) -> usize {
        self.tables
    }

    /// Pieces of all tables together (a table that is several of the set
    /// counts once).
    pub fn pieces(&self) -> usize {
        self.pieces
    }

    /// Device bytes held by the coefficients and the cells.
    pub fn bytes(&self) -> usize {
        (self.coefficients.len() + self.scale.len()) * std::mem::size_of::<T>()
            + self.base.len() * 4
    }

    /// The buffers, for a kernel that reads the tables itself.
    pub fn buffers(&self) -> CellBuffers<'_, T> {
        CellBuffers {
            key_bits: self.key_bits as i32,
            shape: &self.shape,
            range: &self.range,
            base: &self.base,
            scale: &self.scale,
            coefficients: &self.coefficients,
        }
    }

    /// `out[rows][width]` and its derivative with respect to `x[rows]`, row
    /// `r` looked up in table `ids[r]` (which must be below
    /// [`Self::tables`]). A row with a negative id is skipped: its outputs
    /// are left as they were. `flag[0]` is set to 1 when some input lies
    /// outside its table (its row is then evaluated at the end of the
    /// table).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tangent_by_id(
        &self,
        device: &Device,
        x: &impl DevicePtr<T>,
        ids: &impl DevicePtr<i32>,
        rows: usize,
        out: &mut impl DevicePtrMut<T>,
        dout: &mut impl DevicePtrMut<T>,
        flag: &mut impl DevicePtrMut<i32>,
    ) -> Result<()> {
        check_len("cell table input", x.len(), rows)?;
        check_len("cell table ids", ids.len(), rows)?;
        check_len("cell table output", out.len(), rows * self.width)?;
        check_len("cell table tangent", dout.len(), rows * self.width)?;
        // One warp per row.
        let blocks = rows.div_ceil(BLOCK as usize / 32);
        if rows == 0 || blocks > u32::MAX as usize || flag.len() == 0 {
            return Err(Error(format!(
                "cell table over {rows} rows is out of range"
            )));
        }
        let stream = device.stream();
        let (px, _g0) = x.device_ptr(stream);
        let (pids, _g1) = ids.device_ptr(stream);
        let (pout, _g2) = out.device_ptr_mut(stream);
        let (pdout, _g3) = dout.device_ptr_mut(stream);
        let (pflag, _g4) = flag.device_ptr_mut(stream);
        let (rows_i, width_i, bits_i) = (rows as i64, self.width as i32, self.key_bits as i32);
        let mut launch = stream.launch_builder(&self.function);
        launch
            .arg(&px)
            .arg(&pids)
            .arg(&rows_i)
            .arg(&self.coefficients)
            .arg(&width_i)
            .arg(&bits_i)
            .arg(&self.shape)
            .arg(&self.range)
            .arg(&self.base)
            .arg(&self.scale)
            .arg(&pout)
            .arg(&pdout)
            .arg(&pflag);
        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(config) }
            .map(drop)
            .map_err(cuda_error("cell table launch"))
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
