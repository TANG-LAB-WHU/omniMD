//! Tabulated scalar-input networks (plan I-5, DeePMD `dp compress`).
//!
//! An embedding net is a function of one number, so it can be replaced by a
//! piecewise quintic polynomial per output, at about a tenth of the
//! arithmetic of the net and none of its intermediate storage.
//!
//! A trained net turns within a hundredth of a unit of its input in some
//! places and not at all in most, and the input of some nets spans a range
//! far wider than the finest of their features (the product of two
//! directions that se_e3 gives its nets). So the pieces of a [`CellTable`]
//! are of uneven length, each as long as its net allows (whoever makes the
//! table says where they end), and the piece of an input is found from the
//! bits of the input, with nothing rounded on the way. Several tables of the
//! same width form one [`CellTable`]; each row names the table it uses, so
//! one launch serves every embedding net.
//!
//! An input beyond the reach of its table is looked up at the end of that
//! reach and flagged; the caller decides what to do (the descriptors
//! re-evaluate that frame with the exact networks).
use crate::{check_len, cuda_error, Device, Scalar};
use cudarc::driver::{
    CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use omnimd_ml_contract::{Error, Result};

const BLOCK: u32 = 256;

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
// cell before its cell 1}, range[4t..] = {how far the table reaches for
// x >= 0, the largest number below that, and the same two for x < 0}. The
// cells of x >= 0 come first, then those of x < 0, whose pieces are laid out
// along |x| as well. A table may reach less far than its pieces go (they end
// at a power of two), and not at all on a side: its reach there is zero,
// and the cells of that side name pieces of the other, so that a lookup
// stays within the table.
//
// The key of |x|: its exponent and first bits as a float32 has them (a
// float64 has its exponent 896 higher).
__device__ int cell_key(float a, int bits) { return __float_as_int(a) >> (23 - bits); }
__device__ int cell_key(double a, int bits) {
    return (__double2hiint(a) >> (20 - bits)) - (896 << bits);
}

// The piece of `x` in table `table`, its place `u` there (0 to 1, rising with
// |x|) and du/dx. An input beyond the reach of the table (or not a number)
// sets *flag, if there is one, and is looked up at the end of that reach.
template <typename T>
__device__ int cell_piece(T x, int table, int bits, const int* shape, const T* range,
        const int* base, const T* scale, T* u, T* du, int* flag) {
    const int* s = shape + 3 * table;
    const T* reach = range + 4 * table + (x < T(0) ? 2 : 0);
    T a = fabs(x);
    if (!(a < reach[0])) {
        if (flag) *flag = 1;
        a = reach[1];
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

// The quintic of an output, `c` its six coefficients, at the place `u` in
// its piece, and its derivative along the input, du/dx being `du`. Every
// kernel that stores values of a table or uses them where it looks them up
// evaluates them so, and gets the same numbers.
template <typename T>
__device__ T cell_value(const T* c, T u) {
    return c[0] + u * (c[1] + u * (c[2] + u * (c[3] + u * (c[4] + u * c[5]))));
}
template <typename T>
__device__ T cell_tangent(const T* c, T u, T du) {
    return du
        * (c[1] + u * (T(2) * c[2] + u * (T(3) * c[3] + u * (T(4) * c[4] + u * T(5) * c[5]))));
}
"#;

const CELL_SOURCE: &str = r#"
// coef: [pieces][width][6], the quintic of every output in the place within
// the piece. One warp per row: the piece of the row is found once per lane,
// then the lanes stride over the outputs, so coefficients are read and
// results written coalesced. A row with a negative id is skipped: nothing is
// written for it.
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
        out[idx] = cell_value(c, u);
        dout[idx] = cell_tangent(c, u, du);
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
    /// How far the table reaches for `x >= 0` and for `x < 0`: an input at
    /// or beyond that is reported as outside. At most `2^top`, and zero for
    /// a side the table has no pieces of (its cells then name pieces of the
    /// other side).
    pub reach: [f64; 2],
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
            let end = 2f64.powi(segment.top);
            // (A reach of zero is a side without pieces; any other must be a
            // number that a float32 holds.)
            let reaches = |reach: f64| (1e-30..=end).contains(&reach);
            if !reaches(segment.reach[0]) || !(segment.reach[1] == 0.0 || reaches(segment.reach[1]))
            {
                return Err(invalid("the reach"));
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
        // The largest number below `reach` that the precision holds (zero
        // for a reach of zero), next to the reach as the precision has it.
        let ends = |reach: f64| -> [T; 2] {
            let below = match single {
                _ if reach == 0.0 => 0.0,
                true => f64::from(f32::from_bits((reach as f32).to_bits() - 1)),
                false => f64::from_bits(reach.to_bits() - 1),
            };
            [T::from_f64(reach), T::from_f64(below)]
        };
        let (mut shape, mut range) = (Vec::new(), Vec::new());
        for &segment in of {
            let data = &segments[segment];
            let cells = data.cells.len() / 2;
            shape.extend([
                first_cell[segment] as i32,
                cells as i32,
                ((127 + data.lowest) << key_bits) - 1,
            ]);
            range.extend(ends(data.reach[0]));
            range.extend(ends(data.reach[1]));
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
