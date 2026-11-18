//! Tables of an embedding net in pieces of uneven length.
//!
//! The two-body descriptors give their embedding nets a normalized radial
//! entry of some tens of units at most, and their tables have a knot every
//! hundredth of a unit. The θ of se_e3, the product of two normalized
//! directions, spans thousands of units (±1,400 for neighbors from 0.9 Å in
//! the HAP models, whose directions have a width of 0.03), and a trained net
//! turns within a hundredth of a unit where its data is and not at all over
//! most of the rest. A table of one step is then wrong by percent, or has
//! gigabytes.
//!
//! So the pieces here are of uneven length. The range ±2^top is halved again
//! and again, and a piece is kept when
//! - no unit of the net that may be in the bent part of its activation moves
//!   by more than [`MOVE`] in its argument over the piece (bounds by interval
//!   arithmetic, [`Network::bent_movement`]), so that the net has no feature
//!   between the points the piece is checked at, and
//! - the quintic through the value, slope and curvature of the net at the
//!   ends of the piece is close to the net at a quarter, the middle and three
//!   quarters of it: the value within the tolerance, as a fraction of the
//!   largest value the net was seen to take, and the slope within
//!   [`SLOPE_TOLERANCE`] times that fraction of its largest slope.
//!
//! A table counts from an origin of its own ([`TableSpan`]). The nets of
//! se_e3 have theirs at θ = 0 and pieces on either side. Those of se_e2_a
//! and se_e2_r have it at the entry of a neighbor at the cutoff, the lowest
//! their input takes, and pieces above it only: the place in the table is
//! then s(r) over the width of the radial entry, which the device has
//! without a subtraction. (The pieces are for the same reason as for se_e3:
//! a table with a knot every hundredth of a unit follows the nets of a
//! trained HAP model to 2e-5 of their largest value at its worst place,
//! where these follow them to a tolerance.)
//!
//! A net is not computed more exactly than its sums allow: a unit with a
//! large weight and a large bias, which cancel where it turns, is exact to
//! some 2e-16 of the bias. So a piece may also be off by the bound on that
//! rounding at the places it is checked at and at its ends
//! ([`Network::scalar_rounding`]), and its slope by what the rounding of
//! the values at its ends comes to over its width; no halving brings a
//! quintic closer to the net than the net is to itself.
//!
//! All ends of pieces are whole multiples of powers of two, which a float
//! holds exactly, and the piece of an input is found from the input itself:
//! its sign, its exponent and the first [`TableOptions::key_bits`] bits after
//! that name a cell of a directory, and the cell says how wide its pieces are
//! and where they start, `piece = base + floor(|x| / width)`. What is left of
//! `|x| / width` is the place in the piece. Nothing is rounded on the way, in
//! float32 or in float64, whatever the size of the input; a table that counts
//! steps from the lower end of a range of thousands rounds a float32 input
//! to a thousandth of a unit. For this a cell that holds pieces of several
//! lengths has them all cut to its shortest. In a prototype of this builder
//! that added 0.2 percent of pieces for the nets of a trained HAP model; a
//! net with a feature far narrower than a cell, far from zero, pays for the
//! whole cell.
use crate::Network;
use omnimd_ml_contract::{Error, Result};

/// The most the argument of a bent activation may move over one piece.
const MOVE: f64 = 2.0;
/// The slope of a piece may be off by this many times the tolerance, as a
/// fraction of the largest slope of the net.
pub(crate) const SLOPE_TOLERANCE: f64 = 100.0;
/// A table has no more pieces than this, and no more than have their
/// coefficients in this many bytes of float64.
const MOST_PIECES: usize = 1 << 20;
const MOST_BYTES: usize = 256 << 20;
/// The piece at zero is no narrower than two to this power.
const NARROWEST: i32 = -100;

/// How a table is made.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TableOptions {
    /// The table is within this of the net, as a fraction of the largest
    /// value the net takes (see the module for the slope).
    pub tolerance: f64,
    /// The first bits of an input, after its exponent, that tell its cell.
    pub key_bits: u32,
    /// The far end of a piece is at most two to this power times its width:
    /// what the precision of the input leaves of bits for the place within a
    /// piece. A net that needs narrower pieces is refused.
    pub resolution: u32,
}

/// What a table covers of the input of its net.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TableSpan {
    /// The pieces reach `2^top` from the origin.
    pub top: i32,
    /// Where the table counts from: the place `x` of the table is the input
    /// `origin + x` of the net.
    pub origin: f64,
    /// Whether the places below the origin have pieces too. Without, the
    /// cells of that side name the pieces of the other (a lookup then stays
    /// within the table whatever it is given), and whoever looks a place up
    /// must not take one below the origin for good.
    pub below: bool,
}

/// A net as pieces of quintics over the places `|x| < 2^top` of a table.
// (The tests read all of it; the potential takes the cells and the pieces.)
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Debug)]
pub(crate) struct NetTable {
    /// Outputs of the net.
    pub width: usize,
    pub top: i32,
    pub origin: f64,
    /// Cell 0 of either sign holds the places below `2^lowest`; from there
    /// on every exponent has `2^key_bits` cells.
    pub lowest: i32,
    pub key_bits: u32,
    /// `(base, e)` of every cell, those of `x >= 0` first, then as many of
    /// `x < 0`: the pieces of the cell are `2^e` wide and a place is in
    /// piece `base + floor(|x| / 2^e)`.
    pub cells: Vec<(i64, i32)>,
    /// `[pieces][width][6]`: the quintic of every output in the place within
    /// the piece, 0 to 1, which rises with `|x|`.
    pub coefficients: Vec<f64>,
    /// The largest value and slope the net was seen to take: what the
    /// tolerance is a fraction of.
    pub value_scale: f64,
    pub slope_scale: f64,
}

fn pow2(exponent: i32) -> f64 {
    2f64.powi(exponent)
}

/// Value, slope and curvature of every output along `|x|` on one side of
/// zero, and the bounds on the rounding of value and slope there.
struct Jet {
    value: Vec<f64>,
    slope: Vec<f64>,
    curvature: Vec<f64>,
    value_rounding: f64,
    slope_rounding: f64,
}

/// A piece the halving kept: `index` widths of `2^level` from zero.
struct Leaf {
    level: i32,
    index: u64,
    coefficients: Vec<f64>,
}

impl Leaf {
    fn start(&self) -> f64 {
        self.index as f64 * pow2(self.level)
    }

    fn end(&self) -> f64 {
        (self.index + 1) as f64 * pow2(self.level)
    }
}

/// The quintic of every output through the jets at the ends of a piece of
/// width `h`, in the place within the piece.
fn quintic(h: f64, from: &Jet, to: &Jet) -> Vec<f64> {
    let mut coefficients = Vec::with_capacity(6 * from.value.len());
    for m in 0..from.value.len() {
        let c0 = from.value[m];
        let c1 = from.slope[m] * h;
        let c2 = from.curvature[m] * h * h / 2.0;
        // What the quadratic start still misses at the far end.
        let d0 = to.value[m] - c0 - c1 - c2;
        let d1 = to.slope[m] * h - c1 - 2.0 * c2;
        let d2 = to.curvature[m] * h * h - 2.0 * c2;
        coefficients.extend([
            c0,
            c1,
            c2,
            10.0 * d0 - 4.0 * d1 + d2 / 2.0,
            -15.0 * d0 + 7.0 * d1 - d2,
            6.0 * d0 - 3.0 * d1 + d2 / 2.0,
        ]);
    }
    coefficients
}

/// A quintic and its derivative at `u`.
fn at(c: &[f64], u: f64) -> (f64, f64) {
    let value = c[0] + u * (c[1] + u * (c[2] + u * (c[3] + u * (c[4] + u * c[5]))));
    let slope = c[1] + u * (2.0 * c[2] + u * (3.0 * c[3] + u * (4.0 * c[4] + u * 5.0 * c[5])));
    (value, slope)
}

/// The quintic `c` on the part of its piece from `from` on that is `part`
/// of the piece long, in the place within that part.
fn restricted(c: &[f64], from: f64, part: f64) -> [f64; 6] {
    const CHOOSE: [[f64; 6]; 6] = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 0.0, 1.0, 3.0, 6.0, 10.0],
        [0.0, 0.0, 0.0, 1.0, 4.0, 10.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 5.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ];
    let mut out = [0.0; 6];
    let mut scale = 1.0;
    for (n, value) in out.iter_mut().enumerate() {
        // The n-th derivative at `from` over n!, by Horner from the top.
        let mut sum = 0.0;
        for k in (n..6).rev() {
            sum = sum * from + CHOOSE[n][k] * c[k];
        }
        *value = sum * scale;
        scale *= part;
    }
    out
}

struct Builder<'a> {
    network: &'a Network,
    options: TableOptions,
    /// Where the table counts from, and which side of that the pieces are
    /// on.
    origin: f64,
    sign: f64,
    value_scale: f64,
    slope_scale: f64,
    leaves: Vec<Leaf>,
    /// The most pieces the table may have.
    most: usize,
}

impl Builder<'_> {
    fn jet(&mut self, a: f64) -> Result<Jet> {
        let input = self.origin + self.sign * a;
        let [value, slope, curvature] = self.network.scalar_jet(input)?;
        for (v, d) in value.iter().zip(&slope) {
            self.value_scale = self.value_scale.max(v.abs());
            self.slope_scale = self.slope_scale.max(d.abs());
        }
        let (value_rounding, slope_rounding) = self.network.scalar_rounding(input)?;
        let sign = self.sign;
        Ok(Jet {
            value,
            slope: slope.into_iter().map(|d| sign * d).collect(),
            curvature,
            value_rounding,
            slope_rounding,
        })
    }

    /// Whether the quintic of a piece of width `h` is close enough to the
    /// net at the place `u`, where the net has the jet `exact`: within the
    /// tolerance, or within the rounding of the net there and at the ends
    /// of the piece (`rounding`: of value and slope). The quintic goes
    /// through the values at the ends as they are computed: what those are
    /// off by shows in its slope over the width of the piece (up to 15/8 of
    /// the difference of the two), the more the narrower the piece is, so
    /// that halving a piece cannot bring its slope closer than that.
    fn close(
        &self,
        coefficients: &[f64],
        h: f64,
        u: f64,
        exact: &Jet,
        rounding: (f64, f64),
    ) -> bool {
        let value_limit =
            self.options.tolerance * self.value_scale + rounding.0 + exact.value_rounding;
        let slope_limit = SLOPE_TOLERANCE * self.options.tolerance * self.slope_scale
            + rounding.1
            + exact.slope_rounding
            + 4.0 * rounding.0 / h;
        coefficients.chunks_exact(6).enumerate().all(|(m, c)| {
            let (value, slope) = at(c, u);
            (value - exact.value[m]).abs() <= value_limit
                && (slope / h - exact.slope[m]).abs() <= slope_limit
        })
    }

    /// The pieces of `[index, index + 1]·2^level`, whose ends and middle
    /// have the jets given, in the order of `|x|`.
    fn refine(&mut self, level: i32, index: u64, jets: [&Jet; 3]) -> Result<()> {
        let [from, middle, to] = jets;
        let h = pow2(level);
        let start = index as f64 * h;
        let first = self.jet(start + 0.25 * h)?;
        let third = self.jet(start + 0.75 * h)?;
        let coefficients = quintic(h, from, to);
        let (lo, hi) = match self.sign > 0.0 {
            true => (self.origin + start, self.origin + start + h),
            false => (self.origin - (start + h), self.origin - start),
        };
        // What the ends of the piece, which the quintic goes through, are
        // themselves off by.
        let ends = (
            from.value_rounding.max(to.value_rounding),
            from.slope_rounding.max(to.slope_rounding),
        );
        if self.network.bent_movement(lo, hi) <= MOVE
            && self.close(&coefficients, h, 0.25, &first, ends)
            && self.close(&coefficients, h, 0.5, middle, ends)
            && self.close(&coefficients, h, 0.75, &third, ends)
        {
            self.leaves.push(Leaf {
                level,
                index,
                coefficients,
            });
            return Ok(());
        }
        let near = self.origin + self.sign * (start + 0.5 * h);
        if 2 * index + 2 > 1u64 << self.options.resolution || level <= NARROWEST {
            return Err(Error(format!(
                "the net turns too fast near {near} for a table in this precision: its pieces there would be narrower than {:e}",
                h / 2.0
            )));
        }
        if self.leaves.len() >= self.most {
            return Err(Error(format!(
                "the net needs more than {} pieces for a table within {:e} (reached {near})",
                self.most, self.options.tolerance
            )));
        }
        self.refine(level - 1, 2 * index, [from, &first, middle])?;
        self.refine(level - 1, 2 * index + 1, [middle, &third, to])
    }

    /// The pieces of one side of zero.
    fn side(&mut self, sign: f64, top: i32) -> Result<Vec<Leaf>> {
        self.sign = sign;
        let from = self.jet(0.0)?;
        let middle = self.jet(pow2(top - 1))?;
        let to = self.jet(pow2(top))?;
        self.refine(top, 0, [&from, &middle, &to])?;
        Ok(std::mem::take(&mut self.leaves))
    }
}

/// The cells of one side of zero and their pieces, which are added to
/// `coefficients`: the leaves as they are, or cut to the shortest of their
/// cell.
fn directory(
    leaves: &[Leaf],
    lowest: i32,
    top: i32,
    key_bits: u32,
    width: usize,
    most: usize,
    coefficients: &mut Vec<f64>,
) -> Result<Vec<(i64, i32)>> {
    let per_exponent = 1usize << key_bits;
    let cells = 1 + (top - lowest) as usize * per_exponent;
    let mut records = Vec::with_capacity(cells);
    let pieces = |coefficients: &Vec<f64>| coefficients.len() / (6 * width);
    // The leaf that holds the start of the cell, and where the last leaf
    // written whole is.
    let mut leaf = 0;
    let mut whole: Option<(usize, usize)> = None;
    for cell in 0..cells {
        // The cell is `number` widths of 2^level from zero.
        let (level, number) = match cell {
            0 => (lowest, 0u64),
            _ => (
                lowest + ((cell - 1) / per_exponent) as i32 - key_bits as i32,
                (per_exponent + (cell - 1) % per_exponent) as u64,
            ),
        };
        let start = number as f64 * pow2(level);
        let end = (number + 1) as f64 * pow2(level);
        while leaves[leaf].end() <= start {
            leaf += 1;
        }
        let here = &leaves[leaf];
        if here.level >= level {
            // The cell lies within one leaf.
            let piece = match whole {
                Some((written, piece)) if written == leaf => piece,
                _ => {
                    let piece = pieces(coefficients);
                    coefficients.extend(&here.coefficients);
                    whole = Some((leaf, piece));
                    piece
                }
            };
            records.push((piece as i64 - here.index as i64, here.level));
            continue;
        }
        // The cell holds several leaves: all are cut to the shortest.
        let mut last = leaf;
        while last + 1 < leaves.len() && leaves[last + 1].start() < end {
            last += 1;
        }
        let inside = &leaves[leaf..=last];
        let finest = inside.iter().map(|l| l.level).min().unwrap_or(level);
        let first = pieces(coefficients);
        if first + (1usize << (level - finest).min(40)) > most {
            return Err(Error(format!(
                "the net needs more than {most} pieces for a table: it turns within {:e} near {start}",
                pow2(finest)
            )));
        }
        for l in inside {
            let parts = 1u64 << (l.level - finest);
            let part = 1.0 / parts as f64;
            for k in 0..parts {
                for c in l.coefficients.chunks_exact(6) {
                    coefficients.extend(restricted(c, k as f64 * part, part));
                }
            }
        }
        // floor(|x| / 2^finest) at the start of the cell.
        let before = number << (level - finest);
        records.push((first as i64 - before as i64, finest));
        leaf = last;
    }
    Ok(records)
}

impl NetTable {
    /// `network` (of one input) as a table over `span`.
    pub(crate) fn build(network: &Network, span: TableSpan, options: TableOptions) -> Result<Self> {
        let top = span.top;
        let inputs = network.layers.first().map(|layer| layer.weights.len());
        let width = network.layers.last().map_or(0, |layer| layer.bias.len());
        let usable = options.tolerance.is_finite()
            && options.tolerance > 0.0
            && span.origin.is_finite()
            && (1..=12).contains(&options.key_bits)
            && (4..=28).contains(&options.resolution)
            && (NARROWEST + 1..=120).contains(&top);
        if !usable || width == 0 || inputs != Some(1) {
            return Err(Error(format!(
                "a table needs a net of one input, a tolerance above zero and a range it can hold, got tolerance {:e} and 2^{top} from {}",
                options.tolerance, span.origin
            )));
        }
        let most = (MOST_BYTES / (48 * width)).clamp(1, MOST_PIECES);
        let mut builder = Builder {
            network,
            options,
            origin: span.origin,
            sign: 1.0,
            value_scale: 0.0,
            slope_scale: 0.0,
            leaves: Vec::new(),
            most,
        };
        let positive = builder.side(1.0, top)?;
        let negative = match span.below {
            true => Some(builder.side(-1.0, top)?),
            false => None,
        };
        let lowest = negative
            .as_ref()
            .map_or(positive[0].level, |n| positive[0].level.min(n[0].level));
        let mut coefficients = Vec::new();
        let mut cells = directory(
            &positive,
            lowest,
            top,
            options.key_bits,
            width,
            most,
            &mut coefficients,
        )?;
        match &negative {
            Some(negative) => cells.extend(directory(
                negative,
                lowest,
                top,
                options.key_bits,
                width,
                most,
                &mut coefficients,
            )?),
            // No pieces below the origin: the cells of that side are those
            // of the other.
            None => cells.extend_from_within(..),
        }
        if coefficients.iter().any(|c| !c.is_finite()) {
            return Err(Error("nonfinite table coefficient".into()));
        }
        Ok(Self {
            width,
            top,
            origin: span.origin,
            lowest,
            key_bits: options.key_bits,
            cells,
            coefficients,
            value_scale: builder.value_scale,
            slope_scale: builder.slope_scale,
        })
    }

    #[cfg(test)]
    pub(crate) fn pieces(&self) -> usize {
        self.coefficients.len() / (6 * self.width)
    }

    /// The piece of `x` and the place in it, as the device kernel finds
    /// them: `None` outside the table.
    #[cfg(test)]
    fn piece(&self, x: f64) -> Option<(usize, f64, f64)> {
        let a = x.abs();
        if !(a < pow2(self.top)) {
            return None;
        }
        let per_side = self.cells.len() / 2;
        // Exponent and first bits: cell 0 below 2^lowest, then in order.
        let key = (a.to_bits() >> (52 - self.key_bits)) as i64;
        let first_key = ((1023 + self.lowest as i64) << self.key_bits) - 1;
        let cell = (key - first_key).clamp(0, per_side as i64 - 1) as usize;
        let (base, level) = self.cells[if x < 0.0 { per_side + cell } else { cell }];
        let t = a * pow2(-level);
        let whole = t.floor();
        Some(((base + whole as i64) as usize, t - whole, pow2(-level)))
    }

    /// Value and slope of every output at `x` from the table.
    #[cfg(test)]
    pub(crate) fn evaluate(&self, x: f64) -> Option<(Vec<f64>, Vec<f64>)> {
        let (piece, u, inverse) = self.piece(x)?;
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let block = &self.coefficients[piece * 6 * self.width..][..6 * self.width];
        Some(
            block
                .chunks_exact(6)
                .map(|c| {
                    let (value, slope) = at(c, u);
                    (value, sign * inverse * slope)
                })
                .unzip(),
        )
    }

    /// The same in float32 throughout, as a float32 potential does it: the
    /// cell from the bits of the float32 input, float32 coefficients and
    /// arithmetic.
    #[cfg(test)]
    pub(crate) fn evaluate_f32(&self, x: f32) -> Option<(Vec<f32>, Vec<f32>)> {
        let a = x.abs();
        if !(a < pow2(self.top) as f32) {
            return None;
        }
        let per_side = self.cells.len() / 2;
        let key = i64::from(a.to_bits() >> (23 - self.key_bits));
        let first_key = ((127 + self.lowest as i64) << self.key_bits) - 1;
        let cell = (key - first_key).clamp(0, per_side as i64 - 1) as usize;
        let (base, level) = self.cells[if x < 0.0 { per_side + cell } else { cell }];
        let inverse = pow2(-level) as f32;
        let t = a * inverse;
        let whole = t.floor();
        let u = t - whole;
        let piece = (base + whole as i64) as usize;
        let sign = if x < 0.0 { -1.0f32 } else { 1.0 };
        let block = &self.coefficients[piece * 6 * self.width..][..6 * self.width];
        Some(
            block
                .chunks_exact(6)
                .map(|c| {
                    let c: Vec<f32> = c.iter().map(|&v| v as f32).collect();
                    let value = c[0] + u * (c[1] + u * (c[2] + u * (c[3] + u * (c[4] + u * c[5]))));
                    let slope = c[1]
                        + u * (2.0 * c[2] + u * (3.0 * c[3] + u * (4.0 * c[4] + u * 5.0 * c[5])));
                    (value, sign * inverse * slope)
                })
                .unzip(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{NetTable, TableOptions, TableSpan, SLOPE_TOLERANCE};
    use crate::{Activation, Layer, Network, Residual};

    /// Numbers between -1 and 1 that depend on nothing but `seed`.
    fn noise(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn layer(
        input: usize,
        output: usize,
        activation: Activation,
        residual: Residual,
        gain: f64,
        seed: &mut u64,
    ) -> Layer {
        Layer {
            weights: (0..input)
                .map(|_| (0..output).map(|_| gain * noise(seed)).collect())
                .collect(),
            bias: (0..output).map(|_| noise(seed)).collect(),
            activation,
            residual,
            timestep: None,
        }
    }

    /// An embedding net as the trainer makes them, [4, 8, 16] with tanh and
    /// the doubled residual, with first-layer weights up to `gain`.
    fn embedding(gain: f64, seed: u64) -> Network {
        let mut seed = seed;
        Network {
            layers: vec![
                layer(1, 4, Activation::Tanh, Residual::None, gain, &mut seed),
                layer(4, 8, Activation::Tanh, Residual::Concat2, 1.0, &mut seed),
                layer(8, 16, Activation::Tanh, Residual::Concat2, 1.0, &mut seed),
            ],
        }
    }

    const DOUBLE: TableOptions = TableOptions {
        tolerance: 1e-11,
        key_bits: 6,
        resolution: 28,
    };

    /// A table of `|x| < 2^top` about zero.
    fn both(top: i32) -> TableSpan {
        TableSpan {
            top,
            origin: 0.0,
            below: true,
        }
    }

    /// The largest difference between table and net over the places
    /// `points` of the table, as fractions of the scales of the table:
    /// (value, slope).
    fn worst(table: &NetTable, network: &Network, points: impl Iterator<Item = f64>) -> (f64, f64) {
        let mut worst = (0.0_f64, 0.0_f64);
        for x in points {
            let (value, slope) = table.evaluate(x).unwrap();
            let [exact, exact_slope, _] = network.scalar_jet(table.origin + x).unwrap();
            for m in 0..table.width {
                worst.0 = worst.0.max((value[m] - exact[m]).abs() / table.value_scale);
                worst.1 = worst
                    .1
                    .max((slope[m] - exact_slope[m]).abs() / table.slope_scale);
            }
        }
        worst
    }

    /// Checks that the cells of a table name every piece of it and no
    /// other (a piece that no cell names is one no input can reach), and
    /// returns the number of cells that hold several pieces.
    fn cells_of_several_pieces(table: &NetTable) -> usize {
        let pieces = table.pieces();
        let per_exponent = 1usize << table.key_bits;
        let per_side = table.cells.len() / 2;
        assert_eq!(
            per_side,
            1 + (table.top - table.lowest) as usize * per_exponent
        );
        let mut named = vec![false; pieces];
        let mut several = 0;
        for (index, &(base, width)) in table.cells.iter().enumerate() {
            // The cell is `number` widths of 2^level from zero.
            let (level, number) = match index % per_side {
                0 => (table.lowest, 0i64),
                cell => (
                    table.lowest + ((cell - 1) / per_exponent) as i32 - table.key_bits as i32,
                    (per_exponent + (cell - 1) % per_exponent) as i64,
                ),
            };
            let (first, last) = match level - width {
                shift if shift >= 0 => (number << shift, ((number + 1) << shift) - 1),
                shift => (number >> -shift, number >> -shift),
            };
            several += usize::from(last > first);
            for piece in base + first..=base + last {
                named[usize::try_from(piece).unwrap()] = true;
            }
        }
        assert!(
            named.iter().all(|&n| n),
            "{} of {pieces} pieces have no cell",
            named.iter().filter(|&&n| !n).count()
        );
        several
    }

    /// Points all over `|x| < reach`, and as many again packed near zero.
    fn spread(reach: f64, count: usize, seed: u64) -> impl Iterator<Item = f64> {
        let mut seed = seed;
        (0..2 * count).map(move |k| {
            let x = noise(&mut seed);
            if k % 2 == 0 {
                reach * x
            } else {
                // From 2^-30 of the reach on, either sign.
                reach * x.signum() * 2f64.powf(-30.0 * x.abs())
            }
        })
    }

    #[test]
    fn a_table_is_within_its_tolerance_of_the_net_everywhere() {
        // Weights like those of a trained net of se_e3: some units turn
        // within a fraction of a unit of θ, others over hundreds.
        for (gain, seed) in [(4.5, 3), (0.02, 4), (60.0, 5)] {
            let network = embedding(gain, seed);
            let table = NetTable::build(&network, both(13), DOUBLE).unwrap();
            let (value, slope) = worst(&table, &network, spread(8192.0, 20_000, 17 + seed));
            // The checks of the builder are at three places of a piece; in
            // between a quintic strays a little further.
            assert!(value <= 1.5 * DOUBLE.tolerance, "gain {gain}: {value:e}");
            assert!(
                slope <= 1.5 * SLOPE_TOLERANCE * DOUBLE.tolerance,
                "gain {gain}: {slope:e}"
            );
            // Far fewer pieces than a table of one step: at the 1e-3 these
            // nets ask for near zero it would have sixteen million.
            assert!(table.pieces() < 30_000, "gain {gain}: {}", table.pieces());
            assert!(table.pieces() > 100, "gain {gain}: {}", table.pieces());
        }
    }

    #[test]
    fn a_looser_tolerance_takes_fewer_pieces_and_is_looser() {
        let network = embedding(4.5, 3);
        let fine = NetTable::build(&network, both(13), DOUBLE).unwrap();
        let coarse = NetTable::build(
            &network,
            both(13),
            TableOptions {
                tolerance: 1e-5,
                ..DOUBLE
            },
        )
        .unwrap();
        assert!(coarse.pieces() * 3 < fine.pieces());
        let (value, _) = worst(&coarse, &network, spread(8192.0, 20_000, 1));
        assert!(value <= 1.5e-5 && value > 1e-8, "{value:e}");
    }

    #[test]
    fn a_feature_between_the_points_of_a_check_is_not_missed() {
        // The first layer rises over hundreds of units; two units of the
        // second layer with a gain of a thousand switch on at |θ| = 54.9 and
        // off at 57.6. At the ends, the middle and the quarters of the range
        // and of its halves the net is flat at zero, value, slope and
        // curvature: a builder that only compared values there would keep
        // one piece that is zero everywhere. On either side of zero in turn.
        let feature = |side: f64| Network {
            layers: vec![
                Layer {
                    weights: vec![vec![side * 0.01]],
                    bias: vec![0.0],
                    activation: Activation::Tanh,
                    residual: Residual::None,
                    timestep: None,
                },
                Layer {
                    weights: vec![vec![1000.0, -1000.0]],
                    bias: vec![-500.0, 520.0],
                    activation: Activation::Tanh,
                    residual: Residual::None,
                    timestep: None,
                },
                Layer {
                    weights: vec![vec![1.0], vec![1.0]],
                    bias: vec![0.0],
                    activation: Activation::Linear,
                    residual: Residual::None,
                    timestep: None,
                },
            ],
        };
        for side in [1.0, -1.0] {
            let network = feature(side);
            assert!(network.scalar_jet(side * 256.0).unwrap()[0][0].abs() < 1e-15);
            assert!((network.scalar_jet(side * 56.0).unwrap()[0][0] - 2.0).abs() < 1e-6);
            let table = NetTable::build(&network, both(10), DOUBLE).unwrap();
            let (value, slope) = worst(
                &table,
                &network,
                (0..40_000).map(|k| side * (50.0 + k as f64 * 2.5e-4)),
            );
            assert!(value <= 1.5 * DOUBLE.tolerance, "side {side}: {value:e}");
            assert!(
                slope <= 1.5 * SLOPE_TOLERANCE * DOUBLE.tolerance,
                "side {side}: {slope:e}"
            );
            // The other side of zero has no such feature and a few pieces.
            let (value, _) = worst(&table, &network, (1..4000).map(|k| -side * k as f64 * 0.25));
            assert!(value <= 1.5 * DOUBLE.tolerance, "side {side}: {value:e}");
        }
        // A table that counts from elsewhere follows the units over the
        // inputs of a piece, not over its places. From -30 the feature is at
        // the places 84.9 to 87.6, in the piece of the places 64 to 128,
        // whose ends, middle and quarters are the inputs 34, 50, 66, 82 and
        // 98: the net is flat at zero at all of them, and has nothing
        // between the inputs 64 and 128.
        let network = feature(1.0);
        for input in [34.0, 50.0, 66.0, 82.0, 98.0] {
            let jet = network.scalar_jet(input).unwrap();
            assert!(jet.iter().all(|part| part[0].abs() < 1e-15), "{input}");
        }
        let span = TableSpan {
            top: 10,
            origin: -30.0,
            below: false,
        };
        let table = NetTable::build(&network, span, DOUBLE).unwrap();
        assert!((table.evaluate(86.0).unwrap().0[0] - 2.0).abs() < 1e-6);
        let (value, slope) = worst(
            &table,
            &network,
            (0..40_000).map(|k| 80.0 + k as f64 * 2.5e-4),
        );
        assert!(value <= 1.5 * DOUBLE.tolerance, "{value:e}");
        assert!(
            slope <= 1.5 * SLOPE_TOLERANCE * DOUBLE.tolerance,
            "{slope:e}"
        );
    }

    #[test]
    fn every_activation_residual_and_time_step_is_tabulated() {
        let mut seed = 9;
        let mut network = Network {
            layers: vec![
                layer(1, 2, Activation::Sigmoid, Residual::Concat2, 0.7, &mut seed),
                layer(
                    2,
                    4,
                    Activation::Softplus,
                    Residual::Concat2,
                    0.5,
                    &mut seed,
                ),
                layer(4, 4, Activation::Tanh, Residual::Identity, 1.0, &mut seed),
                layer(4, 3, Activation::Linear, Residual::None, 1.0, &mut seed),
            ],
        };
        network.layers[2].timestep = Some(vec![0.5, -0.25, 1.5, 0.1]);
        // The input goes on through the residuals: the net does not level
        // off, and the units of the later layers move all over the range.
        let table = NetTable::build(&network, both(6), DOUBLE).unwrap();
        let (value, slope) = worst(&table, &network, spread(64.0, 20_000, 2));
        assert!(value <= 1.5 * DOUBLE.tolerance, "{value:e}");
        assert!(
            slope <= 1.5 * SLOPE_TOLERANCE * DOUBLE.tolerance,
            "{slope:e}"
        );
    }

    #[test]
    fn pieces_meet_with_value_and_slope_and_cells_name_them() {
        let network = embedding(4.5, 3);
        let table = NetTable::build(&network, both(13), DOUBLE).unwrap();
        let pieces = table.pieces();
        // The cells name every piece: here most cells lie within one piece,
        // and with cells sixteen times as wide many hold several.
        assert!(cells_of_several_pieces(&table) < table.cells.len() / 4);
        let wide = TableOptions {
            key_bits: 2,
            ..DOUBLE
        };
        for (gain, seed) in [(4.5, 3), (60.0, 5)] {
            let table = NetTable::build(&embedding(gain, seed), both(13), wide).unwrap();
            assert!(cells_of_several_pieces(&table) > 10, "gain {gain}");
        }
        // Every piece is some input's, on its own side of zero.
        let mut seen = vec![false; pieces];
        let mut previous: Option<(usize, f64)> = None;
        let mut borders = 0;
        for k in 0..400_000 {
            // From 2^-20 to the top, geometric, then the same below zero.
            let a = 2f64.powf(-20.0 + 33.0 * (k % 200_000) as f64 / 200_000.0);
            let x = if k < 200_000 { a } else { -a };
            if k == 200_000 {
                previous = None;
            }
            let (piece, u, _) = table.piece(x).unwrap();
            assert!(
                piece < pieces && (0.0..1.0).contains(&u),
                "{x}: {piece} {u}"
            );
            seen[piece] = true;
            if let Some((before, at)) = previous.filter(|&(before, _)| before != piece) {
                // Two inputs on either side of an end of a piece: the table
                // changes by no more than the net does between them.
                let (left, left_slope) = table.evaluate(at).unwrap();
                let (right, right_slope) = table.evaluate(x).unwrap();
                let [a_value, a_slope, _] = network.scalar_jet(at).unwrap();
                let [b_value, b_slope, _] = network.scalar_jet(x).unwrap();
                for m in 0..table.width {
                    let jump = (right[m] - left[m]) - (b_value[m] - a_value[m]);
                    assert!(jump.abs() <= 3e-11 * table.value_scale, "{before}->{piece}");
                    let turn = (right_slope[m] - left_slope[m]) - (b_slope[m] - a_slope[m]);
                    assert!(turn.abs() <= 3e-9 * table.slope_scale, "{before}->{piece}");
                }
                borders += 1;
            }
            previous = Some((piece, x));
        }
        assert!(borders > pieces / 2, "{borders} of {pieces}");
        assert!(seen.iter().filter(|&&s| s).count() > pieces * 9 / 10);
        // The ends of the range: inside up to the last float, outside from
        // the end on, and zero of either sign is in the table.
        let end = 8192.0_f64;
        assert!(table.evaluate(f64::from_bits(end.to_bits() - 1)).is_some());
        assert!(table.evaluate(-f64::from_bits(end.to_bits() - 1)).is_some());
        assert!(table.evaluate(end).is_none() && table.evaluate(-end).is_none());
        assert!(table.evaluate(f64::NAN).is_none());
        let (at_zero, _) = table.evaluate(0.0).unwrap();
        let (below_zero, _) = table.evaluate(-0.0).unwrap();
        let exact = network.scalar_jet(0.0).unwrap();
        for m in 0..table.width {
            assert!((at_zero[m] - exact[0][m]).abs() <= 1e-15);
            assert!((below_zero[m] - exact[0][m]).abs() <= 1e-15);
        }
        // The smallest numbers a float holds are in cell 0.
        assert!(table.evaluate(5e-324).is_some() && table.evaluate(-1e-300).is_some());
    }

    #[test]
    fn a_table_counts_from_its_origin_and_may_have_one_side() {
        // As for a net of se_e2_a: the table counts from the entry of a
        // neighbor at the cutoff, -0.37 here, and has pieces above it only.
        let network = embedding(4.5, 3);
        let span = TableSpan {
            top: 6,
            origin: -0.37,
            below: false,
        };
        let table = NetTable::build(&network, span, DOUBLE).unwrap();
        let places = || spread(64.0, 20_000, 31).map(f64::abs);
        let (value, slope) = worst(&table, &network, places());
        assert!(value <= 1.5 * DOUBLE.tolerance, "{value:e}");
        assert!(
            slope <= 1.5 * SLOPE_TOLERANCE * DOUBLE.tolerance,
            "{slope:e}"
        );
        // The place 1 is the input 0.63 of the net, not the input 1.
        let (at_one, _) = table.evaluate(1.0).unwrap();
        let shifted = network.scalar_jet(0.63).unwrap();
        let unshifted = network.scalar_jet(1.0).unwrap();
        assert!((at_one[0] - shifted[0][0]).abs() <= 1e-10);
        assert!((at_one[0] - unshifted[0][0]).abs() > 1e-3);
        // The cells below the origin name the pieces of the other side: a
        // lookup there stays within the table. Every piece has its cell.
        let per_side = table.cells.len() / 2;
        assert_eq!(table.cells[..per_side], table.cells[per_side..]);
        cells_of_several_pieces(&table);
        // With pieces on both sides the table of the same span is larger,
        // and the same above the origin.
        let two = NetTable::build(
            &network,
            TableSpan {
                below: true,
                ..span
            },
            DOUBLE,
        )
        .unwrap();
        assert!(two.pieces() > table.pieces());
        assert_ne!(two.cells[..per_side], two.cells[two.cells.len() / 2..]);
        for place in places().step_by(97) {
            assert_eq!(table.evaluate(place), two.evaluate(place), "place {place}");
        }
        // Below the origin the two-sided table follows the net as well.
        let (value, _) = worst(&two, &network, places().map(|x| -x));
        assert!(value <= 1.5 * DOUBLE.tolerance, "{value:e}");
        // An origin that is not a number is no table.
        let nowhere = TableSpan {
            origin: f64::NAN,
            ..span
        };
        let error = NetTable::build(&network, nowhere, DOUBLE)
            .unwrap_err()
            .to_string();
        assert!(error.contains("a range it can hold"), "{error}");
    }

    #[test]
    fn a_float32_lookup_finds_the_same_pieces() {
        // What a float32 potential does: the input, the cell, the place in
        // the piece and the quintic in float32. The table is as fine as
        // float32 can use, so the difference from the net (in float64, at
        // the float32 input) is rounding: a few times 6e-8 of the value.
        let single = TableOptions {
            tolerance: 1.5e-8,
            key_bits: 4,
            resolution: 20,
        };
        for (gain, seed) in [(4.5, 3), (60.0, 5)] {
            let network = embedding(gain, seed);
            let table = NetTable::build(&network, both(13), single).unwrap();
            let mut worst = (0.0_f64, 0.0_f64);
            for x in spread(8192.0, 20_000, 23).chain(spread(1.0, 20_000, 24)) {
                let x = x as f32;
                let (value, slope) = table.evaluate_f32(x).unwrap();
                let [exact, exact_slope, _] = network.scalar_jet(f64::from(x)).unwrap();
                // The float64 lookup of the same float32 input: the same
                // piece, so the two differ by rounding alone.
                let (double, _) = table.evaluate(f64::from(x)).unwrap();
                for m in 0..table.width {
                    assert!((f64::from(value[m]) - double[m]).abs() <= 4e-6, "{x}");
                    worst.0 = worst
                        .0
                        .max((f64::from(value[m]) - exact[m]).abs() / table.value_scale);
                    worst.1 = worst
                        .1
                        .max((f64::from(slope[m]) - exact_slope[m]).abs() / table.slope_scale);
                }
            }
            assert!(worst.0 <= 1e-6, "gain {gain}: value {:e}", worst.0);
            assert!(worst.1 <= 2e-5, "gain {gain}: slope {:e}", worst.1);
        }
    }

    #[test]
    fn a_net_too_fast_for_the_precision_is_refused() {
        // A unit that switches within a hundredth of a unit at θ = 1000: a
        // float32 there has steps of 6e-5, too few for a piece. (The weight
        // has all its bits, so that its products with the places the
        // builder looks at round like those with any other.)
        let network = Network {
            layers: vec![Layer {
                weights: vec![vec![200.123456789, 0.01]],
                bias: vec![-200_123.456789, 0.0],
                activation: Activation::Tanh,
                residual: Residual::None,
                timestep: None,
            }],
        };
        let single = TableOptions {
            tolerance: 1.5e-8,
            key_bits: 4,
            resolution: 20,
        };
        let error = NetTable::build(&network, both(11), single)
            .unwrap_err()
            .to_string();
        assert!(error.contains("turns too fast near 999.9"), "{error}");
        // Float64 has the bits for it. The weight of 200 and the bias of
        // -200,000 cancel where the unit turns, which leaves the net itself
        // some 2e-11 of rounding there, above the tolerance (the bound on it
        // is 9e-11): the pieces are as close as that, in value and, over
        // their width, in slope, and no halving goes on after what cannot
        // be had. A coarser tolerance is kept. The
        // unit turns within a fifth of a unit, across the end of a cell of
        // eight: the two cells there are cut to its narrowest piece all
        // over, 131,000 pieces where the turn itself has under a thousand.
        // (A feature that narrow for its place is what these tables are
        // wasteful for. The nets of a HAP model trained with se_e3 have
        // none such: in a prototype of the builder, cutting their cells
        // alike added 0.2% of pieces.)
        let table = NetTable::build(&network, both(11), DOUBLE).unwrap();
        let near = || (0..20_000).map(|k| 999.9 + k as f64 * 1e-5);
        let (value, _) = worst(&table, &network, near());
        assert!(value <= 2e-10, "{value:e}");
        assert!(table.pieces() < 140_000, "{}", table.pieces());
        let coarser = TableOptions {
            tolerance: 1e-9,
            ..DOUBLE
        };
        let table = NetTable::build(&network, both(11), coarser).unwrap();
        let (value, _) = worst(&table, &network, near());
        assert!(value <= 1.5e-9, "{value:e}");
        // A tolerance far below the rounding of the net asks for what no
        // piece can give: without the allowance for that rounding the
        // halving would go on to the last bit of θ and fail there.
        let finer = TableOptions {
            tolerance: 1e-14,
            ..DOUBLE
        };
        let table = NetTable::build(&network, both(11), finer).unwrap();
        let (value, _) = worst(&table, &network, near());
        assert!(value <= 2e-10, "{value:e}");
        assert!(table.pieces() < 300_000, "{}", table.pieces());
        // What cannot be a table at all.
        let two_inputs = Network {
            layers: vec![Layer {
                weights: vec![vec![1.0], vec![1.0]],
                bias: vec![0.0],
                activation: Activation::Tanh,
                residual: Residual::None,
                timestep: None,
            }],
        };
        assert!(NetTable::build(&two_inputs, both(4), DOUBLE).is_err());
        for bad in [0.0, -1e-9, f64::NAN] {
            let options = TableOptions {
                tolerance: bad,
                ..DOUBLE
            };
            assert!(NetTable::build(&network, both(4), options).is_err());
        }
    }
}
