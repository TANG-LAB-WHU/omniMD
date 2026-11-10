//! Stochastic cell rescaling: a first-order barostat that samples the
//! isothermal-isobaric ensemble (Bernetti and Bussi, J. Chem. Phys. 153,
//! 114107 (2020); fully flexible cell: Del Tatto, Raiteri, Bernetti and Bussi,
//! Appl. Sci. 12, 1139 (2022)).
//!
//! One move deforms the cell by `D = exp(strain)` with
//!
//! ```text
//! strain = -g (P0 I - P) + sqrt(2 kT g / V) xi,    g = beta_T dt / (3 tau_p)
//! ```
//!
//! where `P` is the instantaneous pressure tensor, `V` the volume, `beta_T`
//! the compressibility, `tau_p` the relaxation time, `dt` the time between
//! moves and `xi` standard normal noise. Positions follow the cell and
//! velocities are mapped by the inverse transpose of `D`, so the momenta
//! conjugate to the fractional coordinates stay fixed and the kinetic part of
//! `P` is the plain `sum m v v^T / V`.
//!
//! With the noise matrix `xi` of nine independent components this is the
//! exponential form of the overdamped Langevin equation
//! `dh = -(beta_T / 3 tau_p) (P0 - P - kT/V) h dt + sqrt(2 kT beta_T / 3 V tau_p) dW h`
//! for the cell matrix `h`, whose stationary density is
//! `det(h)^-2 exp(-(H + P0 V) / kT)`: the Ito term of the exponential cancels
//! the `kT/V` drift. The rotation in `D` is dropped (only its symmetric
//! stretch is applied), which changes nothing physical and keeps the cell
//! from tumbling. The other couplings keep only part of the strain: its
//! diagonal (three independent noises), or one third of its trace on every
//! axis; all three give the same volume dynamics.
//!
//! A discrete move of that equation is only approximately right, so each
//! move is a proposal that a Metropolis-Hastings test accepts or refuses.
//! In the strain variables (with respect to the invariant measure of the
//! cell matrices for the full coupling) all three couplings sample the
//! density `V exp(-(H + P0 V) / kT)`; the test compares it and the
//! probabilities of the move and of its reverse, which needs the energy and
//! pressure after the move. The engine evaluates the model there anyway, so
//! the test is free, and the barostat samples its ensemble exactly for any
//! `dt / tau_p`: a setting that is too aggressive shows up as refused moves,
//! not as a wrong volume distribution.
use omnimd_ml_contract::{Error, Result};
use omnimd_sim::md::Barostat;
use rand::RngCore;
use serde::{Deserialize, Serialize};

/// 1 eV/angstrom^3 in bar (exact, from the SI value of the electron volt).
pub const BAR_PER_EV_PER_ANGSTROM3: f64 = 1.602176634e6;

/// Largest strain component one move may apply in a run.
pub(crate) const MAX_STRAIN: f64 = 0.05;

type Matrix = [[f64; 3]; 3];

const IDENTITY: Matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Which strain components the barostat moves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CellCoupling {
    /// One scale factor for all three axes; the cell keeps its shape.
    #[default]
    Isotropic,
    /// Independent scale factors along the Cartesian x, y and z axes; no shear.
    Anisotropic,
    /// All six strain components; the cell changes shape freely.
    Triclinic,
}

fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn transpose(a: &Matrix) -> Matrix {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}

/// The matrix exponential, by scaling and squaring around a Taylor series.
pub(crate) fn exponential(a: &Matrix) -> Matrix {
    let norm = a
        .iter()
        .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max);
    let mut squarings = 0;
    while norm / f64::from(1u32 << squarings) > 0.25 && squarings < 30 {
        squarings += 1;
    }
    let scale = 1.0 / f64::from(1u32 << squarings);
    let scaled: Matrix = std::array::from_fn(|i| std::array::from_fn(|j| a[i][j] * scale));
    let (mut sum, mut term) = (IDENTITY, IDENTITY);
    // 0.25^18 / 18! is below 1e-26.
    for order in 1..=18 {
        term = multiply(&term, &scaled);
        for i in 0..3 {
            for j in 0..3 {
                term[i][j] /= f64::from(order);
                sum[i][j] += term[i][j];
            }
        }
    }
    for _ in 0..squarings {
        sum = multiply(&sum, &sum);
    }
    sum
}

/// Eigenvalues and eigenvectors (columns) of a symmetric matrix, by cyclic
/// Jacobi rotations.
fn symmetric_eigen(a: &Matrix) -> ([f64; 3], Matrix) {
    let (mut a, mut vectors) = (*a, IDENTITY);
    for _ in 0..64 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        let diagonal = a[0][0].abs() + a[1][1].abs() + a[2][2].abs();
        if off <= 1e-17 * diagonal {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + theta.hypot(1.0));
            let c = 1.0 / t.hypot(1.0);
            let s = t * c;
            let rotation: Matrix = {
                let mut r = IDENTITY;
                r[p][p] = c;
                r[q][q] = c;
                r[p][q] = s;
                r[q][p] = -s;
                r
            };
            a = multiply(&transpose(&rotation), &multiply(&a, &rotation));
            vectors = multiply(&vectors, &rotation);
        }
    }
    ([a[0][0], a[1][1], a[2][2]], vectors)
}

/// The symmetric positive definite stretch `U` of the polar decomposition
/// `d = R U`: the square root of `d^T d`.
pub(crate) fn stretch(d: &Matrix) -> Matrix {
    let (values, vectors) = symmetric_eigen(&multiply(&transpose(d), d));
    let roots: Matrix =
        std::array::from_fn(|i| std::array::from_fn(|j| f64::from(i == j) * values[i].sqrt()));
    multiply(&vectors, &multiply(&roots, &transpose(&vectors)))
}

/// One standard normal variate from two 53-bit uniforms (Box-Muller). The
/// sequence is fixed by this code, so a restart reproduces it.
pub(crate) fn standard_normal(rng: &mut dyn RngCore) -> f64 {
    const UNIT: f64 = 1.0 / (1u64 << 53) as f64;
    let radius = (((rng.next_u64() >> 11) as f64 + 1.0) * UNIT).ln();
    let angle = (rng.next_u64() >> 11) as f64 * UNIT;
    (-2.0 * radius).sqrt() * (std::f64::consts::TAU * angle).cos()
}

fn determinant(a: &Matrix) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

fn inverse(a: &Matrix) -> Matrix {
    let d = determinant(a);
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            // Cofactor of (j, i) over the determinant.
            let (r0, r1) = ((j + 1) % 3, (j + 2) % 3);
            let (c0, c1) = ((i + 1) % 3, (i + 2) % 3);
            (a[r0][c0] * a[r1][c1] - a[r0][c1] * a[r1][c0]) / d
        })
    })
}

/// A uniform variate in (0, 1].
fn uniform(rng: &mut dyn RngCore) -> f64 {
    ((rng.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
}

/// The move tried last, kept until it is accepted or refused.
struct Proposal {
    /// Strain increment: the full matrix, or only its diagonal.
    strain: Matrix,
    /// Mean of the strain for the state before the move.
    drift: Matrix,
    /// Noise variance of each sampled strain component before the move.
    variance: f64,
    /// Volume before the move.
    volume: f64,
    /// The rotation left out of the deformation: `exp(strain) = rotation * stretch`.
    rotation: Matrix,
}

/// One stochastic cell rescaling move; see the module documentation.
pub(crate) struct CellRescaling<R: RngCore> {
    rng: R,
    /// Boltzmann constant times the temperature (eV).
    thermal_energy: f64,
    /// Target pressure (eV/angstrom^3).
    pressure: f64,
    /// `beta_T dt / (3 tau_p)` (angstrom^3/eV).
    gain: f64,
    coupling: CellCoupling,
    /// A move with a larger strain component is an error.
    max_strain: f64,
    proposal: Option<Proposal>,
}

impl<R: RngCore> CellRescaling<R> {
    pub fn new(
        rng: R,
        thermal_energy: f64,
        pressure: f64,
        gain: f64,
        coupling: CellCoupling,
        max_strain: f64,
    ) -> Self {
        Self {
            rng,
            thermal_energy,
            pressure,
            gain,
            coupling,
            max_strain,
            proposal: None,
        }
    }

    /// The strain components that carry independent noise. The isotropic
    /// coupling samples one value and copies it to the whole diagonal.
    fn components(&self) -> &'static [(usize, usize)] {
        const ALL: [(usize, usize); 9] = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ];
        match self.coupling {
            CellCoupling::Isotropic => &[(0, 0)],
            CellCoupling::Anisotropic => &[(0, 0), (1, 1), (2, 2)],
            CellCoupling::Triclinic => &ALL,
        }
    }

    /// Mean strain of a move from a state with this pressure tensor.
    fn drift(&self, pressure: &Matrix) -> Matrix {
        let full: Matrix = std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                let target = if a == b { self.pressure } else { 0.0 };
                -self.gain * (target - 0.5 * (pressure[a][b] + pressure[b][a]))
            })
        });
        let mean = (full[0][0] + full[1][1] + full[2][2]) / 3.0;
        match self.coupling {
            CellCoupling::Isotropic => IDENTITY.map(|row| row.map(|x| x * mean)),
            CellCoupling::Anisotropic => {
                std::array::from_fn(|a| std::array::from_fn(|b| f64::from(a == b) * full[a][b]))
            }
            CellCoupling::Triclinic => full,
        }
    }

    /// Noise variance of one sampled strain component at this volume. One
    /// third of the trace of nine unit normals has variance one third.
    fn variance(&self, volume: f64) -> f64 {
        let full = 2.0 * self.thermal_energy * self.gain / volume;
        match self.coupling {
            CellCoupling::Isotropic => full / 3.0,
            _ => full,
        }
    }
}

impl<R: RngCore> CellRescaling<R> {
    /// Sets up the move with this strain increment and returns its
    /// deformation.
    fn propose(&mut self, pressure: &Matrix, volume: f64, strain: Matrix) -> Result<Matrix> {
        if strain
            .iter()
            .flatten()
            .any(|e| !(e.abs() <= self.max_strain))
        {
            return Err(Error(format!(
                "barostat move exceeds a strain of {}: use a longer barostat_tau_fs, a smaller compressibility_per_bar or barostat_every, or relax the structure first",
                self.max_strain
            )));
        }
        let (deformation, rotation) = match self.coupling {
            CellCoupling::Triclinic => {
                let full = exponential(&strain);
                let symmetric = stretch(&full);
                (symmetric, multiply(&full, &inverse(&symmetric)))
            }
            _ => (
                std::array::from_fn(|a| {
                    std::array::from_fn(|b| f64::from(a == b) * strain[a][a].exp())
                }),
                IDENTITY,
            ),
        };
        self.proposal = Some(Proposal {
            strain,
            drift: self.drift(pressure),
            variance: self.variance(volume),
            volume,
            rotation,
        });
        Ok(deformation)
    }

    /// Logarithm of the Metropolis-Hastings ratio of a move: the density
    /// `V exp(-(H + P0 V) / kT)` after over before, times the probability of
    /// the reverse move over that of the move.
    fn log_ratio(
        &self,
        proposal: &Proposal,
        pressure: &Matrix,
        volume: f64,
        energy_change: f64,
    ) -> f64 {
        // The reverse move starts from the state reached by the full
        // deformation; the applied one differs from it by the rotation.
        let r = &proposal.rotation;
        let unrotated = multiply(r, &multiply(pressure, &transpose(r)));
        let reverse_drift = self.drift(&unrotated);
        let reverse_variance = self.variance(volume);
        let components = self.components();
        let mut log_ratio = 0.0;
        for &(a, b) in components {
            let forward = proposal.strain[a][b] - proposal.drift[a][b];
            let reverse = -proposal.strain[a][b] - reverse_drift[a][b];
            log_ratio += forward * forward / (2.0 * proposal.variance)
                - reverse * reverse / (2.0 * reverse_variance);
        }
        // One power of the volume from the density, and half a power per
        // component from the normalizations of the two Gaussians.
        log_ratio + (1.0 + 0.5 * components.len() as f64) * (volume / proposal.volume).ln()
            - (energy_change + self.pressure * (volume - proposal.volume)) / self.thermal_energy
    }
}

impl<R: RngCore> Barostat for CellRescaling<R> {
    fn deformation(&mut self, pressure: &Matrix, volume: f64) -> Result<Matrix> {
        let drift = self.drift(pressure);
        let deviation = self.variance(volume).sqrt();
        let mut strain: Matrix = [[0.0; 3]; 3];
        for &(a, b) in self.components() {
            strain[a][b] = drift[a][b] + deviation * standard_normal(&mut self.rng);
        }
        if self.coupling == CellCoupling::Isotropic {
            strain[1][1] = strain[0][0];
            strain[2][2] = strain[0][0];
        }
        self.propose(pressure, volume, strain)
    }

    /// Metropolis-Hastings test of the tried move, so the sampled ensemble
    /// does not depend on the size of the moves.
    fn accept(&mut self, pressure: &Matrix, volume: f64, energy_change: f64) -> Result<bool> {
        let proposal = self
            .proposal
            .take()
            .ok_or_else(|| Error("no cell move was tried".into()))?;
        let log_ratio = self.log_ratio(&proposal, pressure, volume, energy_change);
        // Drawn for every move, so the stream does not depend on the outcome.
        let draw = uniform(&mut self.rng).ln();
        if log_ratio.is_nan() {
            return Err(Error("barostat acceptance is undefined".into()));
        }
        Ok(draw < log_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn close(a: &Matrix, b: &Matrix, tolerance: f64) {
        for i in 0..3 {
            for j in 0..3 {
                assert!((a[i][j] - b[i][j]).abs() <= tolerance, "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn exponential_matches_closed_forms() {
        // A diagonal matrix, a nilpotent one, and a rotation generator.
        let diagonal: Matrix = [[0.3, 0.0, 0.0], [0.0, -1.2, 0.0], [0.0, 0.0, 2.5]];
        let mut expected = IDENTITY;
        for i in 0..3 {
            expected[i][i] = diagonal[i][i].exp();
        }
        close(&exponential(&diagonal), &expected, 1e-13);
        let nilpotent = [[0.0, 2.0, 3.0], [0.0, 0.0, -4.0], [0.0, 0.0, 0.0]];
        close(
            &exponential(&nilpotent),
            &[[1.0, 2.0, -1.0], [0.0, 1.0, -4.0], [0.0, 0.0, 1.0]],
            1e-13,
        );
        let angle = 0.7f64;
        let generator = [[0.0, -angle, 0.0], [angle, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let (s, c) = angle.sin_cos();
        let rotation = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        close(&exponential(&generator), &rotation, 1e-14);
        // exp(a) exp(-a) = 1 for a general matrix.
        let general = [[0.02, -0.4, 0.1], [0.3, 0.05, -0.2], [-0.1, 0.25, -0.03]];
        let negative = general.map(|row| row.map(|x| -x));
        close(
            &multiply(&exponential(&general), &exponential(&negative)),
            &IDENTITY,
            1e-14,
        );
    }

    #[test]
    fn stretch_removes_the_rotation_and_nothing_else() {
        let symmetric = [[1.3, 0.2, -0.1], [0.2, 0.9, 0.05], [-0.1, 0.05, 1.1]];
        close(&stretch(&symmetric), &symmetric, 1e-14);
        let (s, c) = 0.4f64.sin_cos();
        let rotation = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]];
        let rotated = multiply(&rotation, &symmetric);
        close(&stretch(&rotated), &symmetric, 1e-14);
        // Equal eigenvalues and an already diagonal matrix.
        close(&stretch(&rotation), &IDENTITY, 1e-14);
    }

    #[test]
    fn normal_variates_have_unit_moments() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let n = 200_000;
        let (mut sum, mut square, mut fourth) = (0.0, 0.0, 0.0);
        for _ in 0..n {
            let x = standard_normal(&mut rng);
            sum += x;
            square += x * x;
            fourth += x.powi(4);
        }
        let n = f64::from(n);
        // Standard errors: 1/sqrt(n), sqrt(2/n), sqrt(96/n).
        assert!((sum / n).abs() < 4.0 / n.sqrt());
        assert!((square / n - 1.0).abs() < 4.0 * (2.0 / n).sqrt());
        assert!((fourth / n - 3.0).abs() < 4.0 * (96.0 / n).sqrt());
    }

    /// A Gamma variate of unit scale and shape at least one (Marsaglia and Tsang).
    fn gamma(shape: f64, rng: &mut ChaCha8Rng) -> f64 {
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = standard_normal(rng);
            let v = (1.0 + c * x).powi(3);
            if v > 0.0 && uniform(rng).ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    /// Mean and its standard error from the scatter of block averages.
    fn mean_and_error(samples: &[f64]) -> (f64, f64) {
        const BLOCKS: usize = 40;
        let size = samples.len() / BLOCKS;
        let means: Vec<f64> = samples
            .chunks_exact(size)
            .map(|block| block.iter().sum::<f64>() / size as f64)
            .collect();
        let mean = means.iter().sum::<f64>() / means.len() as f64;
        let variance =
            means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (means.len() - 1) as f64;
        (mean, (variance / means.len() as f64).sqrt())
    }

    const MEAN_VOLUME: f64 = 1000.0;

    /// The state of the test system: an ideal gas of unit-mass atoms at
    /// kT = 1 whose cell has the shape energy `kappa (tr(M) - 3)` with
    /// `M = h h^T / V^(2/3)`.
    #[derive(Clone)]
    struct Gas {
        h: Matrix,
        velocities: Vec<[f64; 3]>,
        kappa: f64,
    }

    impl Gas {
        fn volume(&self) -> f64 {
            determinant(&self.h)
        }
        fn metric(&self) -> Matrix {
            let scale = self.volume().powf(-2.0 / 3.0);
            multiply(&self.h, &transpose(&self.h)).map(|row| row.map(|x| x * scale))
        }
        /// `sum v v^T`.
        fn kinetic(&self) -> Matrix {
            std::array::from_fn(|a| {
                std::array::from_fn(|b| self.velocities.iter().map(|v| v[a] * v[b]).sum())
            })
        }
        fn energy(&self) -> f64 {
            let (metric, kinetic) = (self.metric(), self.kinetic());
            let trace = |m: &Matrix| m[0][0] + m[1][1] + m[2][2];
            0.5 * trace(&kinetic) + self.kappa * (trace(&metric) - 3.0)
        }
        /// The virial is minus the strain derivative of the shape energy.
        fn pressure(&self) -> Matrix {
            let (metric, kinetic, volume) = (self.metric(), self.kinetic(), self.volume());
            let trace = metric[0][0] + metric[1][1] + metric[2][2];
            std::array::from_fn(|a| {
                std::array::from_fn(|b| {
                    let virial =
                        -self.kappa * (2.0 * metric[a][b] - f64::from(a == b) * 2.0 / 3.0 * trace);
                    (kinetic[a][b] + virial) / volume
                })
            })
        }
        fn deformed(&self, d: &Matrix) -> Self {
            let map = transpose(&inverse(d));
            Self {
                h: multiply(d, &self.h),
                velocities: self
                    .velocities
                    .iter()
                    .map(|v| std::array::from_fn(|a| (0..3).map(|b| map[a][b] * v[b]).sum()))
                    .collect(),
                kappa: self.kappa,
            }
        }
    }

    /// Samples of the test gas under the barostat and an exact resampling of
    /// the kinetic energy every twentieth move (so the velocity map of the
    /// barostat matters). `ratio` is the time between moves over the
    /// relaxation time. Returns the volumes; for an axial and a shear strain
    /// direction `A` the samples of
    /// `2 tr(A^2 M) - tr(A M) (2 kappa tr(A M) - tr(A K))`, whose mean is zero
    /// in the sampled ensemble (`K = sum v v^T`); and the accepted fraction.
    fn sample(
        coupling: CellCoupling,
        kappa: f64,
        atoms: usize,
        moves: usize,
        ratio: f64,
    ) -> ([Vec<f64>; 3], f64) {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let pressure = (atoms as f64 + 1.0) / MEAN_VOLUME;
        // The compressibility of the ideal gas at the target pressure.
        let gain = ratio / (3.0 * pressure);
        let mut barostat = CellRescaling::new(
            ChaCha8Rng::seed_from_u64(12),
            1.0,
            pressure,
            gain,
            coupling,
            1.0,
        );
        let mut gas = Gas {
            h: IDENTITY.map(|row| row.map(|x| 10.0 * x)),
            velocities: (0..atoms)
                .map(|_| std::array::from_fn(|_| standard_normal(&mut rng)))
                .collect(),
            kappa,
        };
        let mut samples = [Vec::new(), Vec::new(), Vec::new()];
        let mut accepted = 0usize;
        for step in 0..moves {
            if step % 20 == 0 {
                let kinetic: f64 = gas.velocities.iter().flatten().map(|v| 0.5 * v * v).sum();
                let scale = (gamma(1.5 * atoms as f64, &mut rng) / kinetic).sqrt();
                gas.velocities
                    .iter_mut()
                    .flatten()
                    .for_each(|v| *v *= scale);
            }
            let (metric, kinetic) = (gas.metric(), gas.kinetic());
            samples[0].push(gas.volume());
            let square = metric[0][0] + metric[1][1];
            for (slot, strain, stress) in [
                (
                    1,
                    metric[0][0] - metric[1][1],
                    kinetic[0][0] - kinetic[1][1],
                ),
                (2, 2.0 * metric[0][1], 2.0 * kinetic[0][1]),
            ] {
                samples[slot].push(2.0 * square - strain * (2.0 * kappa * strain - stress));
            }
            let d = barostat.deformation(&gas.pressure(), gas.volume()).unwrap();
            let moved = gas.deformed(&d);
            let change = moved.energy() - gas.energy();
            if barostat
                .accept(&moved.pressure(), moved.volume(), change)
                .unwrap()
            {
                gas = moved;
                accepted += 1;
            }
        }
        (samples, accepted as f64 / moves as f64)
    }

    /// The logarithm of a symmetric positive definite matrix.
    fn symmetric_log(a: &Matrix) -> Matrix {
        let (values, vectors) = symmetric_eigen(a);
        let logs: Matrix =
            std::array::from_fn(|i| std::array::from_fn(|j| f64::from(i == j) * values[i].ln()));
        multiply(&vectors, &multiply(&logs, &transpose(&vectors)))
    }

    #[test]
    fn moves_follow_the_rescaling_equation() {
        // Mean strain -g (P0 - P) and noise variance 2 kT g / V per component.
        let tensor = [
            [0.012, 0.002, -0.001],
            [0.004, 0.007, 0.003],
            [-0.001, 0.001, 0.010],
        ];
        let (target, gain, thermal, volume) = (0.008, 0.02, 0.005, 400.0);
        let variance = 2.0 * thermal * gain / volume;
        let full: Matrix = std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                -gain * (f64::from(a == b) * target - 0.5 * (tensor[a][b] + tensor[b][a]))
            })
        });
        let mean_diagonal = (full[0][0] + full[1][1] + full[2][2]) / 3.0;
        let draws = 100_000;
        for coupling in [
            CellCoupling::Isotropic,
            CellCoupling::Anisotropic,
            CellCoupling::Triclinic,
        ] {
            let mut barostat = CellRescaling::new(
                ChaCha8Rng::seed_from_u64(3),
                thermal,
                target,
                gain,
                coupling,
                1.0,
            );
            let (mut sum, mut square) = ([[0.0; 3]; 3], [[0.0; 3]; 3]);
            for _ in 0..draws {
                let d = barostat.deformation(&tensor, volume).unwrap();
                // Symmetric, and for the isotropic coupling a multiple of one.
                close(&d, &transpose(&d), 1e-15);
                if coupling == CellCoupling::Isotropic {
                    assert!(d[0][0] == d[1][1] && d[1][1] == d[2][2]);
                }
                let strain = symmetric_log(&d);
                for a in 0..3 {
                    for b in 0..3 {
                        sum[a][b] += strain[a][b];
                        square[a][b] += strain[a][b] * strain[a][b];
                    }
                }
            }
            for a in 0..3 {
                for b in 0..3 {
                    // The symmetric part of an off-diagonal pair has half the variance.
                    let (expected_mean, expected_variance) = match (coupling, a == b) {
                        (CellCoupling::Isotropic, true) => (mean_diagonal, variance / 3.0),
                        (CellCoupling::Anisotropic, true) => (full[a][a], variance),
                        (CellCoupling::Triclinic, true) => (full[a][a], variance),
                        (CellCoupling::Triclinic, false) => (full[a][b], variance / 2.0),
                        (_, false) => (0.0, 0.0),
                    };
                    let mean = sum[a][b] / f64::from(draws);
                    let spread = square[a][b] / f64::from(draws) - mean * mean;
                    let error = (expected_variance / f64::from(draws)).sqrt();
                    assert!(
                        (mean - expected_mean).abs() <= 5.0 * error + 1e-12,
                        "{coupling:?} ({a}, {b}): mean {mean} vs {expected_mean} +- {error}"
                    );
                    assert!(
                        (spread - expected_variance).abs() <= 0.03 * expected_variance + 1e-18,
                        "{coupling:?} ({a}, {b}): variance {spread} vs {expected_variance}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_move_and_its_reverse_have_opposite_ratios() {
        for coupling in [
            CellCoupling::Isotropic,
            CellCoupling::Anisotropic,
            CellCoupling::Triclinic,
        ] {
            let mut rng = ChaCha8Rng::seed_from_u64(21);
            let mut gas = Gas {
                h: [[9.0, 1.5, -0.7], [0.4, 11.0, 0.9], [-1.1, 0.3, 10.0]],
                velocities: (0..4)
                    .map(|_| std::array::from_fn(|_| standard_normal(&mut rng)))
                    .collect(),
                kappa: 3.0,
            };
            // Large moves, so the ratios are far from zero.
            let mut barostat =
                CellRescaling::new(ChaCha8Rng::seed_from_u64(9), 0.7, 0.004, 5.0, coupling, 1.0);
            let mut largest: f64 = 0.0;
            for _ in 0..25 {
                let d = barostat.deformation(&gas.pressure(), gas.volume()).unwrap();
                let moved = gas.deformed(&d);
                let change = moved.energy() - gas.energy();
                let proposal = barostat.proposal.take().unwrap();
                let forward =
                    barostat.log_ratio(&proposal, &moved.pressure(), moved.volume(), change);
                // The strain that undoes the move, in the frame of the moved state.
                let r = proposal.rotation;
                let back_strain = multiply(&transpose(&r), &multiply(&proposal.strain, &r))
                    .map(|row| row.map(|x| -x));
                let back = barostat
                    .propose(&moved.pressure(), moved.volume(), back_strain)
                    .unwrap();
                close(&multiply(&back, &d), &IDENTITY, 1e-12);
                let proposal = barostat.proposal.take().unwrap();
                let reverse = barostat.log_ratio(&proposal, &gas.pressure(), gas.volume(), -change);
                assert!(
                    (forward + reverse).abs() < 1e-9 * (1.0 + forward.abs()),
                    "{coupling:?}: {forward} and {reverse}"
                );
                largest = largest.max(forward.abs());
                gas = moved;
            }
            assert!(
                largest > 0.1,
                "{coupling:?}: ratios stayed near one ({largest})"
            );
        }
    }

    /// The volume of `atoms` ideal-gas atoms at pressure `P0` follows a Gamma
    /// distribution of shape `atoms + 1` and scale `kT / P0`, whatever the
    /// cell shape does.
    fn check_volume(volume: &[f64], atoms: usize) {
        let (mean, error) = mean_and_error(volume);
        assert!(error < 0.015 * MEAN_VOLUME, "volume error {error}");
        assert!(
            (mean - MEAN_VOLUME).abs() < 4.0 * error,
            "mean volume {mean} +- {error}"
        );
        let squares: Vec<f64> = volume.iter().map(|v| (v - mean).powi(2)).collect();
        let (variance, variance_error) = mean_and_error(&squares);
        let expected = MEAN_VOLUME * MEAN_VOLUME / (atoms as f64 + 1.0);
        assert!(
            variance_error < 0.05 * expected,
            "variance error {variance_error}"
        );
        assert!(
            (variance - expected).abs() < 4.0 * variance_error,
            "volume variance {variance} +- {variance_error} vs {expected}"
        );
    }

    /// The mean of the strain balance of [`sample`] is zero.
    fn check_balance(balance: &[f64], name: &str) {
        let (mean, error) = mean_and_error(balance);
        // The two terms of the balance are each about four.
        assert!(error < 0.08, "{name} balance error {error}");
        assert!(mean.abs() < 4.0 * error, "{name} balance {mean} +- {error}");
    }

    #[test]
    fn isotropic_moves_sample_the_volume_distribution() {
        let ([volume, _, _], accepted) = sample(CellCoupling::Isotropic, 0.0, 5, 150_000, 0.1);
        check_volume(&volume, 5);
        assert!(accepted > 0.85, "accepted {accepted}");
    }

    #[test]
    fn anisotropic_moves_sample_volume_and_axis_ratios() {
        let ([volume, axial, shear], accepted) =
            sample(CellCoupling::Anisotropic, 20.0, 5, 150_000, 0.03);
        check_volume(&volume, 5);
        check_balance(&axial, "axial");
        // No shear: the cell stays orthorhombic, and the shear balance is
        // the constant 2 (M_xx + M_yy), not zero.
        assert!(shear.iter().all(|x| *x > 2.0));
        assert!(accepted > 0.85, "accepted {accepted}");
    }

    #[test]
    fn triclinic_moves_sample_volume_and_shape() {
        let ([volume, axial, shear], accepted) =
            sample(CellCoupling::Triclinic, 20.0, 5, 150_000, 0.03);
        check_volume(&volume, 5);
        check_balance(&axial, "axial");
        check_balance(&shear, "shear");
        assert!(accepted > 0.85, "accepted {accepted}");
    }
}
