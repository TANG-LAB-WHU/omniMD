//! Small dense least squares: Householder QR with column pivoting, no normal equations.
use omnimd_ml_contract::{Error, Result};

pub(crate) struct Solution {
    pub x: Vec<f64>,
    pub rank: usize,
}

pub(crate) fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>, rtol: f64) -> Result<Solution> {
    let m = a.len();
    let n = a.first().map_or(0, Vec::len);
    if m == 0
        || n == 0
        || b.len() != m
        || !(0.0 < rtol && rtol < 1.0)
        || a.iter()
            .any(|r| r.len() != n || r.iter().any(|x| !x.is_finite()))
        || b.iter().any(|x| !x.is_finite())
    {
        return Err(Error(
            "invalid least-squares dimensions, data, or rank tolerance".into(),
        ));
    }
    let mut permutation: Vec<usize> = (0..n).collect();
    let mut rank = 0;
    let mut leading_norm = 0.0;
    for k in 0..m.min(n) {
        let mut pivot = k;
        let mut largest = 0.0;
        for j in k..n {
            let norm = (k..m).fold(0.0_f64, |v, i| v.hypot(a[i][j]));
            if norm > largest {
                largest = norm;
                pivot = j;
            }
        }
        if !largest.is_finite() {
            return Err(Error("least-squares norm overflow".into()));
        }
        if k == 0 {
            leading_norm = largest;
        }
        if largest == 0.0 || largest <= rtol * leading_norm {
            break;
        }
        for row in &mut a {
            row.swap(k, pivot);
        }
        permutation.swap(k, pivot);
        let sign = if a[k][k] >= 0.0 { 1.0 } else { -1.0 };
        // Scale before constructing the reflector to avoid x0 + norm overflow.
        let mut v: Vec<f64> = (k..m).map(|i| a[i][k] / largest).collect();
        v[0] += sign;
        let tau = 2.0 / v.iter().map(|x| x * x).sum::<f64>();
        for j in k..n {
            let dot = tau
                * v.iter()
                    .enumerate()
                    .map(|(i, x)| x * a[k + i][j])
                    .sum::<f64>();
            for (i, x) in v.iter().enumerate() {
                a[k + i][j] -= x * dot;
            }
        }
        let dot = tau * v.iter().enumerate().map(|(i, x)| x * b[k + i]).sum::<f64>();
        for (i, x) in v.iter().enumerate() {
            b[k + i] -= x * dot;
        }
        a[k][k] = -sign * largest;
        for row in a.iter_mut().skip(k + 1) {
            row[k] = 0.0;
        }
        rank += 1;
    }
    let mut pivoted = vec![0.0; n];
    for i in (0..rank).rev() {
        pivoted[i] = (b[i] - (i + 1..rank).map(|j| a[i][j] * pivoted[j]).sum::<f64>()) / a[i][i];
    }
    if pivoted.iter().any(|x| !x.is_finite()) || b.iter().any(|x| !x.is_finite()) {
        return Err(Error("nonfinite least-squares solution".into()));
    }
    let mut x = vec![0.0; n];
    for i in 0..n {
        x[permutation[i]] = pivoted[i];
    }
    // Rank-deficient result is a pivoted BASIC solution, not minimum norm.
    Ok(Solution { x, rank })
}

/// Minimum-norm least squares `min |a·x - b|`, like `numpy.linalg.lstsq`:
/// singular values at or below `rcond` times the largest count as zero
/// (`None` uses machine epsilon times `max(rows, columns)`). Returns the
/// solution and the rank.
///
/// One-sided Jacobi SVD on the columns of `a`, so small singular values keep
/// their relative accuracy (no normal equations).
pub(crate) fn min_norm(a: &[Vec<f64>], b: &[f64], rcond: Option<f64>) -> Result<Solution> {
    let m = a.len();
    let n = a.first().map_or(0, Vec::len);
    if m == 0
        || n == 0
        || b.len() != m
        || a.iter()
            .any(|r| r.len() != n || r.iter().any(|x| !x.is_finite()))
        || b.iter().any(|x| !x.is_finite())
        || rcond.is_some_and(|r| !(0.0..1.0).contains(&r))
    {
        return Err(Error(
            "invalid least-squares dimensions, data, or rank tolerance".into(),
        ));
    }
    // Column-major copies: u starts as a and ends as U·Σ, v accumulates V.
    let mut u: Vec<Vec<f64>> = (0..n)
        .map(|j| a.iter().map(|row| row[j]).collect())
        .collect();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..n).map(|i| f64::from(u8::from(i == j))).collect())
        .collect();
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>();
    for _sweep in 0..60 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let (alpha, beta, gamma) =
                    (dot(&u[p], &u[p]), dot(&u[q], &u[q]), dot(&u[p], &u[q]));
                if gamma == 0.0 || gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for columns in [&mut u, &mut v] {
                    let (left, right) = columns.split_at_mut(q);
                    for (x, y) in left[p].iter_mut().zip(right[0].iter_mut()) {
                        (*x, *y) = (c * *x - s * *y, s * *x + c * *y);
                    }
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let sigma: Vec<f64> = u.iter().map(|column| dot(column, column).sqrt()).collect();
    let largest = sigma.iter().fold(0.0_f64, |s, &x| s.max(x));
    let cutoff = rcond.unwrap_or(f64::EPSILON * m.max(n) as f64) * largest;
    let mut x = vec![0.0; n];
    let mut rank = 0;
    for j in 0..n {
        if sigma[j] > cutoff {
            rank += 1;
            // x += v_j (u_j·b) / sigma_j, with u[j] = u_j·sigma_j.
            let coefficient = dot(&u[j], b) / (sigma[j] * sigma[j]);
            for (xi, vi) in x.iter_mut().zip(&v[j]) {
                *xi += coefficient * vi;
            }
        }
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(Error("nonfinite least-squares solution".into()));
    }
    Ok(Solution { x, rank })
}

#[cfg(test)]
mod tests {
    use super::min_norm;

    #[test]
    fn min_norm_matches_closed_forms() {
        // Full rank: exact solution.
        let a = vec![vec![1.0, 2.0], vec![3.0, 1.0], vec![2.0, 2.0]];
        let b: Vec<f64> = a.iter().map(|r| -3.0 * r[0] + 0.5 * r[1]).collect();
        let s = min_norm(&a, &b, None).unwrap();
        assert_eq!(s.rank, 2);
        assert!((s.x[0] + 3.0).abs() < 1e-13 && (s.x[1] - 0.5).abs() < 1e-13);
        // Water-like counts (H = 2·O in every frame): rank 1, and the minimum
        // norm solution is along (1, 2): e = x_O + 2 x_H with x = e·(1, 2)/5.
        let a = vec![vec![64.0, 128.0], vec![32.0, 64.0], vec![1.0, 2.0]];
        let b = vec![-640.0, -320.0, -10.0];
        let s = min_norm(&a, &b, None).unwrap();
        assert_eq!(s.rank, 1);
        assert!((s.x[0] + 2.0).abs() < 1e-12 && (s.x[1] + 4.0).abs() < 1e-12);
        // A type that never appears gets zero.
        let a = vec![vec![2.0, 0.0], vec![4.0, 0.0]];
        let s = min_norm(&a, &[6.0, 12.0], None).unwrap();
        assert_eq!((s.rank, s.x[1]), (1, 0.0));
        assert!((s.x[0] - 3.0).abs() < 1e-13);
        assert!(min_norm(&a, &[1.0], None).is_err());
    }
}
