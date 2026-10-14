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
