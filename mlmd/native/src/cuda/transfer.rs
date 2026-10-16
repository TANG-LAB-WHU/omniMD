//! Byte-exact coalescing of training inputs. Kernel arithmetic/source is unchanged.
use omnimd_ml_contract::{Error, Result};
pub(super) const ALIGNMENT_RESERVE: usize = 64; // <= 9 fields, each padded by at most seven bytes.
pub(super) struct Layout {
    pub offsets: [usize; 31],
    pub bytes: usize,
}
impl Layout {
    pub fn new(parts: impl IntoIterator<Item = (usize, usize)>) -> Result<Self> {
        let mut out = Self {
            offsets: [0; 31],
            bytes: 0,
        };
        let mut seen = 0u32;
        for (index, bytes) in parts {
            if index >= 31 || bytes == 0 || seen & (1u32 << index) != 0 {
                return Err(Error("invalid packed CUDA input field".into()));
            }
            seen |= 1u32 << index;
            out.bytes = out
                .bytes
                .checked_add(7)
                .map(|n| n & !7)
                .ok_or_else(|| Error("packed CUDA alignment overflow".into()))?;
            out.offsets[index] = out.bytes;
            out.bytes = out
                .bytes
                .checked_add(bytes)
                .ok_or_else(|| Error("packed CUDA input size overflow".into()))?;
        }
        Ok(out)
    }
}
pub(super) fn input_bytes(n: usize, s: usize, f: usize, gpu_loss: bool) -> Result<usize> {
    let mul = |a: usize, b: usize| {
        a.checked_mul(b)
            .ok_or_else(|| Error("packed CUDA input size overflow".into()))
    };
    let ns = mul(n, s)?;
    let layout = if gpu_loss {
        Layout::new([
            (2, mul(ns, 32)?),
            (3, mul(n, 4)?),
            (17, mul(ns, 4)?),
            (18, mul(ns, 120)?),
            (19, mul(n, 4)?),
            (20, mul(f, 8)?),
            (21, mul(n, 24)?),
            (22, mul(f, 80)?),
            (23, mul(f, 24)?),
        ])
    } else {
        Layout::new([
            (2, mul(ns, 32)?),
            (3, mul(n, 4)?),
            (5, mul(ns, 32)?),
            (16, mul(n, 8)?),
        ])
    };
    layout.map(|l| l.bytes)
}
// Only fully initialized numeric slices are accepted: neither has padding bytes.
pub(super) fn f64_bytes(x: &[f64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), std::mem::size_of_val(x)) }
}
pub(super) fn i32_bytes(x: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), std::mem::size_of_val(x)) }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_layout_alignment_sizes_and_exact_numeric_bits() {
        assert!(Layout::new([(31, 8)]).is_err());
        assert!(Layout::new([(2, 8), (2, 8)]).is_err());
        assert!(Layout::new([(2, usize::MAX), (3, 8)]).is_err());
        assert!(input_bytes(usize::MAX, 64, 32, true).is_err());
        let l = Layout::new([(2, 24), (3, 12), (17, 4), (18, 120)]).unwrap();
        assert_eq!(l.offsets[18] % 8, 0);
        assert_eq!(l.offsets[17], 40);
        assert_eq!(l.bytes, 168);
        assert_eq!(
            i32_bytes(&[-1, 7]),
            [(-1i32).to_ne_bytes(), 7i32.to_ne_bytes()].concat()
        );
        let x = [-0.0, f64::from_bits(0x3ff0000000000001)];
        assert_eq!(&f64_bytes(&x)[..8], &(-0.0f64).to_ne_bytes());
        assert_eq!(&f64_bytes(&x)[8..], &x[1].to_ne_bytes());
        for n in [1, 3, 64, 83, 256, 1024] {
            let raw = n * 16 * 156 + n * 32 + 32 * 112;
            assert!(
                (raw..=raw + ALIGNMENT_RESERVE).contains(&input_bytes(n, 16, 32, true).unwrap())
            );
        }
    }
}
