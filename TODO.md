# TODO - Pending Issues

This document tracks unresolved issues in the `dev` branch for future development.

---

## ✅ Resolved Issues

### 1. CI Stability and Chemfiles Integration
**Problem:** The chemfiles library triggered a SIGFPE signal in CI. Several tests were ignored, and integration tests used CI-skips to avoid crashes.
**Resolution:**
- Upgraded to **Chemfiles 0.11.0** (via git dependency) to resolve zero-initialization SIGFPE.
- Re-enabled all ignored tests in `omnimd-core` and `omnimd-sim`.
- Fixed broken doctests by exporting and properly scoping items (`Ewald3DArray`, `impl_box_clone!`).
- Created a new **Ubuntu 24.04** CI workflow (`rust-ubuntu-24.yml`) to ensure future compatibility.
- Cleaned up all `is_ci()` workarounds and redundant test arguments.
**Date Resolved:** 2026-01-31

---

## 🟡 Medium Priority

### 2. Cross-Platform Floating-Point Precision

**Problem:**
Different CPU architectures (local vs GitHub Actions) produce slightly different floating-point results (~1e-6 magnitude).

**Current Workaround:**

- Using relative tolerance comparison in `omnimd-sim/src/output/tests.rs`
- Added `approx_eq` helper function in `omnimd-sim/src/output/custom.rs`

**Potential Solutions:**

- [ ] Consider using `approx` crate for unified float comparison
- [ ] Evaluate whether to fix SIMD instruction sets

---

## 🟢 Low Priority

### 3. Test Framework Modernization

**Completed:**

- ✅ Replaced `rustc-test` with `libtest-mimic` (`omnimd-input`)

**TODO:**

- [ ] Evaluate if other tests need migration to `libtest-mimic`
- [ ] Consider adding benchmarks

---

## 📝 Notes

- Last updated: 2026-01-31
- Related CI config: `.github/workflows/rust.yml`
