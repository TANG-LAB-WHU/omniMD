# TODO - Pending Issues

This document tracks unresolved issues in the `dev` branch for future development.

---

## ✅ Resolved Issues

### 1. Chemfiles Library SIGFPE Issue
**Problem:** The chemfiles library triggered a SIGFPE signal during initialization in CI, likely due to a $0.0/0.0$ division in `UnitCell`.
**Resolution:** Upgraded to **Chemfiles 0.11.0** (via git dependency), which correctly handles zero-initialization and avoids the crash. All CI-specific test skips have been removed.
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
