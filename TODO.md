# TODO - Pending Issues

This document tracks unresolved issues in the `dev` branch for future development.

---

## 🔴 High Priority

### 1. Chemfiles Library SIGFPE Issue

**Problem:**  
The chemfiles library triggers a SIGFPE (floating-point exception) signal during initialization on GitHub Actions CI environment (Ubuntu 22.04), causing tests to crash.

**Affected Areas:**
- All simulation tests in `omnimd-input` (loading `.xyz` files)
- chfl-related tests in `omnimd-core`
- Trajectory tests in `omnimd-sim`
- Integration tests in `tests/docs.rs` (argon, nacl, water)

**Current Workaround:**
- Automatically skip these tests in CI environment
- Related code locations:
  - `omnimd-input/tests/input.rs`: `is_ci()` detection + `uses_chemfiles` flag
  - `tests/docs.rs`: `is_ci()` detection with early return
  - `omnimd-core/src/sys/chfl.rs`: `#[ignore]` attribute
  - `omnimd-sim/src/output/trajectory.rs`: `#[ignore]` attribute

**Potential Solutions:**
- [ ] Investigate upstream chemfiles for fixes
- [ ] Test on different CI environments (macOS, other Linux distros)
- [ ] Consider using Docker containers for testing
- [ ] Report issue to chemfiles maintainers

**Related Links:**
- chemfiles repository: https://github.com/chemfiles/chemfiles

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
