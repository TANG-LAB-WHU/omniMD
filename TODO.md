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
- All integration tests in `tests/*.rs` (docs, mc-*, md-*, nist-*) when using chemfiles

**Current Workaround:**

- Automatically skip these tests in CI environment
- Related code locations:
  - `omnimd-input/tests/input.rs`: `is_ci()` detection + `uses_chemfiles` flag
  - `tests/*.rs`: Added `is_ci()` check with early return in all test functions
  - `tests/utils/mod.rs`: Added `is_ci()` helper for tests using utils
  - `omnimd-core/src/sys/chfl.rs`: `#[ignore]` attribute
  - `omnimd-sim/src/output/trajectory.rs`: `#[ignore]` attribute

**Research Findings (2026-01-31):**

- **Root Cause:** SIGFPE (Signal Floating-Point Exception) usually indicates an invalid arithmetic operation (e.g., division by zero, overflow) in the underlying C++ library.
- **Context:** No widespread "known bug" for `chemfiles` on Ubuntu 22.04 matching this exactly, suggesting it may be specific to:
  - Certain input files used in tests (e.g., malformed `.xyz` or invalid box dimensions).
  - Subtle environment differences (math library linking) on GitHub Actions runners.
- **Version:** Current dependency `0.10` likely resolves to `0.10.x`. Latest available is `0.10.41`.

**Potential Solutions:**

- [ ] **Data Validation:** Audit test data files (`.xyz`, `.toml`) for values that could cause division by zero (e.g., zero box lengths, overlapping atoms).
- [ ] **Dependency Update:** Explicitly try updating to the absolute latest `chemfiles` patch version (`cargo update -p chemfiles`).
- [ ] **Debugging:** If possible, enable core dumps in a custom CI workflow or add specific logging around `chemfiles` calls in Rust to identify exactly *which* file or call triggers the crash.
- [ ] **Upstream:** Report the specific reproduction case (with data) to chemfiles maintainers if isolated.

**Related Links:**

- chemfiles repository: <https://github.com/chemfiles/chemfiles>

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
