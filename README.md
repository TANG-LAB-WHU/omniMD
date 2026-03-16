# omniMD molecular simulation engine

[![Build Status](https://travis-ci.org/TANG-LAB-WHU/omniMD.svg?branch=master)](https://travis-ci.org/TANG-LAB-WHU/omniMD)
[![Coverage](https://codecov.io/gh/TANG-LAB-WHU/omniMD/branch/master/graph/badge.svg)](https://codecov.io/gh/TANG-LAB-WHU/omniMD)
[![Documentation](https://img.shields.io/badge/documentation-latest-brightgreen.svg)](https://TANG-LAB-WHU.github.io/omniMD/latest/index.html)
[![Gitter](https://badges.gitter.im/TANG-LAB-WHU/omniMD.svg)](https://gitter.im/TANG-LAB-WHU/omniMD)

**omniMD: Omnipotent Molecular Dynamics** is a versatile simulation engine designed to bridge the dimensional gap in unbiased molecular interactions by leveraging machine learning-based interatomic potentials.

omniMD provides a solid foundation for developing new simulation algorithms and methods. It allows users to fully customize the behavior of algorithms during a simulation. Whether you are adding a new force field, customizing Monte Carlo moves, or implementing new molecular dynamics integrators, the process is straightforward and well-documented.

Our primary goals for omniMD are flexibility, reliability, and extensibility. Specifically, this means the software is:

- **Flexible**: Capable of simulating a wide variety of systems—from proteins to crystals—using diverse methods such as molecular dynamics and Monte Carlo algorithms.
- **Reliable**: Rigorously tested at both the function level and the simulation level to ensure the accuracy of the systems' thermodynamic properties.
- **Extensible**: Designed with a modular, object-oriented, and open-source architecture that is highly readable and easy to build upon.

omniMD is under active development and should currently be considered alpha software. If you are interested in the project, have questions, or would like to contribute, please feel free to open a [GitHub issue][issues].

## Features

- Pair, molecular and electrostatic interactions (with Ewald or Wolf methods);
- Energy minimization;
- Molecular dynamics simulations in the NVE, NVT and NPT ensembles;
- Monte Carlo simulations in the NVT ensemble;
- and many others! Have a look at the [documentation](#documentation) for more
  information

## Getting started

omniMD provides both a command line tool for running simulations; and a Rust
library for writing your own simulations algorithms using the pre-existing
building blocks.

### Documentation

Documentation is hosted on [Github Pages](http://TANG-LAB-WHU.github.io/omniMD), and separated
in multiple parts:

- The [user manual][user_manual] contains information about the general
  concepts of systems and simulations used in omniMD. Additionally, it has
  tutorials on how to use and extend omniMD. Use this documentation if you want
  to know basic concepts and how they are used in omniMD.
- The [input reference][input_reference] contains information about - well,
  the input file system of omniMD.
  Use this document if you want to use omniMD as a command line tool
  without writing code.
- To use omniMD as a library inside your own code, we have a [developer
  documentation][devdoc], which contains documentation for all the library
  public functions, and examples for most of them.

### Prerequisites

omniMD requires the following dependencies:

- **Rust**: Latest stable version recommended.
- **LibTorch**: The C++ frontend for PyTorch (required for ML potentials).

#### Setting up LibTorch

omniMD uses [tch-rs](https://github.com/LaurentMazare/tch-rs) bindings. You must have LibTorch available.

To utilize your GPU (e.g., RTX 5080) with CUDA 12.9 support, we recommend using the following build.

1. **Install PyTorch (Python) with CUDA 12.9:**
   Run the following command as recommended:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
   ```

2. **Download matching LibTorch C++ Library:**
   You must download the **LibTorch C++ library** that corresponds to the installed PyTorch version.
   - Go to [pytorch.org](https://pytorch.org/get-started/locally/).
   - Ensure the version matches the PyTorch version installed above (check `python -c "import torch; print(torch.__version__)"`).
   - Please download the **cxx11 ABI** version if you are on Linux and rely on gcc > 5.
   - **Important:** Since we are using cutting-edge features/versions, `omnimd-core` points to the latest `tch-rs` git repository to ensure compatibility.

3. Extract the downloaded `libtorch` archive.
4. Set the `LIBTORCH` environment variable to the extracted path.
5. Add the `lib` subdirectory to your system `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux/macOS).

### Installation

You can install omniMD directly from the source code:

```bash
# 1. Clone the repository
git clone https://github.com/TANG-LAB-WHU/omniMD.git
cd omniMD

# 2. Set Environment Variables (Example for PowerShell)
# Assume LibTorch is extracted to C:\libtorch
$env:LIBTORCH = "C:\libtorch"
$env:PATH = "$env:LIBTORCH\lib;$env:PATH"

# 2. Set Environment Variables (Example for Bash)
# export LIBTORCH=/path/to/libtorch
# export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

# 3. Install via Cargo
cargo install --path .
```

This will compile the project and install the `omnimd` binary to your Cargo bin directory (usually `~/.cargo/bin`).

### Usage as a library

You can add omniMD as a dependency in your project's `Cargo.toml`:

```toml
[dependencies]
omnimd = {path = "/path/to/omnimd"}
```

A tutorial about how to implement new algorithms in omniMD is coming.

## Contributing

If you want to contribute to omniMD, there are several ways to go: improving the
documentation and helping with language issues; testing the code on your systems
to find bugs; adding new algorithms and potentials; providing feature requests.
providing feature requests. Please open an [issue][issues] to discuss improvements.

## License

This software is licensed under the MIT license, see the LICENSE file for legal
text.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, shall be licensed under the same MIT license,
without any additional terms or conditions.


[issues]: https://github.com/TANG-LAB-WHU/omniMD/issues/new

[user_manual]: http://TANG-LAB-WHU.github.io/omniMD/latest/book/
[input_reference]: http://TANG-LAB-WHU.github.io/omniMD/latest/book/
[devdoc]: http://TANG-LAB-WHU.github.io/omniMD/latest/omniMD/
