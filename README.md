# omniMD molecular simulation engine

[![Build Status](https://travis-ci.org/TANG-LAB-WHU/omniMD.svg?branch=master)](https://travis-ci.org/TANG-LAB-WHU/omniMD)
[![Coverage](https://codecov.io/gh/TANG-LAB-WHU/omniMD/branch/master/graph/badge.svg)](https://codecov.io/gh/TANG-LAB-WHU/omniMD)
[![Documentation](https://img.shields.io/badge/documentation-latest-brightgreen.svg)](https://TANG-LAB-WHU.github.io/omniMD/latest/index.html)
[![Gitter](https://badges.gitter.im/TANG-LAB-WHU/omniMD.svg)](https://gitter.im/TANG-LAB-WHU/omniMD)

omniMD: Omnipotent Molecular Dynamics bridging dimensional dilemma over unbiased molecular interatomic potentials. Machine learning based interatomic potentials are adopted.

omniMD provides a solid base for developing new algorithms and methods. Using omniMD, you can customize the behavior of all the algorithms in a simulation. Adding a new force field, customizing Monte Carlo moves or molecular dynamics integrators is easy and well documented.

omniMD goals are to be flexible, reliable and extensible. For us, this means that
this software should be:

- **flexible**: the code can simulate all kind of systems, from proteins to
  crystals, using various methods: molecular dynamics, Monte Carlo, *etc.*
- **reliable**: the code is well tested, both at the function level; and at the
  simulation level, checking thermodynamic properties of the systems;
- **extendable**: the code is modular, object-oriented, well documented,
  open-source, and easy to read.

omniMD is actively developed, and should be considered as alpha software. If
you are interested, have some questions or want to participate, you can open a
[Github issue][issues].

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

### Installation as a command line tool

You will need a stable Rust compiler, [grab one][Rust] if you do not have one
yet. Then, you can download the code, build it and install it by running:

```bash
cargo install --path .
```

This will produce the a `omnimd` binary in `~/.cargo/bin`.

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
Please come by and [talk with us][Gitter] a bit before staring new work, or open
an [issue][issues] to discuss improvements. We also have

## License

This software is licensed under the MIT license, see the LICENSE file for legal
text.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, shall be licensed under the same MIT license,
without any additional terms or conditions.

[Rust]: https://www.rust-lang.org/downloads.html
[Gitter]: https://gitter.im/TANG-LAB-WHU/omniMD
[issues]: https://github.com/TANG-LAB-WHU/omniMD/issues/new

[user_manual]: http://TANG-LAB-WHU.github.io/omniMD/latest/book/
[input_reference]: http://TANG-LAB-WHU.github.io/omniMD/latest/book/
[devdoc]: http://TANG-LAB-WHU.github.io/omniMD/latest/omniMD/
