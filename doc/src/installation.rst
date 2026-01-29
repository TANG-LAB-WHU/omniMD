************
Installation
************

OmniMD is written in `rust`_ (:ref:`why? <faq-why-rust>`), and you will need a
Rust compiler to compile it. You can find rust installation instructions `here
<rust-install_>`_, or use your system package manager. OmniMD also depends on
some C++ libraries, so you will need a C++ compiler and CMake to be installed.

OmniMD is tested on Linux and OS X, and should build on Windows without any
issue. You will need a C++11 capable compiler on Windows (MSVC > 15 or Mingw
with gcc > 4.9). Be sure to pick the corresponding version of the Rust compiler.

When all the dependencies are installed on you system, you can install the
the latest development version with:

.. code-block:: bash

    cargo install --git https://github.com/omnimd-org/omnimd omnimd

This command will download and install omnimd in ``~/.cargo/bin/omnimd``, where
``~`` is your home directory. You may want to add ``~/.cargo/bin`` to your PATH
or move the ``omnimd`` binary in another directory accessible in your PATH.

You can check that the installation worked by running

.. code-block:: bash

    omnimd --version

.. _rust: https://www.rust-lang.org/
.. _rust-install: https://www.rust-lang.org/downloads.html
