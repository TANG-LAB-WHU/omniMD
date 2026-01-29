**************************
Frequently Asked Questions
**************************

Here are some commons questions about OmniMD. If you have more questions, please
contact us on `Gitter`_ to ask it, so that we can add it here!

-  :ref:`faq-simulation-kind`
-  :ref:`faq-why-omnimd`
-  :ref:`faq-why-not-omnimd`
-  :ref:`faq-initial-config`
-  :ref:`faq-parallel`
-  :ref:`faq-why-rust`
-  :ref:`faq-gui`

.. _Gitter: https://gitter.im/omnimd-org/omnimd

.. _faq-simulation-kind:

What kind of simulation can I run with OmniMD?
---------------------------------------------

You should be able to run any kind of classical simulation, the only
limit being the number of atoms fitting in memory.

.. _faq-why-omnimd:

Why should I use OmniMD?
-----------------------

If any of these statement is true for you, you should consider using
OmniMD:

-  You need to use a specific potential that is not yet available in
   other codes, or develop your own potential. Adding a new potential in
   OmniMD is very simple and take less than 20 lines of code;
-  You are developing new simulation algorithms, for example more
   efficient free-energy computations or better parallel scaling of
   Coulomb computations. OmniMD allow you to write the specific
   algorithm, and reuse all the other part of the simulation engine;

Other nice goodies include:

-  Nicely formatted and easy to read input files;
-  *(and more to come ...)*

.. _faq-why-not-omnimd:

Why should I **not** use OmniMD?
-------------------------------

Here are some reasons for you not to use OmniMD:

-  You need to get the fastest code for your simulations because you are
   working with a lot of atoms. OmniMD is relatively young and is not yet
   fully optimized;
-  You need to run your simulation on a cluster. OmniMD can run on
   multiple cores (think OpenMP), but not yet on multiple nodes (think
   MPI).

.. _faq-initial-config:

How can I build the initial configuration?
------------------------------------------

OmniMD does not provide tools for building the initial simulation configuration.
There are already a lot of very good tools around, that you can use. Examples
include `VMD`_, `packmol`_, and many others.  Because OmniMD uses `chemfiles`_ to
read initial configuration, any `format supported by chemfiles
<chemfiles-formats_>`_ can be used.

.. _VMD: http://www.ks.uiuc.edu/Research/vmd/
.. _packmol: http://www.ime.unicamp.br/~martinez/packmol/
.. _chemfiles: http://chemfiles.org/
.. _chemfiles-formats: http://chemfiles.org/chemfiles/latest/formats.html

.. _faq-parallel:

Is the code parallel?
---------------------

OmniMD can run in parallel on a single computer, using the multiple cores
of the processor (this is shared memory parallelism, like OpenMP). It is
not yet possible to run OmniMD on multiple nodes in a cluster (message
passing parallelism, like MPI).

.. _faq-why-rust:

Why is OmniMD written in Rust?
-----------------------------

`Rust <http://www.rust-lang.org/>`__ is a language created by Mozilla,
and was released in 1.0 version in may 2015. It is a modern language,
that provides the same access to the bare metal performances as C or
C++, but prevents some programmer mistakes leading to crashes and
corruptions.

This allow to build better software faster, because the programmer does
not need to spend as much time debugging the code. At the same time, it
also allow to check at compile-time that a code is data-race free, and
allow to build parallel programs more easily.

.. _faq-gui:

Is there any graphical interface to OmniMD?
------------------------------------------

Not yet. But because OmniMD is built as a library implementing all the
simulation algorithms, it should be relatively easy to create a
graphical interface around it. If you are interested in a graphical user
interface (using it or building it), please contact us!
