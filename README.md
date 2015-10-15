CANDMC
=====

##**Communication Avoiding Numerical Dense Matrix Computations** 

**Purpose:**

This repository contains studies for algorithms to perform matrix multiplication and dense matrix factorizations, currently: LU, QR, and the symmetric eigensolve.

**Requirements:**

Some version of BLAS and LAPACK required for any build. LAPACK version 3.40 or higher required to build QR codes.

**Build Instructions:**

Running ./configure will generate a config.mk file with build parameters and a Makefile.
See the configure file for build options. Profiling may be activated with flag -DPROFILE (add to DEFS in config.mk).
After running configure, running 'make' will build the library and place it into lib/libCANDMC.a.
Specific contents may be built individually, 
* library of all routines 'CANDMC'
* library of all shared routines required for use of any individual algorithmic library component 'CANShared'
* library for matrix multiplication algorithms 'CANMM' 
* library for LU factorization algorithms 'CANLU' 
* library for QR factorization algorithms 'CANQR' 
* library for symmetric eigensolve algorithms 'CANSE' 
* all unit tests 'test', executables appear in bin/tests/
* all benchmarks 'bench', executables appear in bin/benchmarks/

**Accrediation:**

Code is available under a two-clause BSD license.

Repository created and maintained by Edgar Solomonik (ETH Zurich). Please contact solomonik@inf.ethz.ch with any questions or inquiries.

Thanks to the following developers, snippets of whose code are used in a few places of this repository.
* Grey Ballard (Sandia Laboratory)
* Mathias Jacquelin (Lawrence Berkeley National Laboratory)
* Devin Matthews (University of Texas at Austin)

