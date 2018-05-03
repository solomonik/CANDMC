BLAS_LIBS   =  -L/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64 -lmkl_scalapack_ilp64  -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_ilp64 -liomp5 -lpthread -lm -ldl

LDFLAGS     = 
INCLUDES    = -I/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/include

DEFS        = -DMKL_ILP64 -DCPP11 -DUSE_SCALAPACK -DLAPACKHASTSQR=1 -DFTN_UNDERSCORE=1

#uncomment below to enable performance profiling
DEFS       += -DPROFILE -DPMPI
#uncomment below to enable debugging
#DEFS       += -DDEBUG=1

AR          = ar

CXX         = mpicxx
CXXFLAGS    = -O3 -xMIC-AVX512 -std=c++11 

