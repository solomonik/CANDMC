BLAS_LIBS   =  -L/home1/05608/tg849075/critter/lib -lcritter  /opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group /opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64/libmkl_intel_lp64.a /opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64/libmkl_intel_thread.a /opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64/libmkl_core.a /opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64/libmkl_blacs_intelmpi_lp64.a  -Wl,--end-group -lpthread -lm

LDFLAGS     = 
INCLUDES    = -I/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/include

DEFS        =  -DCPP11 -DUSE_SCALAPACK -DLAPACKHASTSQR=1 -DFTN_UNDERSCORE=1

#uncomment below to enable performance profiling
DEFS       += -DPROFILE -DPMPI
#uncomment below to enable debugging
#DEFS       += -DDEBUG=1

AR          = ar

CXX         = mpicxx
CXXFLAGS    = -O3 -g -fopenmp -xMIC-AVX512 -std=c++11 

