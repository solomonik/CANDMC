BLAS_LIBS   =    -L/home1/05608/tg849075/RefScalapack/build/lib -lscalapack -lm -lifcore

LDFLAGS     = 
INCLUDES    = 

DEFS        =  -DCPP11 -DUSE_SCALAPACK -DLAPACKHASTSQR=1 -DFTN_UNDERSCORE=1

#uncomment below to enable performance profiling
DEFS       += -DPROFILE -DPMPI
#uncomment below to enable debugging
#DEFS       += -DDEBUG=1

AR          = ar

CXX         = mpicxx
CXXFLAGS    = -O3 -g -mkl=parallel -std=c++11 -xMIC-AVX512 

