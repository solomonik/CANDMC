BLAS_LIBS   =  -mkl=cluster  -mkl=cluster

LDFLAGS     = 
INCLUDES    = 

DEFS        =  -DCPP11 -DUSE_SCALAPACK -DLAPACKHASTSQR=1 -DFTN_UNDERSCORE=1

#uncomment below to enable performance profiling
DEFS       += -DPROFILE -DPMPI
#uncomment below to enable debugging
#DEFS       += -DDEBUG=1

AR          = ar

CXX         = CC
CXXFLAGS    = -fast -std=c++11 

