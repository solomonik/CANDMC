include ../../config.mk

LU_BENCHMARKS = lu_25d_np_bench lu_25d_pp_bench lu_25d_tp_bench bench_scala_lu

INCLUDES := -I../../include/
BIN_DIR = ../../bin/benchmarks
LIB_DIR = ../../lib

lu_25d_pp_bench: $(BIN_DIR)/lu_25d_pp_bench 
lu_25d_tp_bench: $(BIN_DIR)/lu_25d_tp_bench 
lu_25d_np_bench: $(BIN_DIR)/lu_25d_np_bench
bench_scala_lu: $(BIN_DIR)/bench_scala_lu
bench_scala_cholesky: $(BIN_DIR)/bench_scala_cholesky

LU_benchmarks: $(LU_BENCHMARKS)

$(BIN_DIR)/lu_25d_np_bench: lu_25d_pvt_bench.cxx 
	$(CXX) -o $(BIN_DIR)/lu_25d_np_bench lu_25d_pvt_bench.cxx $(CXXFLAGS)  $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANLU -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm -DNO_PVT

$(BIN_DIR)/lu_25d_pp_bench: lu_25d_pvt_bench.cxx
	$(CXX) -o $(BIN_DIR)/lu_25d_pp_bench lu_25d_pvt_bench.cxx $(CXXFLAGS)  $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANLU -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm -DPARTIAL_PVT

$(BIN_DIR)/lu_25d_tp_bench: lu_25d_pvt_bench.cxx 	
	$(CXX) -o $(BIN_DIR)/lu_25d_tp_bench lu_25d_pvt_bench.cxx $(CXXFLAGS)  $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANLU -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm -DTNMT_PVT

$(BIN_DIR)/par_tnmt_bench: par_tnmt_bench.cxx 
	$(CXX) -o par_tnmt_bench par_tnmt_bench.cxx $(CXXFLAGS)  $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANLU -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm

$(BIN_DIR)/bench_scala_lu: bench_scala_lu.cxx $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANShared.a
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/bench_scala_lu bench_scala_lu.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANQR -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

$(BIN_DIR)/bench_scala_cholesky: bench_scala_cholesky.cxx $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANShared.a
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/bench_scala_cholesky bench_scala_cholesky.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANQR -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

clean:
	rm -f *.o
