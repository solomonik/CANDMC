include ../../config.mk

LU_BENCHMARKS = lu_25d_np_bench lu_25d_pp_bench lu_25d_tp_bench

INCLUDES := -I../../include/ 
BIN_DIR = ../../bin/benchmarks
LIB_DIR = ../../lib

lu_25d_pp_bench: $(BIN_DIR)/lu_25d_pp_bench 
lu_25d_tp_bench: $(BIN_DIR)/lu_25d_tp_bench 
lu_25d_np_bench: $(BIN_DIR)/lu_25d_np_bench

LU_benchmarks: $(LU_BENCHMARKS)

$(BIN_DIR)/lu_25d_np_bench: lu_25d_pvt_bench.cxx 
	$(CXX) -o $(BIN_DIR)/lu_25d_np_bench lu_25d_pvt_bench.cxx $(CXXFLAGS)  $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuLU -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm -DNO_PVT

$(BIN_DIR)/lu_25d_pp_bench: lu_25d_pvt_bench.cxx
	$(CXX) -o $(BIN_DIR)/lu_25d_pp_bench lu_25d_pvt_bench.cxx $(CXXFLAGS)  $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuLU -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm -DPARTIAL_PVT

$(BIN_DIR)/lu_25d_tp_bench: lu_25d_pvt_bench.cxx 	
	$(CXX) -o $(BIN_DIR)/lu_25d_tp_bench lu_25d_pvt_bench.cxx $(CXXFLAGS)  $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuLU -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm -DTNMT_PVT

$(BIN_DIR)/par_tnmt_bench: par_tnmt_bench.cxx 
	$(CXX) -o par_tnmt_bench par_tnmt_bench.cxx $(CXXFLAGS)  $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuLU -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm

clean:
	rm -f *.o
