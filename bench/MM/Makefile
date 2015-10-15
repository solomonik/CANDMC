include ../../config.mk

INCLUDES := -I../../include/ 
BIN_DIR = ../../bin/benchmarks
LIB_DIR = ../../lib

MM_benchmarks: $(MM_BENCHMARKS)
MM_BENCHMARKS = bench_spc topo_pdgemm_bench

bench_spc: $(BIN_DIR)/bench_spc 
topo_pdgemm_bench: $(BIN_DIR)/topo_pdgemm_bench

$(BIN_DIR)/topo_pdgemm_bench: topo_pdgemm_bench.cxx $(LIB_DIR)/libNuMM.a $(LIB_DIR)/libNuShared.a
	$(CXX) -o $(BIN_DIR)/topo_pdgemm_bench topo_pdgemm_bench.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuMM -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm -DNO_PVT

$(BIN_DIR)/bench_spc: bench_spc.cxx $(LIB_DIR)/libNuMM.a $(LIB_DIR)/libNuShared.a
	$(CXX) -o $(BIN_DIR)/bench_spc bench_spc.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuMM -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm -DNO_PVT

clean:
	rm -f *.o 
