include ../../config.mk

INCLUDES := -I../../include/ 
BIN_DIR = ../../bin/benchmarks
LIB_DIR = ../../lib

SE_BENCHMARKS = bench_scala_sym_eig bench_full2band bench_elpa_sym_eig bench_full2band_3d
.PHONY: $(SE_BENCHMARKS)
bench_scala_sym_eig SE_benchmarks: $(BIN_DIR)/bench_scala_sym_eig 
bench_elpa_sym_eig SE_benchmarks: $(BIN_DIR)/bench_elpa_sym_eig 
bench_full2band SE_benchmarks: $(BIN_DIR)/bench_full2band 
bench_full2band_3d SE_benchmarks: $(BIN_DIR)/bench_full2band_3d

#$(BIN_DIR)/compare_sytrd

$(BIN_DIR)/bench_elpa_sym_eig: bench_elpa_sym_eig.cxx $(LIB_DIR)/libNuSE.a $(LIB_DIR)/libNuShared.a
ifneq (,$(findstring DUSE_ELPA,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/bench_elpa_sym_eig bench_elpa_sym_eig.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuSE -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

$(BIN_DIR)/bench_scala_sym_eig: bench_scala_sym_eig.cxx $(LIB_DIR)/libNuSE.a $(LIB_DIR)/libNuShared.a
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/bench_scala_sym_eig bench_scala_sym_eig.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuSE -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

$(BIN_DIR)/bench_full2band: bench_full2band.cxx $(LIB_DIR)/libNuSE.a $(LIB_DIR)/libNuShared.a
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/bench_full2band bench_full2band.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuSE -lNuQR -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

$(BIN_DIR)/bench_full2band_3d: bench_full2band_3d.cxx $(LIB_DIR)/libNuSE.a $(LIB_DIR)/libNuShared.a
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/bench_full2band_3d bench_full2band_3d.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuSE -lNuQR -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

#$(BIN_DIR)/compare_sytrd: compare_sytrd.cxx $(LIB_DIR)/libNuSE.a $(LIB_DIR)/libNuShared.a
#	$(CXX) -o $(BIN_DIR)/compare_sytrd compare_sytrd.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
#	-L$(LIB_DIR) -lNuSE -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm  

clean:
	rm -f *.o
