include ../../config.mk

INCLUDES := -I../../include/ 
BIN_DIR = ../../bin/benchmarks
LIB_DIR = ../../lib

QR_BENCHMARKS = bench_hh_recon bench_qr_2d bench_qr_butterfly_2d bench_qr_tree_2d \
                bench_scala_qr  bench_qr_seq bench_qr_2d_hh_scala bench_qr_y2d

QR_benchmarks: $(QR_BENCHMARKS)

bench_hh_recon: $(BIN_DIR)/bench_hh_recon 
bench_qr_seq: $(BIN_DIR)/bench_qr_seq 
bench_qr_2d: $(BIN_DIR)/bench_qr_2d 
bench_qr_y2d: $(BIN_DIR)/bench_qr_y2d 
bench_qr_tree_2d: $(BIN_DIR)/bench_qr_tree_2d 
bench_scala_qr: $(BIN_DIR)/bench_scala_qr
bench_qr_2d_hh_scala: $(BIN_DIR)/bench_qr_2d_hh_scala
bench_qr_butterfly_2d: $(BIN_DIR)/bench_qr_butterfly_2d 

#QR_benchmarks: $(BIN_DIR)/bench_hh_recon \
				  $(BIN_DIR)/bench_qr_2d $(BIN_DIR)/bench_qr_tree_2d $(BIN_DIR)/bench_scala_qr_2d $(BIN_DIR)/bench_qr_butterfly_2d 

$(BIN_DIR)/bench_hh_recon: bench_hh_recon.cxx $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANShared.a
ifneq (,$(findstring DLAPACKHASTSQR=1,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/bench_hh_recon bench_hh_recon.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANQR -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

$(BIN_DIR)/bench_qr_2d: bench_qr_2d.cxx $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANShared.a
ifneq (,$(findstring DLAPACKHASTSQR=1,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/bench_qr_2d bench_qr_2d.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANQR -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

$(BIN_DIR)/bench_qr_y2d: bench_qr_y2d.cxx $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANShared.a
ifneq (,$(findstring DLAPACKHASTSQR=1,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/bench_qr_y2d bench_qr_y2d.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANQR -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

$(BIN_DIR)/bench_qr_seq: bench_qr_seq.cxx $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANShared.a
ifneq (,$(findstring DLAPACKHASTSQR=1,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/bench_qr_seq bench_qr_seq.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANQR -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

$(BIN_DIR)/bench_scala_qr: bench_scala_qr.cxx $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANShared.a
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/bench_scala_qr bench_scala_qr.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANQR -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

$(BIN_DIR)/bench_qr_2d_hh_scala: bench_qr_2d_hh_scala.cxx $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANShared.a
ifneq (,$(findstring DLAPACKHASTSQR=1,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/bench_qr_2d_hh_scala bench_qr_2d_hh_scala.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANQR -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

$(BIN_DIR)/bench_qr_tree_2d: bench_qr_tree_2d.cxx $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANShared.a
ifneq (,$(findstring DLAPACKHASTSQR=1,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/bench_qr_tree_2d bench_qr_tree_2d.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANQR -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

$(BIN_DIR)/bench_qr_butterfly_2d: bench_qr_butterfly_2d.cxx $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANShared.a
ifneq (,$(findstring DLAPACKHASTSQR=1,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/bench_qr_butterfly_2d bench_qr_butterfly_2d.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANQR -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

$(BIN_DIR)/bench_qr_seq: bench_qr_seq.cxx $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANShared.a
ifneq (,$(findstring DLAPACKHASTSQR=1,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/bench_qr_seq bench_qr_seq.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANQR -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif


clean:
	rm -f *.o
