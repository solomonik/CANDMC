include ../../config.mk

INCLUDES := -I../../include/ 
BIN_DIR = ../../bin/tests
LIB_DIR = ../../lib

SE_TESTS = test_scala_sym_eig test_full2band
.PHONY: $(SE_TESTS)
SE_tests: $(SE_TESTS)
test_scala_sym_eig: $(BIN_DIR)/test_scala_sym_eig 
test_full2band: $(BIN_DIR)/test_full2band 
test_full2band_3d: $(BIN_DIR)/test_full2band_3d 
test_full2band_scala: $(BIN_DIR)/test_full2band_scala 


$(BIN_DIR)/test_scala_sym_eig: test_scala_sym_eig.cxx $(LIB_DIR)/libCANSE.a $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANShared.a
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/test_scala_sym_eig test_scala_sym_eig.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANSE -lCANQR -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

$(BIN_DIR)/test_full2band_3d: test_full2band_3d.cxx $(LIB_DIR)/libCANSE.a $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANShared.a
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/test_full2band_3d test_full2band_3d.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANSE -lCANQR -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif


$(BIN_DIR)/test_full2band: test_full2band.cxx $(LIB_DIR)/libCANSE.a $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANShared.a
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/test_full2band test_full2band.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANSE -lCANQR -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

$(BIN_DIR)/test_full2band_scala: test_full2band_scala.cxx $(LIB_DIR)/libCANSE.a $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANShared.a
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(CXX) -o $(BIN_DIR)/test_full2band_scala test_full2band_scala.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANSE -lCANQR -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm  
endif

clean:
	rm -f *.o
