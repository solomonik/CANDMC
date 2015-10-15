include ../../config.mk

INCLUDES := -I../../include/ 
BIN_DIR = ../../bin/tests
LIB_DIR = ../../lib

MM_tests: $(MM_TESTS)
MM_TESTS = test_spc topo_pdgemm_unit
test_spc: $(BIN_DIR)/test_spc 
topo_pdgemm_unit: $(BIN_DIR)/topo_pdgemm_unit


$(BIN_DIR)/topo_pdgemm_unit: topo_pdgemm_unit.cxx $(LIB_DIR)/libNuMM.a $(LIB_DIR)/libNuShared.a
	$(CXX) -o $(BIN_DIR)/topo_pdgemm_unit topo_pdgemm_unit.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuMM -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm
 
$(BIN_DIR)/test_spc: test_spc.cxx $(LIB_DIR)/libNuMM.a $(LIB_DIR)/libNuShared.a
	$(CXX) -o $(BIN_DIR)/test_spc test_spc.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuMM -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm 

clean:
	rm -f *.o 
