include ../../config.mk

INCLUDES := -I../../include/ 
BIN_DIR = ../../bin/tests
LIB_DIR = ../../lib

LU_TESTS = lu_25d_np_test lu_25d_pp_test lu_25d_tp_test  
LU_tests: $(LU_TESTS)
lu_25d_pp_test: $(BIN_DIR)/lu_25d_pp_test
lu_25d_tp_test: $(BIN_DIR)/lu_25d_tp_test
lu_25d_np_test: $(BIN_DIR)/lu_25d_np_test 


$(BIN_DIR)/lu_25d_np_test: unit_test.o lu_25d_pvt_unit_test.cxx $(LIB_DIR)/libCANLU.a $(LIB_DIR)/libCANShared.a
	$(CXX) unit_test.o -o $(BIN_DIR)/lu_25d_np_test lu_25d_pvt_unit_test.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANLU -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm -DNO_PVT

$(BIN_DIR)/lu_25d_pp_test: unit_test.o lu_25d_pvt_unit_test.cxx $(LIB_DIR)/libCANLU.a $(LIB_DIR)/libCANShared.a
	$(CXX) unit_test.o -o $(BIN_DIR)/lu_25d_pp_test lu_25d_pvt_unit_test.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANLU -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm -DPARTIAL_PVT

$(BIN_DIR)/lu_25d_tp_test: unit_test.o lu_25d_pvt_unit_test.cxx $(LIB_DIR)/libCANLU.a $(LIB_DIR)/libCANShared.a
	$(CXX) unit_test.o -o $(BIN_DIR)/lu_25d_tp_test lu_25d_pvt_unit_test.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANLU -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm -DTNMT_PVT

unit_test.o: unit_test.cxx unit_test.h
	$(CXX) -c unit_test.cxx -o unit_test.o $(CXXFLAGS)   $(DEFS) $(INCLUDES) 

unit_test: unit_test.cxx seq_tnmt_unit_test.o par_tnmt_unit_test.o \
	   lu_25d_unit_test.o  unit_test.h par_pivot_unit_test.o lu_25d_pvt_unit_test.o  
	$(CXX) lu_25d_unit_test.o par_pivot_unit_test.o lu_25d_pvt_unit_test.o -o $(BIN_DIR)/unit_test unit_test.cxx $(CXXFLAGS)   $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANLU -lCANShared $(BLAS_LIBS) $(LDFLAGS)    -lm

$(BIN_DIR)/lu_25d_unit_test: lu_25d_unit_test.cxx 
	$(CXX) -o $(BIN_DIR)/lu_25d_unit_test lu_25d_unit_test.cxx $(CXXFLAGS)  $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lCANLU -lCANShared $(BLAS_LIBS) $(LDFLAGS) -lm

par_pivot_unit_test.o: par_pivot_unit_test.cxx unit_test.h
	$(CXX) -c par_pivot_unit_test.cxx -o par_pivot_unit_test.o $(CXXFLAGS)  $(DEFS) $(INCLUDES)

par_tnmt_unit_test.o: par_tnmt_unit_test.cxx unit_test.h
	$(CXX) -c par_tnmt_unit_test.cxx -o par_tnmt_unit_test.o $(CXXFLAGS)  $(DEFS) $(INCLUDES)

clean:
	rm -f *.o
