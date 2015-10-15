include ../../config.mk

INCLUDES := -I../../include/ 
BIN_DIR := ../../bin/tests
LIB_DIR := ../../lib

QR_TESTS := test_bitree_tsqr test_construct_tsqr_Q test_hh_recon \
            test_qr_2d test_qr_y2d test_qr_butterfly_2d test_qr_tree_2d test_scala_qr_2d 

QR_tests: $(QR_TESTS)
test_hh_recon: $(BIN_DIR)/test_hh_recon 
test_construct_tsqr_Q: $(BIN_DIR)/test_construct_tsqr_Q 
test_bitree_tsqr: $(BIN_DIR)/test_bitree_tsqr 
test_qr_2d: $(BIN_DIR)/test_qr_2d 
test_qr_y2d: $(BIN_DIR)/test_qr_y2d 
test_qr_tree_2d: $(BIN_DIR)/test_qr_tree_2d 
test_scala_qr_2d: $(BIN_DIR)/test_scala_qr_2d 
test_qr_butterfly_2d: $(BIN_DIR)/test_qr_butterfly_2d 

$(BIN_DIR)/test_hh_recon: test_hh_recon.cxx $(LIB_DIR)/libNuQR.a $(LIB_DIR)/libNuShared.a
	$(CXX) -o $(BIN_DIR)/test_hh_recon test_hh_recon.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuQR -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm  

$(BIN_DIR)/test_construct_tsqr_Q: test_construct_tsqr_Q.cxx $(LIB_DIR)/libNuQR.a $(LIB_DIR)/libNuShared.a
	$(CXX) -o $(BIN_DIR)/test_construct_tsqr_Q test_construct_tsqr_Q.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuQR -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm  

$(BIN_DIR)/test_bitree_tsqr: test_bitree_tsqr.cxx $(LIB_DIR)/libNuQR.a $(LIB_DIR)/libNuShared.a
	$(CXX) -o $(BIN_DIR)/test_bitree_tsqr test_bitree_tsqr.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuQR -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm  

$(BIN_DIR)/test_qr_2d: test_qr_2d.cxx $(LIB_DIR)/libNuQR.a $(LIB_DIR)/libNuShared.a
	$(CXX) -o $(BIN_DIR)/test_qr_2d test_qr_2d.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuQR -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm  

$(BIN_DIR)/test_qr_y2d: test_qr_y2d.cxx $(LIB_DIR)/libNuQR.a $(LIB_DIR)/libNuShared.a
	$(CXX) -o $(BIN_DIR)/test_qr_y2d test_qr_y2d.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuQR -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm  

$(BIN_DIR)/test_scala_qr_2d: test_scala_qr_2d.cxx $(LIB_DIR)/libNuQR.a $(LIB_DIR)/libNuShared.a
	$(CXX) -o $(BIN_DIR)/test_scala_qr_2d test_scala_qr_2d.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuQR -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm  

$(BIN_DIR)/test_qr_tree_2d: test_qr_tree_2d.cxx $(LIB_DIR)/libNuQR.a $(LIB_DIR)/libNuShared.a
	$(CXX) -o $(BIN_DIR)/test_qr_tree_2d test_qr_tree_2d.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuQR -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm  

$(BIN_DIR)/test_qr_butterfly_2d: test_qr_butterfly_2d.cxx $(LIB_DIR)/libNuQR.a $(LIB_DIR)/libNuShared.a
	$(CXX) -o $(BIN_DIR)/test_qr_butterfly_2d test_qr_butterfly_2d.cxx $(CXXFLAGS) $(DEFS) $(INCLUDES) \
	-L$(LIB_DIR) -lNuQR -lNuShared $(BLAS_LIBS) $(LDFLAGS) -lm  

clean:
	rm -f *.o

