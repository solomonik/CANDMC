include ../../../config.mk

LIB_DIR = ../../../lib

NuQR: $(LIB_DIR)/libNuQR.a

$(LIB_DIR)/libNuQR.a: butterfly_tsqr.o bitree_tsqr.o construct_tsqr_Q.o apply_tsqr_QT.o apply_butterfly_tsqr_QT.o butterfly_construct_Q.o

butterfly_tsqr.o: bitree_tsqr.o butterfly_tsqr.cxx 
	$(CXX) -c butterfly_tsqr.cxx -o butterfly_tsqr.o $(CXXFLAGS) $(DEFS)

bitree_tsqr.o: bitree_tsqr.h bitree_tsqr.cxx
	$(CXX) -c bitree_tsqr.cxx -o bitree_tsqr.o $(CXXFLAGS) $(DEFS)

construct_tsqr_Q.o: bitree_tsqr.h construct_tsqr_Q.cxx 
	$(CXX) -c construct_tsqr_Q.cxx -o construct_tsqr_Q.o $(CXXFLAGS) $(DEFS)

butterfly_construct_Q.o: butterfly_tsqr.h butterfly_construct_Q.cxx 
	$(CXX) -c butterfly_construct_Q.cxx -o butterfly_construct_Q.o $(CXXFLAGS) $(DEFS)

apply_tsqr_QT.o: bitree_tsqr.h apply_tsqr_QT.cxx 
	$(CXX) -c apply_tsqr_QT.cxx -o apply_tsqr_QT.o $(CXXFLAGS) $(DEFS)

apply_butterfly_tsqr_QT.o: butterfly_tsqr.o apply_butterfly_tsqr_QT.cxx 
	$(CXX) -c apply_butterfly_tsqr_QT.cxx -o apply_butterfly_tsqr_QT.o $(CXXFLAGS) $(DEFS)

clean:
	rm -f *.o 
