include ../../config.mk

LIB_DIR = ../../lib

CANLU: $(LIB_DIR)/libCANLU.a

$(LIB_DIR)/libCANLU.a: lu_25d_pvt.o tnmt_pvt.o partial_pvt.o
	$(AR) -crs $(LIB_DIR)/libCANLU.a *.o

lu_25d_pvt.o: lu_25d_pvt.cxx
	$(CXX) -c lu_25d_pvt.cxx -o lu_25d_pvt.o $(CXXFLAGS) $(DEFS) 

tnmt_pvt.o: tnmt_pvt.h tnmt_pvt.cxx
	$(CXX) -c tnmt_pvt.cxx -o tnmt_pvt.o $(CXXFLAGS) $(DEFS) 

partial_pvt.o: partial_pvt.h partial_pvt.cxx
	$(CXX) -c partial_pvt.cxx -o partial_pvt.o $(CXXFLAGS) $(DEFS) 

clean:
	rm -f *.o 
