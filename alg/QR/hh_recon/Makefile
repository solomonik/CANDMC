include ../../../config.mk

LIB_DIR = ../../../lib

CANQR: $(LIB_DIR)/libCANQR.a

$(LIB_DIR)/libCANQR.a:  hh_recon.o yamamoto.o

hh_recon.o: hh_recon.h hh_recon.cxx
	$(CXX) -c hh_recon.cxx -o hh_recon.o $(CXXFLAGS) $(DEFS)

yamamoto.o: yamamoto.h yamamoto.cxx
	$(CXX) -c yamamoto.cxx -o yamamoto.o $(CXXFLAGS) $(DEFS)


clean:
	rm -f *.o 
