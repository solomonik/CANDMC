include ../../../config.mk

LIB_DIR = ../../../lib

CANMM: $(LIB_DIR)/libCANMM.a

$(LIB_DIR)/libCANMM.a: spcannon.o

spcannon.o: spcannon.cxx spcannon.h spcannon_internal.h
	$(CXX) -c spcannon.cxx -o spcannon.o $(CXXFLAGS) $(DEFS)

clean: 
	rm -f *.o
