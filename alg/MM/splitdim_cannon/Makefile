include ../../../config.mk

LIB_DIR = ../../../lib

NuMM: $(LIB_DIR)/libNuMM.a

$(LIB_DIR)/libNuMM.a: spcannon.o

spcannon.o: spcannon.cxx spcannon.h spcannon_internal.h
	$(CXX) -c spcannon.cxx -o spcannon.o $(CXXFLAGS) $(DEFS)

clean: 
	rm -f *.o
