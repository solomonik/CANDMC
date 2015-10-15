include ../../config.mk

LIB_DIR=../../lib

NuShared: $(LIB_DIR)/libNuShared.a

$(LIB_DIR)/libNuShared.a: util.o timer.o lapack.o
	$(AR) -crs $(LIB_DIR)/libNuShared.a util.o timer.o lapack.o

util.o: util.h util.cxx
	$(CXX) -c  util.cxx -o util.o $(CXXFLAGS) $(DEFS) 

timer.o: timer.h timer.cxx
	$(CXX) -c  timer.cxx -o timer.o $(CXXFLAGS) $(DEFS) 

lapack.o: lapack.h lapack.cxx
	$(CXX) -c  lapack.cxx -o lapack.o $(CXXFLAGS) $(DEFS) 

clean:
	rm -f *.o
