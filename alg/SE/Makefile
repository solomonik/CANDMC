include ../../config.mk

LIB_DIR = ../../lib

CANSE: $(LIB_DIR)/libCANSE.a

$(LIB_DIR)/libCANSE.a: full_to_band.o full_to_band_scala.o dmatrix.o full_to_band_3d.o 
	$(AR) -crs $(LIB_DIR)/libCANSE.a *.o

dmatrix.o: dmatrix.cxx 
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(CXX) -c dmatrix.cxx -o dmatrix.o $(CXXFLAGS) $(DEFS)
endif

full_to_band.o: full_to_band.cxx CANSE.h
	$(CXX) -c full_to_band.cxx -o full_to_band.o $(CXXFLAGS) $(DEFS)

full_to_band_3d.o: dmatrix.o full_to_band_3d.cxx CANSE.h
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(CXX) -c full_to_band_3d.cxx -o full_to_band_3d.o $(CXXFLAGS) $(DEFS)
endif

full_to_band_scala.o: full_to_band_scala.cxx CANSE.h
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(CXX) -c full_to_band_scala.cxx -o full_to_band_scala.o $(CXXFLAGS) $(DEFS)
endif

clean:  
	rm -f *.o 
