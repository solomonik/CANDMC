include ../../../config.mk

LIB_DIR = ../../../lib

NuQR: $(LIB_DIR)/libNuQR.a

$(LIB_DIR)/libNuQR.a:  qr_2d.o qr_tree_2d.o qr_butterfly_2d.o qr_y2d.o 

qr_y2d.o: qr_y2d.cxx 
	$(CXX) -c qr_y2d.cxx -o qr_y2d.o $(CXXFLAGS) $(DEFS)

qr_2d.o: qr_2d.cxx 
	$(CXX) -c qr_2d.cxx -o qr_2d.o $(CXXFLAGS) $(DEFS)

qr_tree_2d.o: qr_tree_2d.cxx 
	$(CXX) -c qr_tree_2d.cxx -o qr_tree_2d.o $(CXXFLAGS) $(DEFS)

qr_butterfly_2d.o: qr_butterfly_2d.cxx 
	$(CXX) -c qr_butterfly_2d.cxx -o qr_butterfly_2d.o $(CXXFLAGS) $(DEFS)

clean:
	rm -f *.o 
