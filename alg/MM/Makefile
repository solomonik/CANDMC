include ../../config.mk

LIB_DIR = ../../lib

NuMM: NuMMrec $(LIB_DIR)/libNuMM.a

NuMMrec:
	$(MAKE) NuMM -C splitdim_cannon; \
	$(MAKE) NuMM -C topo_pdgemm; 

#splitdim_cannon/*.o topo_pdgemm/*.o
$(LIB_DIR)/libNuMM.a: NuMMrec
	$(AR) -crs $(LIB_DIR)/libNuMM.a splitdim_cannon/*.o topo_pdgemm/*.o

clean:
	$(MAKE) $@ -C splitdim_cannon; \
	$(MAKE) $@ -C topo_pdgemm; 
