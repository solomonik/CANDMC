include ../../config.mk

LIB_DIR = ../../lib

CANMM: CANMMrec $(LIB_DIR)/libCANMM.a

CANMMrec:
	$(MAKE) CANMM -C splitdim_cannon; \
	$(MAKE) CANMM -C topo_pdgemm; 

#splitdim_cannon/*.o topo_pdgemm/*.o
$(LIB_DIR)/libCANMM.a: CANMMrec
	$(AR) -crs $(LIB_DIR)/libCANMM.a splitdim_cannon/*.o topo_pdgemm/*.o

clean:
	$(MAKE) $@ -C splitdim_cannon; \
	$(MAKE) $@ -C topo_pdgemm; 
