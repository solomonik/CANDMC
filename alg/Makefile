include ../config.mk

LIB_DIR = ../lib

CANDMC: $(LIB_DIR)/libCANDMC.a

$(LIB_DIR)/libCANDMC.a: $(LIB_DIR)/libCANQR.a $(LIB_DIR)/libCANLU.a \
  $(LIB_DIR)/libCANMM.a $(LIB_DIR)/libCANShared.a $(LIB_DIR)/libCANSE.a
ifneq (,$(findstring DLAPACKHASTSQR=1,$(DEFS)))
	$(AR) -x $(LIB_DIR)/libCANSE.a; \
	$(AR) -x $(LIB_DIR)/libCANQR.a; \
	$(AR) -x $(LIB_DIR)/libCANMM.a; \
	$(AR) -x $(LIB_DIR)/libCANShared.a; \
	$(AR) -crs $(LIB_DIR)/libCANDMC.a *.o; rm *.o;
else
	$(AR) -x $(LIB_DIR)/libCANMM.a; $(AR) -x $(LIB_DIR)/libCANLU.a; \
	$(AR) -x $(LIB_DIR)/libCANShared.a; \
	$(AR) -crs $(LIB_DIR)/libCANDMC.a *.o; rm *.o;
endif 

$(LIB_DIR)/libCANQR.a: $(LIB_DIR)/libCANShared.a CANQR
CANQR: 
ifneq (,$(findstring DLAPACKHASTSQR=1,$(DEFS))) 
	$(MAKE) $@ -C QR; 
else
	$(AR) -crs $(LIB_DIR)/libCANQR.a;
endif

$(LIB_DIR)/libCANSE.a: $(LIB_DIR)/libCANQR.a CANSE
CANSE: 
	$(MAKE) $@ -C SE; 
#	ar crs $(LIB_DIR)/libCANSE.a;

$(LIB_DIR)/libCANMM.a: $(LIB_DIR)/libCANShared.a  CANMM
CANMM:
	$(MAKE) $@ -C MM; 

$(LIB_DIR)/libCANShared.a: CANShared
CANShared:
	$(MAKE) $@ -C shared; 

$(LIB_DIR)/libCANLU.a: $(LIB_DIR)/libCANShared.a CANLU
CANLU:
	$(MAKE) $@ -C LU; 

clean:
	$(MAKE) $@ -C MM; \
	$(MAKE) $@ -C LU; \
	$(MAKE) $@ -C QR; \
	$(MAKE) $@ -C SE; \
	$(MAKE) $@ -C shared         
