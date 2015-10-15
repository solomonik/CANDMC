include ../../config.mk

LIB_DIR = ../../lib

CANQR: CANQRrec 
	$(MAKE) $(LIB_DIR)/libCANQR.a

CANQRrec:
	if ($(MAKE) CANQR -C tsqr); then \
  	if ($(MAKE) CANQR -C hh_recon); then \
 	    $(MAKE) CANQR -C qr_2d;  \
    fi \
  fi

$(LIB_DIR)/libCANQR.a: tsqr/* hh_recon/* qr_2d/*
	$(AR) -crs $(LIB_DIR)/libCANQR.a tsqr/*.o hh_recon/*.o qr_2d/*.o
      
clean: 
	$(MAKE) $@ -C tsqr; \
	$(MAKE) $@ -C hh_recon; \
	$(MAKE) $@ -C qr_2d;
