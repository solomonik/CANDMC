include ../config.mk

test: MM_tests LU_tests QR_tests SE_tests

MM_TESTS := test_spc topo_pdgemm_unit
LU_TESTS := lu_25d_np_test lu_25d_pp_test lu_25d_tp_test  
QR_TESTS := test_bitree_tsqr test_construct_tsqr_Q test_hh_recon test_qr_y2d \
            test_qr_2d test_qr_butterfly_2d test_qr_tree_2d test_scala_qr_2d 
SE_TESTS := test_scala_sym_eig test_full2band test_full2band_scala \
            test_full2band_3d test_band_to_band 

MM_tests $(MM_TESTS):
	$(MAKE) $@ -C MM

LU_tests $(LU_TESTS):
	$(MAKE) $@ -C LU

QR_tests $(QR_TESTS):
ifneq (,$(findstring DLAPACKHASTSQR=1,$(DEFS))) 
	$(MAKE) $@ -C QR; 
endif

SE_tests $(SE_TESTS):
ifneq (,$(findstring DLAPACKHASTSQR=1,$(DEFS))) 
	$(MAKE) $@ -C SE; 
endif

clean:
	$(MAKE) $@ -C MM; \
	$(MAKE) $@ -C LU; \
	$(MAKE) $@ -C QR; \
	$(MAKE) $@ -C SE;
