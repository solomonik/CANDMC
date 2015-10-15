include config.mk

TESTS := lu_25d_np_test  lu_25d_pp_test  lu_25d_tp_test  test_bitree_tsqr \
         test_construct_tsqr_Q  test_hh_recon  test_qr_2d test_qr_y2d  test_qr_butterfly_2d \
         test_qr_tree_2d  test_scala_qr_2d  test_spc  topo_pdgemm_unit test_scala_sym_eig \
         test_full2band test_full2band_scala test_full2band_3d

BENCHMARKS := bench_hh_recon  bench_qr_2d  bench_qr_butterfly_2d  bench_qr_tree_2d \
              bench_scala_qr bench_qr_2d_hh_scala  bench_spc  bench_scala_sym_eig  lu_25d_np_bench  \
              lu_25d_pp_bench  lu_25d_tp_bench  topo_pdgemm_bench bench_qr_seq \
              bench_full2band bench_elpa_sym_eig bench_full2band_3d bench_qr_y2d


lib: CANDMC

CANMM CANLU CANQR CANSE CANDMC: 
	$(MAKE) $@ -C alg

test: CANDMC 
	$(MAKE) $@ -C test

bench: CANDMC
	$(MAKE) $@ -C bench

$(TESTS): CANDMC
	$(MAKE) $@ -C test

$(BENCHMARKS): CANDMC
	$(MAKE) $@ -C bench

clean:
	rm -f lib/libCANDMC.a lib/libCANMM.a lib/libCANLU.a lib/libCANQR.a lib/libCANSE.a lib/libCANShared.a; \
  cd bin/tests; rm -f $(TESTS); cd ../..; \
  cd bin/benchmarks; rm -f $(BENCHMARKS); cd ../..; \
	$(MAKE) $@ -C alg; \
	$(MAKE) $@ -C test; \
	$(MAKE) $@ -C bench; 	
