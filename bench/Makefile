include ../config.mk

MM_BENCHMARKS := bench_spc topo_pdgemm_bench

LU_BENCHMARKS := lu_25d_np_bench lu_25d_pp_bench lu_25d_tp_bench bench_scala_lu bench_scala_cholesky

QR_BENCHMARKS := bench_hh_recon bench_qr_2d bench_qr_butterfly_2d bench_qr_tree_2d \
                 bench_scala_qr bench_qr_2d_hh_scala  bench_qr_seq bench_qr_y2d

SE_BENCHMARKS := bench_scala_sym_eig bench_full2band bench_elpa_sym_eig bench_band_to_band bench_full2band_3d

bench: MM_benchmarks LU_benchmarks QR_benchmarks SE_benchmarks

MM_benchmarks $(MM_BENCHMARKS):
	$(MAKE) $@ -C MM

LU_benchmarks $(LU_BENCHMARKS):
	$(MAKE) $@ -C LU

SE_benchmarks $(SE_BENCHMARKS):
ifneq (,$(findstring DLAPACKHASTSQR=1,$(DEFS))) 
	$(MAKE) $@ -C SE
endif

QR_benchmarks $(QR_BENCHMARKS):
	$(MAKE) $@ -C QR; 

clean:
	$(MAKE) $@ -C MM; \
	$(MAKE) $@ -C LU; \
	$(MAKE) $@ -C QR; \
	$(MAKE) $@ -C SE;
