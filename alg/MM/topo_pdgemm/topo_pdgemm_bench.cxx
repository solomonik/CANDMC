/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <algorithm>
#include "../../shared/util.h"
#include "topo_pdgemm_algs.h"

static
void dcn_bench(int64_t const        n,
               int const        myRank,
               int const        numPes,
               int const        c_rep,
               int const        niter,
               int const        nwarm,
               int const        ovp,
               CommData         *cdt){

  CommData_t *cdt_glb = cdt;

  int info;

  if (myRank == 0){
    printf("BENCHMARKING 4D MM SUMMA / TOPO BCAST ALG\n");
  }

  const int x1_np = sqrt(numPes/c_rep);
  const int x2_np = sqrt(c_rep);
  const int64_t b = n / (x1_np*x2_np);

  if (myRank == 0){
    printf("NUM X1 PROCS IS %d\n", x1_np);
    printf("NUM Y1 PROCS IS %d\n", x1_np);
    printf("NUM X2 PROCS IS %d\n", x2_np);
    printf("NUM Y2 PROCS IS %d\n", x2_np);
  }

  if (x1_np*x1_np*x2_np*x2_np != numPes){
    if (myRank == 0)
      printf("ERROR: PROCESSOR GRID MISMATCH\n");
    ABORT;
  }

  if (n%(x1_np*x2_np) != 0){
    if (myRank == 0)
      printf("MATRIX DIMENSION MUST BE DIVISBLE BY X1_NP*X2_NP\n");
    ABORT;
  }

  const int nreq = 2;
  const int nbcast = 2;

  CommData_t cdt_x1, cdt_x2, cdt_y1, cdt_y2;

  SETUP_SUB_COMM(cdt_glb, (&cdt_y2),
                 myRank/(x1_np*x1_np*x2_np),
                 myRank%(x1_np*x1_np*x2_np),
                 x2_np, nreq, nbcast);
  SETUP_SUB_COMM(cdt_glb, (&cdt_x2),
                 ((myRank/(x1_np*x1_np))%x2_np),
                 (myRank%(x1_np*x1_np))*x2_np + myRank/(x1_np*x1_np*x2_np),
                 x2_np, nreq, nbcast);
  SETUP_SUB_COMM(cdt_glb, (&cdt_y1),
                 ((myRank/x1_np)%x1_np),
                 (myRank/(x1_np*x1_np))*x1_np + (myRank%x1_np),
                 x1_np, nreq, nbcast);
  SETUP_SUB_COMM(cdt_glb, (&cdt_x1),
                 (myRank%x1_np),
                 myRank/x1_np,
                 x1_np, nreq, nbcast);

  int64_t i,j,ib,jb;
  int iter;

  double * mat_A, * mat_B, * mat_C, * buffer;

  double start_t, end_t, sum_t;
  int seed = 1000;

  assert(posix_memalign((void**)&mat_A,
                        ALIGN_BYTES,
                        b*b*sizeof(double)) == 0);
  assert(posix_memalign((void**)&mat_B,
                        ALIGN_BYTES,
                        b*b*sizeof(double)) == 0);
  assert(posix_memalign((void**)&mat_C,
                        ALIGN_BYTES,
                        b*b*sizeof(double)) == 0);
  if (ovp)
    assert(posix_memalign((void**)&buffer,
                          ALIGN_BYTES,
                          5*b*b*sizeof(double)) == 0);
  else
    assert(posix_memalign((void**)&buffer,
                          ALIGN_BYTES,
                          3*b*b*sizeof(double)) == 0);

  for (i=0; i<b; i++){
    srand48(seed);
    for (j=0; j<b; j++){
      mat_A[i*b+j] = drand48();
      mat_B[i*b+j] = drand48();
    }
  }

  ctb_args_t p;

  p.n           = n;
  p.lda_A       = b;
  p.lda_B       = b;
  p.lda_C       = b;
  if (ovp)
    p.buffer_size = 5*b*b*sizeof(double);
  else
    p.buffer_size = 3*b*b*sizeof(double);
  p.trans_A     = 'N';
  p.trans_B     = 'N';
  p.ovp         = ovp;

  for (iter = 0; iter < nwarm; iter++){
    bcast_cannon_4d(&p, mat_A, mat_B, mat_C, buffer,
                    &cdt_x1, &cdt_y1, &cdt_x2, &cdt_y2);
  }
  __CM(3, cdt_glb, numPes, niter, myRank);
  start_t = TIME_SEC();
  for (iter = 0; iter < niter; iter++){
    bcast_cannon_4d(&p, mat_A, mat_B, mat_C, buffer,
                    &cdt_x1, &cdt_y1, &cdt_x2, &cdt_y2);
  }
  end_t = TIME_SEC();
  sum_t = end_t-start_t;


  if (myRank == 0){
    printf("Completed %u iterations\n", niter);
    printf("Time elapsed per iteration: %lf\n", sum_t/niter);
    printf("Gigaflops: %lf\n", 2.*n*n*n/(sum_t/niter)*1.E-9);
  }
  __CM(2, cdt_glb, numPes, niter, myRank);
  FREE_CDT((&cdt_x1));
  FREE_CDT((&cdt_x2));
  FREE_CDT((&cdt_y1));
  FREE_CDT((&cdt_y2));
}

static
void d25_bench(int64_t const        n,
               int const        myRank,
               int const        numPes,
               int const        c_rep,
#ifdef USE_MIC
			   int 			     micportion,
			   int				 mic_id,
#endif
               int const        niter,
               int const        nwarm,
               int const        ovp,
               CommData const   *cdt){

  const CommData_t *cdt_glb = cdt;

  int info;

  if (myRank == 0){
    printf("BENCHMARKING 2.5D MM TOPO BCAST ALG\n");
  }

  const int num_pes_dim = sqrt(numPes/c_rep);
  const int64_t b = n / num_pes_dim;

#ifdef USE_MIC
  if(micportion < 0)
	micportion = b;
#endif

  if (myRank == 0){
    printf("NUM X PROCS IS %d\n", num_pes_dim);
    printf("NUM Y PROCS IS %d\n", num_pes_dim);
    printf("NUM Z PROCS IS %d\n", c_rep);
  }

  if (num_pes_dim*num_pes_dim*c_rep != numPes){
    if (myRank == 0)
      printf("ERROR: PROCESSOR GRID MISMATCH\n");
    ABORT;
  }

  if (n%num_pes_dim != 0){
    if (myRank == 0)
      printf("MATRIX DIMENSION MUST BE DIVISBLE BY THE NUMBER OF PROCESSORS\n");
    ABORT;
  }

  int layerRank, intraLayerRank, myRow, myCol;
  const int nreq = 2;
  const int nbcast = 2;

  CommData_t *cdt_row = (CommData_t*)malloc(sizeof(CommData_t));
  CommData_t *cdt_col = (CommData_t*)malloc(sizeof(CommData_t));
  CommData_t *cdt_kdir = (CommData_t*)malloc(sizeof(CommData_t));
  RSETUP_KDIR_COMM(myRank, numPes, c_rep, cdt_kdir, layerRank, intraLayerRank,1,1);
  RSETUP_LAYER_COMM(num_pes_dim, layerRank, intraLayerRank, cdt_row,
                    cdt_col, myRow, myCol, nreq, nbcast);

  int64_t i,j;
  int iter;

  double * mat_A, * mat_B, * mat_C, * buffer;

  double start_t, end_t, sum_t;
  int seed = 1000;

  assert(posix_memalign((void**)&mat_A,
                        ALIGN_BYTES,
                        b*b*sizeof(double)) == 0);
  assert(posix_memalign((void**)&mat_B,
                        ALIGN_BYTES,
                        b*b*sizeof(double)) == 0);
  assert(posix_memalign((void**)&mat_C,
                        ALIGN_BYTES,
                        b*b*sizeof(double)) == 0);
  if (ovp)
    assert(posix_memalign((void**)&buffer,
                          ALIGN_BYTES,
                          5*b*b*sizeof(double)) == 0);
  else
    assert(posix_memalign((void**)&buffer,
                          ALIGN_BYTES,
                          3*b*b*sizeof(double)) == 0);

  if (layerRank == 0){
    srand48(seed);
    for (i=0; i<b; i++){
      for (j=0; j<b; j++){
        mat_A[i*b+j] = drand48();
        mat_B[i*b+j] = drand48();
      }
    }
  }
  BCAST(mat_A, b*b, MPI_DOUBLE, 0, cdt_kdir);
  BCAST(mat_B, b*b, MPI_DOUBLE, 0, cdt_kdir);

  ctb_args_t p;

  p.n           = n;
  p.lda_A       = b;
  p.lda_B       = b;
  p.lda_C       = b;
  p.buffer_size = 5*b*b*sizeof(double);
  p.trans_A     = 'N';
  p.trans_B     = 'N';

  sum_t = 0.0;

  for (iter = 0; iter < nwarm; iter++){
    if (ovp)
      d25_summa_ovp(&p, mat_A, mat_B, mat_C, buffer,
#ifdef USE_MIC
					micportion, mic_id,
#endif
					cdt_row, cdt_col, cdt_kdir);
    else
      d25_summa(&p, mat_A, mat_B, mat_C, buffer,
#ifdef USE_MIC
				micportion, mic_id,
#endif
				cdt_row, cdt_col, cdt_kdir);
  }
  __CM(3, cdt_glb, numPes, niter, myRank);
  start_t = TIME_SEC();
  for (iter = 0; iter < niter; iter++){
    if (ovp)
      d25_summa_ovp(&p, mat_A, mat_B, mat_C, buffer,
#ifdef USE_MIC
					micportion, mic_id,
#endif
					cdt_row, cdt_col, cdt_kdir);
    else
      d25_summa(&p, mat_A, mat_B, mat_C, buffer,
#ifdef USE_MIC
				micportion, mic_id,
#endif
				cdt_row, cdt_col, cdt_kdir);
  }
  end_t = TIME_SEC();
  sum_t = end_t-start_t;


  if (myRank == 0){
    printf("Completed %u iterations\n", niter);
    printf("Time elapsed per iteration: %lf\n", sum_t/niter);
    printf("Gigaflops: %lf\n", 2.*n*n*n/(sum_t/niter)*1.E-9);
  }
  __CM(2, cdt_glb, numPes, niter, myRank);
  FREE_CDT(cdt_row);
  FREE_CDT(cdt_col);
}


static
void ctb_bench(int64_t const        n,
               int const        myRank,
               int const        numPes,
               int const        niter,
               int const        nwarm,
               CommData const   *cdt){

  const CommData_t *cdt_glb = cdt;

  int info;

  if (myRank == 0){
    printf("BENCHMARKING SUMMA TOPO BCAST ALG\n");
  }


  const int num_pes_dim = sqrt(numPes);
  const int64_t b = n / num_pes_dim;

  if (myRank == 0){
    printf("NUM X PROCS IS %d\n", num_pes_dim);
    printf("NUM Y PROCS IS %d\n", num_pes_dim);
  }

  if (num_pes_dim*num_pes_dim != numPes){
    if (myRank == 0)
      printf("ERROR: PROCESSOR GRID MISMATCH\n");
    ABORT;
  }

  if (n%num_pes_dim != 0){
    if (myRank == 0)
      printf("MATRIX DIMENSION MUST BE DIVISBLE BY THE NUMBER OF PROCESSORS\n");
    ABORT;
  }

  int layerRank, intraLayerRank, myRow, myCol;
  const int nreq = 2;
  const int nbcast = 2;

  CommData_t *cdt_row = (CommData_t*)malloc(sizeof(CommData_t));
  CommData_t *cdt_col = (CommData_t*)malloc(sizeof(CommData_t));
  RSETUP_LAYER_COMM(num_pes_dim, 0, myRank, cdt_row,
                    cdt_col, myRow, myCol, nreq, nbcast);

  int64_t i,j;
  int iter;

  double * mat_A, * mat_B, * mat_C, * buffer;

  double start_t, end_t, sum_t;
  int seed = 1000;

  assert(posix_memalign((void**)&mat_A,
                        ALIGN_BYTES,
                        b*b*sizeof(double)) == 0);
  assert(posix_memalign((void**)&mat_B,
                        ALIGN_BYTES,
                        b*b*sizeof(double)) == 0);
  assert(posix_memalign((void**)&mat_C,
                        ALIGN_BYTES,
                        b*b*sizeof(double)) == 0);
  assert(posix_memalign((void**)&buffer,
                        ALIGN_BYTES,
                        4*b*b*sizeof(double)) == 0);

  srand48(seed);
  for (i=0; i<b; i++){
    for (j=0; j<b; j++){
      mat_A[i*b+j] = drand48();
      mat_B[i*b+j] = drand48();
    }
  }

  ctb_args_t p;

  p.n           = n;
  p.lda_A       = b;
  p.lda_B       = b;
  p.lda_C       = b;
  p.buffer_size = 4*b*b*sizeof(double);
  p.trans_A     = 'N';
  p.trans_B     = 'N';

  sum_t = 0.0;

  for (iter = 0; iter < nwarm; iter++){
    summa(&p, mat_A, mat_B, mat_C, buffer, cdt_row, cdt_col);
  }
  __CM(3, cdt_glb, numPes, niter, myRank);
  start_t = TIME_SEC();
  for (iter = 0; iter < niter; iter++){
    summa(&p, mat_A, mat_B, mat_C, buffer, cdt_row, cdt_col);
  }
  end_t = TIME_SEC();
  sum_t = end_t-start_t;


  if (myRank == 0){
    printf("Completed %u iterations\n", niter);
    printf("Time elapsed per iteration: %lf\n", sum_t/niter);
    printf("Gigaflops: %lf\n", 2.*n*n*n/(sum_t/niter)*1.E-9);
  }
  __CM(2, cdt_glb, numPes, niter, myRank);
  FREE_CDT(cdt_row);
  FREE_CDT(cdt_col);
}

static
char* getCmdOption(char ** begin, char ** end, const std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

int main(int argc, char **argv) {
  int myRank, numPes;
  int64_t n;
  int niter, nwarm, c_rep, ovp, cut4d, exec_mask;

  CommData_t *cdt_glb = (CommData_t*)malloc(sizeof(CommData_t));
  RINIT_COMM(numPes, myRank, 4, 4, cdt_glb);

#ifdef USE_MIC
  int micportion, twomics, mic_id=0;
	if (getCmdOption(argv, argv+argc, "-micportion")){
		micportion = atoi(getCmdOption(argv, argv+argc, "-micportion"));
	} else micportion = -1;
	if (getCmdOption(argv, argv+argc, "-twomics")){
		twomics = atoi(getCmdOption(argv, argv+argc, "-twomics"));
	} else twomics = 1;
	if(twomics)
		mic_id = myRank%2;
#endif

  if (getCmdOption(argv, argv+argc, "-n")){
    n = atoi(getCmdOption(argv, argv+argc, "-n"));
    if (n <= 0) n = 128;
  } else n = 128;

  if (getCmdOption(argv, argv+argc, "-niter")){
    niter = atoi(getCmdOption(argv, argv+argc, "-niter"));
    if (niter <= 0) niter = 1;
  } else niter = 3;

  if (getCmdOption(argv, argv+argc, "-nwarm")){
    nwarm = atoi(getCmdOption(argv, argv+argc, "-nwarm"));
    if (nwarm <= 0) nwarm = 0;
  } else nwarm = 1;

  if (getCmdOption(argv, argv+argc, "-c_rep")){
    c_rep = atoi(getCmdOption(argv, argv+argc, "-c_rep"));
    if (c_rep <= 1) c_rep = 1;
  } else c_rep = 1;
  if (getCmdOption(argv, argv+argc, "-ovp")){
    ovp = atoi(getCmdOption(argv, argv+argc, "-ovp"));
    if (ovp < 0) ovp = 1;
  } else ovp = 0;
  if (getCmdOption(argv, argv+argc, "-cut4d")){
    cut4d = atoi(getCmdOption(argv, argv+argc, "-cut4d"));
    if (cut4d < 0) cut4d = 1;
  } else cut4d = 1;

  if (myRank == 0) {
    printf("benchmarking topology-aware pdgemms.\n");
#ifdef USE_MIC
    printf("-n=" PRId64 " -niter=%d -nwarm=%d -c_rep=%d -micportion=%d\n", n,niter,nwarm,c_rep,micportion);
#else
    printf("-n=" PRId64 " -niter=%d -nwarm=%d -c_rep=%d\n", n,niter,nwarm,c_rep);
#endif
    printf("-ovp=%d (overlap)\n", ovp);
#ifdef USE_DCMF
    printf("USING DCMF FOR COMMUNICATION\n");
#else
    printf("USING MPI FOR COMMUNICATION\n");
#endif

#ifdef USE_MIC
	printf("USE_MIC\n");
	printf("USE 2 MICs per node: %d\n", twomics);
#endif

  }

#ifdef TAU
  TAU_PROFILE_TIMER(timer, "main", "int (int, char**)", TAU_USER);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(myRank);
  TAU_PROFILE_SET_CONTEXT(0);
#endif

  GLOBAL_BARRIER(cdt_glb);
  if (c_rep >= 1)
	d25_bench(n, myRank, numPes, c_rep,
#ifdef USE_MIC
			  micportion, mic_id,
#endif
			  niter, nwarm, ovp, cdt_glb);
  GLOBAL_BARRIER(cdt_glb);
  TAU_PROFILE_STOP(timer);
  COMM_EXIT;
  return 0;
}

