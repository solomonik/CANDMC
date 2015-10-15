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
void dcn_unit(int64_t const   n,
        int const   myRank,
        int const   numPes,
        int const   c_rep,
        int const   seed,
        int const   ovp,
        CommData    *cdt){

  CommData_t *cdt_glb = cdt;

  int info;

  if (myRank == 0){
    printf("TESTING 4D MM CANNON / TOPO BCAST ALG\n");
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

  const int nreq = 4;
  const int nbcast = 4;

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

  int i,j,iter,ib,jb;

  double * mat_A, * mat_B, * mat_C, * buffer;

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
      5*b*b*sizeof(double)) == 0);

  for (i=0; i<b; i++){
    for (j=0; j<b; j++){
      srand48((cdt_x1.rank*cdt_x2.np*b+cdt_x2.rank*b+i)*n +
         cdt_y1.rank*cdt_y2.np*b+cdt_y2.rank*b+j);
      mat_A[i*b+j] = drand48();
      mat_B[i*b+j] = drand48();
    }
  }

  ctb_args_t p;

  p.n   = n;
  p.lda_A = b;
  p.lda_B = b;
  p.lda_C = b;
  if (ovp)
    p.buffer_size = 5*b*b*sizeof(double);
  else
    p.buffer_size = 3*b*b*sizeof(double);
  p.trans_A = 'N';
  p.trans_B = 'N';
  p.ovp   = ovp;

  DEBUG_PRINTF("P[%d][%d], mat_A[0] = %lf, mat_B[0] = %lf, mat_C[0] = %lf\n",
         cdt_x1.rank,cdt_y1.rank,mat_A[0],mat_B[0],mat_C[0]);
  bcast_cannon_4d(&p, mat_A, mat_B, mat_C, buffer,
      &cdt_x1, &cdt_y1, &cdt_x2, &cdt_y2);
  DEBUG_PRINTF("P[%d][%d], mat_A[0] = %lf, mat_B[0] = %lf, mat_C[0] = %lf\n",
         cdt_x1.rank,cdt_y1.rank,mat_A[0],mat_B[0],mat_C[0]);

  free(mat_A);
  free(mat_B);
  free(buffer);

  assert(posix_memalign((void**)&mat_A,
      ALIGN_BYTES,
      n*n*sizeof(double)) == 0);
  assert(posix_memalign((void**)&mat_B,
      ALIGN_BYTES,
      n*n*sizeof(double)) == 0);
  assert(posix_memalign((void**)&buffer,
      ALIGN_BYTES,
      n*n*sizeof(double)) == 0);

  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      srand48(i*n+j);
      mat_A[i*n+j] = drand48();
      mat_B[i*n+j] = drand48();
    }
  }
  cdgemm('N','N',n,n,n,1.0,mat_A,n,mat_B,n,0.0,buffer,n);
#ifdef VERBOSE
  GLOBAL_BARRIER(cdt_glb);
  if (cdt_glb->rank == 0){
    print_matrix(mat_A,n,n);
    print_matrix(mat_B,n,n);
    print_matrix(buffer,n,n);
  }
  GLOBAL_BARRIER(cdt_glb);
#endif

  bool pass = true;
  bool global_pass;
  for (i=0; i<b; i++){
    for (j=0; j<b; j++){
      ib = cdt_x1.rank*cdt_x2.np*b+cdt_x2.rank*b+i;
      jb = cdt_y1.rank*cdt_y2.np*b+cdt_y2.rank*b+j;
      if (fabs(buffer[ib*n+jb]-mat_C[i*b+j]) > 1.E-6){
  pass = false;
  DEBUG_PRINTF("[%d] mat_C[%d,%d]=%lf should have been %lf\n",
          myRank,jb,ib,mat_C[i*b+j],buffer[ib*n+jb]);
      }
    }
  }
  REDUCE(&pass, &global_pass, 1, COMM_CHAR_T, COMM_OP_BAND, 0, cdt_glb);
  if (cdt_glb->rank == 0) {
    if (global_pass){
      printf("[%d] DC UNIT TEST PASSED\n",cdt_glb->rank);
    } else {
      printf("[%d] !!!DC UNIT TEST FAILED!!!\n",cdt_glb->rank);
    }
  }

  FREE_CDT((&cdt_x1));
  FREE_CDT((&cdt_x2));
  FREE_CDT((&cdt_y1));
  FREE_CDT((&cdt_y2));
}

static
void d25_unit(int64_t const   n,
        int const   myRank,
        int const   numPes,
        int const   c_rep,
        int const   seed,
        int const   ovp,
#ifdef	USE_MIC
		int const	 mic_id,
#endif
        CommData    *cdt){

  CommData_t *cdt_glb = cdt;

  int info;

  if (myRank == 0){
    printf("TESTING 2.5D MM TOPO BCAST ALG\n");
  }

  const int num_pes_dim = sqrt(numPes/c_rep);
  const int b = n / num_pes_dim;

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

  int64_t i,j,ib,jb;
  int iter;

  double * mat_A, * mat_B, * mat_C, * buffer;

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
    for (j=0; j<b; j++){
      srand48((myCol*b+i)*n + myRow*b+j);
      mat_A[i*b+j] = drand48();
      mat_B[i*b+j] = drand48();
    }
  }

  ctb_args_t p;

  p.n   = n;
  p.lda_A = b;
  p.lda_B = b;
  p.lda_C = b;
  p.buffer_size = 5*b*b*sizeof(double);
  p.trans_A = 'N';
  p.trans_B = 'N';

#ifdef USE_MIC
  int micportion = b/2;
#endif

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
  DEBUG_PRINTF("P[%d][%d], mat_A[0] = %lf, mat_B[0] = %lf, mat_C[0] = %lf\n",
         myRow,myCol,mat_A[0],mat_B[0],mat_C[0]);

  free(mat_A);
  free(mat_B);
  free(buffer);

  assert(posix_memalign((void**)&mat_A,
      ALIGN_BYTES,
      n*n*sizeof(double)) == 0);
  assert(posix_memalign((void**)&mat_B,
      ALIGN_BYTES,
      n*n*sizeof(double)) == 0);
  assert(posix_memalign((void**)&buffer,
      ALIGN_BYTES,
      n*n*sizeof(double)) == 0);

  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      srand48(i*n+j);
      mat_A[i*n+j] = drand48();
      mat_B[i*n+j] = drand48();
    }
  }
  cdgemm('N','N',n,n,n,1.0,mat_A,n,mat_B,n,0.0,buffer,n);
#ifdef VERBOSE
  GLOBAL_BARRIER(cdt_glb);
  if (myRow == 0 && myCol == 0){
    print_matrix(mat_A,n,n);
    print_matrix(mat_B,n,n);
    print_matrix(buffer,n,n);
  }
  GLOBAL_BARRIER(cdt_glb);
#endif

  bool pass = true;
  bool global_pass;
  if (layerRank == 0){
    for (i=0; i<b; i++){
      for (j=0; j<b; j++){
  ib = i+myCol*b;
  jb = j+myRow*b;
  if (fabs(buffer[ib*n+jb]-mat_C[i*b+j]) > 1.E-6){
    pass = false;
    DEBUG_PRINTF("[%d] mat_C[%d,%d]=%lf should have been %lf\n",
      myRank,jb,ib,mat_C[i*b+j],buffer[ib*n+jb]);
  }
      }
    }
  }
  REDUCE(&pass, &global_pass, 1, COMM_CHAR_T, COMM_OP_BAND, 0, cdt_glb);
  if (myRow ==0 && myCol == 0 && layerRank == 0){
    if (global_pass){
      printf("[%d][%d] D25 UNIT TEST PASSED\n",myRow,myCol);
    } else {
      printf("[%d][%d] !!!D25 UNIT TEST FAILED!!!\n",myRow,myCol);
    }
  }

  FREE_CDT(cdt_row);
  FREE_CDT(cdt_col);
  FREE_CDT(cdt_kdir);
}

static
void ctb_unit(int64_t const   n,
        int const   myRank,
        int const   numPes,
        int const   seed,
        CommData    *cdt){

  CommData_t *cdt_glb = cdt;

  int info;

  if (myRank == 0){
    printf("TESTING CANNON TOPO BCAST ALG\n");
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
  int iter,ib,jb;

  double * mat_A, * mat_B, * mat_C, * buffer;

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

  for (i=0; i<b; i++){
    for (j=0; j<b; j++){
      srand48((myCol*b+i)*n + myRow*b+j);
      mat_A[i*b+j] = drand48();
      mat_B[i*b+j] = drand48();
    }
  }

  ctb_args_t p;

  p.n   = n;
  p.lda_A = b;
  p.lda_B = b;
  p.lda_C = b;
  p.buffer_size = 4*b*b*sizeof(double);
  p.trans_A = 'N';
  p.trans_B = 'N';

  summa(&p, mat_A, mat_B, mat_C, buffer, cdt_row, cdt_col);
  DEBUG_PRINTF("P[%d][%d], mat_A[0] = %lf, mat_B[0] = %lf, mat_C[0] = %lf\n",
         myRow,myCol,mat_A[0],mat_B[0],mat_C[0]);

  free(mat_A);
  free(mat_B);
  free(buffer);

  assert(posix_memalign((void**)&mat_A,
      ALIGN_BYTES,
      n*n*sizeof(double)) == 0);
  assert(posix_memalign((void**)&mat_B,
      ALIGN_BYTES,
      n*n*sizeof(double)) == 0);
  assert(posix_memalign((void**)&buffer,
      ALIGN_BYTES,
      n*n*sizeof(double)) == 0);

  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      srand48(i*n+j);
      mat_A[i*n+j] = drand48();
      mat_B[i*n+j] = drand48();
    }
  }
  cdgemm('N','N',n,n,n,1.0,mat_A,n,mat_B,n,0.0,buffer,n);
#ifdef VERBOSE
  GLOBAL_BARRIER(cdt_glb);
  if (myRow == 0 && myCol == 0){
    print_matrix(mat_A,n,n);
    print_matrix(mat_B,n,n);
    print_matrix(buffer,n,n);
  }
  GLOBAL_BARRIER(cdt_glb);
#endif

  bool pass = true;
  bool global_pass;
  for (i=0; i<b; i++){
    for (j=0; j<b; j++){
      ib = i+myCol*b;
      jb = j+myRow*b;
      if (fabs(buffer[ib*n+jb]-mat_C[i*b+j]) > 1.E-6){
  pass = false;
  DEBUG_PRINTF("mat_C[%d,%d]=%lf should have been %lf\n",
          jb,ib,mat_C[i*b+j],buffer[ib*n+jb]);
      }
    }
  }
  REDUCE(&pass, &global_pass, 1, COMM_CHAR_T, COMM_OP_BAND, 0, cdt_glb);
  if (myRow ==0 && myCol == 0){
    if (global_pass){
      printf("[%d][%d] CTB UNIT TEST PASSED\n",myRow,myCol);
    } else {
      printf("[%d][%d] !!!CTB UNIT TEST FAILED!!!\n",myRow,myCol);
    }
  }

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
  int seed, c_rep, ovp, cut4d;

  CommData_t *cdt_glb = (CommData_t*)malloc(sizeof(CommData_t));
  RINIT_COMM(numPes, myRank, 4, 4, cdt_glb);

  if (getCmdOption(argv, argv+argc, "-n")){
    n = atoi(getCmdOption(argv, argv+argc, "-n"));
    if (n <= 0) n = 128;
  } else n = 128;
  if (getCmdOption(argv, argv+argc, "-seed")){
    seed = atoi(getCmdOption(argv, argv+argc, "-seed"));
    if (seed <= 0) seed = 1000;
  } else seed = 1000;
  if (getCmdOption(argv, argv+argc, "-c_rep")){
    c_rep = atoi(getCmdOption(argv, argv+argc, "-c_rep"));
    if (c_rep <= 1) c_rep = 1;
  } else c_rep = 1;
  if (getCmdOption(argv, argv+argc, "-ovp")){
    ovp = atoi(getCmdOption(argv, argv+argc, "-ovp"));
    if (ovp < 0) ovp = 1;
  } else ovp = 1;
  if (getCmdOption(argv, argv+argc, "-cut4d")){
    cut4d = atoi(getCmdOption(argv, argv+argc, "-cut4d"));
    if (cut4d < 0) cut4d = 1;
  } else cut4d = 1;

#ifdef USE_MIC
  int twomics, mic_id=0;
  if (getCmdOption(argv, argv+argc, "-twomics")){
    twomics = atoi(getCmdOption(argv, argv+argc, "-twomics"));
  } else twomics = 1;
  if(twomics)
	mic_id = myRank%2;
#endif

  if (myRank == 0) {
    printf("benchmarking topology-aware pdgemms.\n");
    printf("-n=" PRId64 " -seed = %d -c_rep=%d -ovp=%d\n",n,seed,c_rep,ovp);
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

  GLOBAL_BARRIER(cdt_glb);
  //if (c_rep == 1)
  //  ctb_unit(n, myRank, numPes, seed, cdt_glb);
  //GLOBAL_BARRIER(cdt_glb);
#ifdef	USE_MIC
  d25_unit(n, myRank, numPes, c_rep, seed, ovp, mic_id, cdt_glb);
#else
  d25_unit(n, myRank, numPes, c_rep, seed, ovp, cdt_glb);
#endif
  GLOBAL_BARRIER(cdt_glb);
  //dcn_unit(n, myRank, numPes, cut4d, seed, ovp, cdt_glb);
  //GLOBAL_BARRIER(cdt_glb);
  COMM_EXIT;
  return 0;
}

