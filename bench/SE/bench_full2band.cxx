/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <assert.h>
#include "CANDMC.h"
#include "../../alg/shared/util.h"

#define NUM_ITER 3

static
char* getopt(char ** begin, char ** end, const std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}



int main(int argc, char **argv) {
  int myRank, numPes, n, b, niter, pr, pc, ipr, ipc, i, j, loc_off, m, b_agg, iter;
  double * loc_A;
  double time;

  CommData_t cdt_glb;
  CommData_t cdt_row, cdt_col;
  INIT_COMM(numPes, myRank, 1, cdt_glb);
  CommData_t cdt_diag;


  if (myRank == 0)
    printf("Usage: %s -n 'matrix dimension' -b 'distribution blocking factor' -b_agg 'aggregation blocking factor' -niter 'number of iterations'\n", argv[0]);

  pr = sqrt(numPes);
  if (pr != sqrt(numPes)){
    if (myRank == 0)
      printf("Full to banded benchmark needs square processor grid, terminating...\n");
    return 0;
  }
  assert(numPes%pr == 0);
  if (     getopt(argv, argv+argc, "-niter") &&
      atoi(getopt(argv, argv+argc, "-niter")) > 0 )
    niter = atoi(getopt(argv, argv+argc, "-niter"));
  else 
    niter = NUM_ITER;
  if (     getopt(argv, argv+argc, "-b") &&
      atoi(getopt(argv, argv+argc, "-b")) > 0 )
    b = atoi(getopt(argv, argv+argc, "-b"));
  else 
    b = 16;
  if (     getopt(argv, argv+argc, "-b_agg") &&
      atoi(getopt(argv, argv+argc, "-b_agg")) > 0 )
    b_agg = atoi(getopt(argv, argv+argc, "-b_agg"));
  else 
    b_agg = 32;
  if (     getopt(argv, argv+argc, "-n") &&
      atoi(getopt(argv, argv+argc, "-n")) > 0 )
    n = atoi(getopt(argv, argv+argc, "-n"));
  else 
    n = 8*b*pr;

  if (myRank == 0)
    printf("Executed as '%s -n %d b_agg %d -b %d -niter %d'\n", 
            argv[0], n, b_agg, b, niter);

  if (numPes % pr != 0) {
    if (myRank == 0){
      printf("%d % %d != 0 Number of processor grid ", numPes, pr);
      printf("rows must divide into number of processors\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (n % pr != 0) {
    if (myRank == 0){
      printf("%d % %d != 0 Number of processor grid ", n, pr);
      printf("rows must divide into the matrix dimension\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  pc = numPes / pr;
  if (numPes % pr != 0) {
    if (myRank == 0){
      printf("%d % %d != 0 Number of processor grid ", n, pc);
      printf("columns must divide into the matrix dimension\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  ipc = myRank / pr;
  ipr = myRank % pr;
  
  
  if (myRank == 0){ 
    printf("Benchmarking symmetric eigensolve full to bandwidth %d of ",b_agg);
    printf("%d-by-%d matrix with block size %d\n",n,n,b);
    printf("Using %d processors in %d-by-%d grid.\n", numPes, pr, pc);
  }
  SETUP_SUB_COMM(cdt_glb, cdt_row, 
                 myRank/pr, 
                 myRank%pr, 
                 pc);
  SETUP_SUB_COMM(cdt_glb, cdt_col, 
                 myRank%pr, 
                 myRank/pr, 
                 pr);
  if (ipr == ipc){
    SETUP_SUB_COMM(cdt_glb, cdt_diag, 
                   ipr, 
                   0, 
                   pr);
  } else {
    SETUP_SUB_COMM(cdt_glb, cdt_diag, 
                   myRank, 
                   1, 
                   pr);
  }
  TAU_PROFILE_TIMER(timer, "main", "int (int, char**)", TAU_USER);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_SET_NODE(myRank);
  TAU_PROFILE_SET_CONTEXT(0);

  loc_A    = (double*)malloc(n*n*sizeof(double)/numPes);
  srand48(666*myRank);

  time = MPI_Wtime();

  CTF_Timer_epoch ep1("full2band_with_TSQR"); 
  ep1.begin();
  for (iter=0; iter<niter; iter++){
    //FIXME: nonsymmetric
    for (i=0; i<n*n/numPes; i++){
      loc_A[i] = drand48();
    }
      
    pview pv;
    pv.rrow = 0;
    pv.rcol = 0;
    pv.crow = cdt_row;
    pv.ccol = cdt_col;
    pv.cdiag = cdt_diag;
    pv.cworld = cdt_glb;
    sym_full2band(loc_A, n/pr, n, b_agg, b, &pv);
  }
  time = MPI_Wtime()-time;
  if(myRank == 0){
    printf("Completed %u iterations of full to band with TSQR\n", iter);
    printf("2D CANDMC with TSQR full to band n = %d b_agg = %d b = %d: sec/iteration: %f ", n, b_agg, b, time/niter);
    printf("Gigaflops: %f\n", ((4./3.)*n*n*n)/(time/niter)*1E-9);
  }
  ep1.end();


  srand48(666*myRank);

  time = MPI_Wtime();

#ifdef USE_SCALAPACK

  int icontxt, info, iam, inprocs, lwork;
  char cC = 'C';
  int desc_A[9];
  Cblacs_pinfo(&iam,&inprocs);
  Cblacs_get(-1, 0, &icontxt);
  Cblacs_gridinit(&icontxt, &cC, pr, pc);
  cdescinit(desc_A, n, n,
		        b, b,
		        0, 0,
		        icontxt, n/pr, 
				    &info);

  CTF_Timer_epoch ep2("full2band_with_ScaLAPACK_1D_QR"); 
  ep2.begin();
  for (iter=0; iter<niter; iter++){
    for (i=0; i<n*n/numPes; i++){
      loc_A[i] = drand48();
    }
      
    pview pv;
    pv.rrow = 0;
    pv.rcol = 0;
    pv.crow = cdt_row;
    pv.ccol = cdt_col;
    pv.cdiag = cdt_diag;
    pv.cworld = cdt_glb;
    sym_full2band_scala(loc_A, n/pr, n, b_agg, b, &pv, desc_A, loc_A);
  }
  time = MPI_Wtime()-time;
  if(myRank == 0){
    printf("Completed %u iterations of full to band with 1D ScaLAPACK QR\n", iter);
    printf("2D CANDMC with 1D ScaLAPACK QR full to band n = %d b_agg = %d b = %d: sec/iteration: %f ", n, b_agg, b, time/niter);
    printf("Gigaflops: %f\n", ((4./3.)*n*n*n)/(time/niter)*1E-9);
  }
  ep2.end();
#endif

  TAU_PROFILE_STOP(timer);

  MPI_Finalize();
  return 0;
} /* end function main */


