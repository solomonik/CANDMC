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
  int myRank, numPes, niter, pr, pc, m, ipr, ipc;
  int64_t n, b, i;
  double * loc_A, * scala_EL, * loc_EC;
  double * scala_D, * scala_E, * work, * gap;
  double * scala_T;
  int * iwork, * iclustr, * ifail;
  int icontxt, info, iam, inprocs, iter, lwork;
  char cC = 'C';
  int desc_A[9], desc_EC[9];
  volatile double time;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);


  if (myRank == 0)
    printf("Usage: %s -n 'matrix dimension' -b 'distribution blocking factor' -pr 'number of processor rows in processor grid' -niter 'number of iterations'\n", argv[0]);

  if (     getopt(argv, argv+argc, "-pr") &&
      atoi(getopt(argv, argv+argc, "-pr")) > 0 ){
    pr = atoi(getopt(argv, argv+argc, "-pr"));
  } else {
    pr = sqrt(numPes);
    while (numPes%pr!=0) pr++;
  }
  if (     getopt(argv, argv+argc, "-niter") &&
      atoi(getopt(argv, argv+argc, "-niter")) > 0 )
    niter = atoi(getopt(argv, argv+argc, "-niter"));
  else 
    niter = NUM_ITER;
  if (     getopt(argv, argv+argc, "-b") &&
      atoi(getopt(argv, argv+argc, "-b")) > 0 )
    b = atoi(getopt(argv, argv+argc, "-b"));
  else 
    b = 32;
  if (     getopt(argv, argv+argc, "-n") &&
      atoi(getopt(argv, argv+argc, "-n")) > 0 )
    n = atoi(getopt(argv, argv+argc, "-n"));
  else 
    n = 4*b*pr;

  if (myRank == 0)
    printf("Executed as '%s -n %ld -b %ld -pr %d -niter %d'\n", 
            argv[0], n, b, pr, niter);

  if (numPes % pr != 0) {
    if (myRank == 0){
      printf("%d mod %d != 0 Number of processor grid ", numPes, pr);
      printf("rows must divide into number of processors\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (n % pr != 0) {
    if (myRank == 0){
      printf("%ld mod %d != 0 Number of processor grid ", n, pr);
      printf("rows must divide into the matrix dimension\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  pc = numPes / pr;
  if (numPes % pr != 0) {
    if (myRank == 0){
      printf("%ld mod %d != 0 Number of processor grid ", n, pc);
      printf("columns must divide into the matrix dimension\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  ipc = myRank / pr;
  ipr = myRank % pr;
  
  
  if (myRank == 0){ 
    printf("Benchmarking ScaLAPACK symmetric eigensolve of ");
    printf("%ld-by-%ld matrix with block size %ld\n",n,n,b);
    printf("Using %d processors in %d-by-%d grid.\n", numPes, pr, pc);
  }

  loc_A    = (double*)malloc(n*n*sizeof(double)/numPes);
  loc_EC   = (double*)malloc(n*n*sizeof(double)/numPes);
  scala_EL = (double*)malloc(n*sizeof(double));
  scala_D  = (double*)malloc(n*sizeof(double));
  scala_E  = (double*)malloc(n*sizeof(double));
  scala_T  = (double*)malloc(n*sizeof(double));
  lwork       = MAX(5*n*n/numPes,30*n);
  work        = (double*)malloc(lwork*sizeof(double));
  iwork       = (int*)malloc(30*n*sizeof(int));
  ifail       = (int*)malloc(n*sizeof(int));
  iclustr     = (int*)malloc(2*pr*pc*sizeof(int));
  gap         = (double*)malloc(pr*pc*sizeof(double));

  srand48(666*myRank);

  Cblacs_pinfo(&iam,&inprocs);
  Cblacs_get(-1, 0, &icontxt);
  Cblacs_gridinit(&icontxt, &cC, pr, pc);
  cdescinit(desc_A, n, n,
		        b, b,
		        0, 0,
		        icontxt, n/pr, 
				    &info);
  cdescinit(desc_EC, n, n,
		        b, b,
		        0, 0,
		        icontxt, n/pr, 
				    &info);
  assert(info==0);

  time = MPI_Wtime();
  for (iter=0; iter<niter; iter++){
    for (i=0; i<n*n/numPes; i++){
      loc_A[i] = drand48();
    }
    for (i=0; i<n/b-1; i++){
      cpdlatrd('L', n-i*b, b, loc_A, i*b+1, i*b+1, desc_A, scala_D, scala_E, scala_T, 
               loc_EC, i*b+1, 1, desc_EC, work);
    }
  }
  time = MPI_Wtime()-time;
  
  if(myRank == 0){
    printf("Completed %u iterations (panel to tridiagonal)\n", iter);
    printf("(panel to tridiagonal) n = %ld b=%ld: sec/iterations: %lf ", 
            n,b, time/niter);
    printf("Gigaflops: %lf\n", ((4./3.)*n*b*b)/(time/niter)*1E-9);
  }
  
  time = MPI_Wtime();
  for (iter=0; iter<niter; iter++){
    for (i=0; i<n*n/numPes; i++){
      loc_A[i] = drand48();
    }
    cpdsytrd('L', n, loc_A, 1, 1, desc_A, scala_D, scala_E, scala_T, 
             work, lwork, &info);
  }
  time = MPI_Wtime()-time;
  
  if(myRank == 0){
    printf("Completed %u iterations (symmetric to tridiagonal)\n", iter);
    printf("(symmetric to tridiagonal) n = %ld: sec/iteration: %lf ", n, time/niter);
    printf("Gigaflops: %lf\n", ((4./3.)*n*n*n)/(time/niter)*1E-9);
  }

  time = MPI_Wtime();

  for (iter=0; iter<niter; iter++){
    for (i=0; i<n*n/numPes; i++){
      loc_A[i] = drand48();
    }
    cpdsyevx('N', 'A', 'L', n, loc_A, 1, 1, desc_A, 0.0, 0.0, 0, 0, 0.0,
             &m, 0, scala_EL, 0.0, NULL, 0, 0, NULL, work, lwork, iwork,
             30*n, NULL, NULL, NULL, &info);
    
  }
  time = MPI_Wtime()-time;
  
  if(myRank == 0){
    printf("Completed %u iterations (eigenvalues only)\n", iter);
    printf("(eigenvalues only) n = %ld: sec/iteration: %lf ", n, time/niter);
    printf("Gigaflops: %lf\n", ((4./3.)*n*n*n)/(time/niter)*1E-9);
  }

//  time = MPI_Wtime();
//
//  for (iter=0; iter<niter; iter++){
//    for (i=0; i<n*n/numPes; i++){
//      loc_A[i] = drand48();
//    }
//    cpdsyevx('V', 'A', 'L', n, loc_A, 1, 1, desc_A, 0.0, 0.0, 0, 0, 0.0,
//             &m, &nz, scala_EL, 0.0, loc_EC, 1, 1, desc_EC, work, lwork, iwork,
//             30*n, ifail, iclustr, gap, &info);
//    
//  }
//  time = MPI_Wtime()-time;
//  
//  if(myRank == 0){
//    printf("Completed %u iterations (with eigenvectors)\n", iter);
//    printf("(with eigenvectors) n = %d: sec/iteration: %f ", n, time/niter);
//    printf("Gigaflops: %f\n", ((4./3.)*n*n*n)/(time/niter)*1E-9);
//  }


  MPI_Finalize();
  return 0;
} /* end function main */


