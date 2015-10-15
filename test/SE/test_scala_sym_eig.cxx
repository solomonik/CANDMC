/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <assert.h>
#include "CANDMC.h"
#include "../../alg/shared/util.h"

static
char* getopt(char ** begin, char ** end, const std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

int main(int argc, char **argv) {
  int myRank, numPes, n, b, pr, pc, ipr, ipc, i, j, loc_off, m, nz;
  double * loc_A, * full_A, * scala_EL, * full_EL, * loc_EC, * full_EC;
  double * scala_D, * scala_E, * full_D, * full_E, * work, * gap;
  double * scala_T, * full_T;
  int * iwork, * iclustr, * ifail;
  int icontxt, info, iam, inprocs, lwork, liwork;
  char cC = 'C';
  int desc_A[9], desc_EC[9];
  double time;

  CommData_t cdt_glb;
  CommData_t cdt_row, cdt_col;
  INIT_COMM(numPes, myRank, 1, cdt_glb);
  CommData_t cdt_diag;

  if (myRank == 0)
    printf("Usage: %s -n 'matrix dimension' -b 'distribution blocking factor' -pr 'number of processor rows in processor grid'\n", argv[0]);

  if (     getopt(argv, argv+argc, "-pr") &&
      atoi(getopt(argv, argv+argc, "-pr")) > 0 ){
    pr = atoi(getopt(argv, argv+argc, "-pr"));
  } else {
    pr = sqrt(numPes);
    while (numPes%pr!=0) pr++;
  }
  if (     getopt(argv, argv+argc, "-b") &&
      atoi(getopt(argv, argv+argc, "-b")) > 0 )
    b = atoi(getopt(argv, argv+argc, "-b"));
  else 
    b = 4;
  if (     getopt(argv, argv+argc, "-n") &&
      atoi(getopt(argv, argv+argc, "-n")) > 0 )
    n = atoi(getopt(argv, argv+argc, "-n"));
  else 
    n = 2*b*pr;

  if (myRank == 0)
    printf("Executed as '%s -n %d -b %d -pr %d'\n", 
            argv[0], n, b, pr);


  if (numPes % pr != 0) {
    if (myRank == 0){
      printf("%d mod %d != 0 Number of processor grid ", numPes, pr);
      printf("rows must divide into number of processors\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (n % pr != 0) {
    if (myRank == 0){
      printf("%d mod %d != 0 Number of processor grid ", n, pr);
      printf("rows must divide into the matrix dimension\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  pc = numPes / pr;
  if (numPes % pr != 0) {
    if (myRank == 0){
      printf("%d mod %d != 0 Number of processor grid ", n, pc);
      printf("columns must divide into the matrix dimension\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  ipc = myRank / pr;
  ipr = myRank % pr;
  
  
  if (myRank == 0){ 
    printf("Testing ScaLAPACK symmetric eigensolve of ");
    printf("%d-by-%d matrix with block size %d\n",n,n,b);
    printf("Using %d processors in %d-by-%d grid.\n", numPes, pr, pc);
  }

  loc_A    = (double*)malloc(n*n*sizeof(double)/numPes);
  loc_EC   = (double*)malloc(n*n*sizeof(double)/numPes);
  scala_EL = (double*)malloc(n*sizeof(double));
  scala_D  = (double*)malloc(n*sizeof(double));
  scala_E  = (double*)malloc(n*sizeof(double));
  scala_T  = (double*)malloc(n*sizeof(double));
  full_A  = (double*)malloc(n*n*sizeof(double));
  full_EC = (double*)malloc(n*n*sizeof(double));
  full_EL = (double*)malloc(n*sizeof(double));
  full_D  = (double*)malloc(n*sizeof(double));
  full_E  = (double*)malloc(n*sizeof(double));
  full_T  = (double*)malloc(n*sizeof(double));
  lwork       = MAX(5*n*n/numPes,30*n);
  work        = (double*)malloc(lwork*sizeof(double));
  liwork      = 6*MAX(5*n, MAX(4, pr*pc+1));
  iwork       = (int*)malloc(liwork*sizeof(int));
  ifail       = (int*)malloc(n*sizeof(int));
  iclustr     = (int*)malloc(2*pr*pc*sizeof(int));
  gap         = (double*)malloc(pr*pc*sizeof(double));

  srand48(666);

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

  if (myRank == 0) printf("Testing of scalapack pdlatrd\n");
  init_dist_sym_matrix(n,ipr,pr,ipc,pc,b,full_A,loc_A); 
  cdlatrd( 'L', n, b, full_A, n, full_E, full_T, full_EC, n);
  cpdlatrd('L', n, b, loc_A, 1, 1, desc_A, scala_D, scala_E, scala_T, 
           loc_EC, 1, 1, desc_EC, work);
  loc_off = 0;
  for (i=0; i<b-1; i++){
    j=i+1;
    if ((i/b)%pc == ipc && (j/b)%pr == ipr){
      if (fabs((full_E[i] - scala_E[loc_off])/full_E[i]) > 1.E-3 &&
          fabs((full_E[i] + scala_E[loc_off])/full_E[i]) > 1.E-3)
        printf("incorrect subdiagonal %d scalapack computed %E, lapack computed %E\n",
                i, loc_EC[loc_off], full_EC[i]);
      loc_off++;
    }
  }
  if (myRank == 0) printf("Verification of scalapack panel to tridiagonal completed\n");
  
  if (myRank == 0) printf("Testing of scalapack pdsytrd\n");
  init_dist_sym_matrix(n,ipr,pr,ipc,pc,b,full_A,loc_A); 
  cdsytrd( 'L', n, full_A, n, full_D, full_E, full_T, work, lwork, &info);
  cpdsytrd('L', n, loc_A, 1, 1, desc_A, scala_D, scala_E, scala_T, 
           work, lwork, &info);
  loc_off = 0;
  for (i=0; i<n-1; i++){
    j=i+1;
    if ((i/b)%pc == ipc /*&& (i/b)%pr == ipr*/){
      if (fabs((full_E[i] - scala_E[loc_off])/full_E[i]) > 1.E-3 &&
          fabs((full_E[i] + scala_E[loc_off])/full_E[i]) > 1.E-3)
        printf("[%d] incorrect subdiagonal %d scalapack computed %E, lapack computed %E\n",
                myRank, i, scala_E[loc_off], full_E[i]);
      loc_off++;
    }
  }
  if (myRank == 0) printf("Verification of scalapack symmetric to tridiagonal completed\n");
 
  if (myRank == 0) printf("Testing of scalapack pdsyevx eigenvalues\n");
  init_dist_sym_matrix(n,ipr,pr,ipc,pc,b,full_A,loc_A); 
  cdsyevx( 'N', 'A', 'L', n, full_A, n, 0.0, 0.0, 0, 0, 0.0,
           &m, full_EL, NULL, n, work, lwork, iwork,
           NULL, &info);

  cpdsyevx('N', 'A', 'L', n, loc_A, 1, 1, desc_A, 0.0, 0.0, 0, 0, 0.0,
           &m, 0, scala_EL, 0.0, NULL, 0, 0, NULL, work, lwork, iwork,
           liwork, NULL, NULL, NULL, &info);

  if (myRank == 0){
    for (i=0; i<n; i++){
      if (fabs(full_EL[i] - scala_EL[i]) > 1.E-6){
        printf("incorrect eigenvalue %d, scalapack computed %E, lapack computed %E\n",
                i, scala_EL[i], full_EL[i]);
      }
    }
    printf("Verification of eigenvalues completed\n");
  }
  
  if (myRank == 0) printf("Testing of scalapack pdsyevx eigenvectors\n");
  init_dist_sym_matrix(n,ipr,pr,ipc,pc,b,full_A,loc_A); 
  cdsyevx( 'V', 'A', 'L', n, full_A, n, 0.0, 0.0, 0, 0, 0.0,
           &m, full_EL, full_EC, n, work, lwork, iwork,
           ifail, &info);

  cpdsyevx('V', 'A', 'L', n, loc_A, 1, 1, desc_A, 0.0, 0.0, 0, 0, 0.0,
           &m, &nz, scala_EL, 0.0, loc_EC, 1, 1, desc_EC, work, lwork, iwork,
           liwork, ifail, iclustr, gap, &info);

  loc_off = 0;
  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      if ((i/b)%pc == ipc && (j/b)%pr == ipr){
        if (fabs((full_EC[i*n+j] - loc_EC[loc_off])/full_EC[i*n+j]) > 1.E-3 &&
            fabs((full_EC[i*n+j] + loc_EC[loc_off])/full_EC[i*n+j]) > 1.E-3)
          printf("incorrect eigenvector %d, element %d scalapack computed %E, lapack computed %E\n",
                  i, j, loc_EC[loc_off], full_EC[i*n+j]);
        loc_off++;
      }
    }
  }
  if (myRank == 0) printf("Verification of scalapack eigenvectors completed\n");
  MPI_Finalize();
  return 0;
} /* end function main */


