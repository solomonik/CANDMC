/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <assert.h>
#include "CANDMC.h"
#include "../../alg/shared/util.h"

#ifndef FTN_UNDERSCORE
#define FTN_UNDERSCORE 1
#endif

#define NUM_ITER 1
//subroutine tridiag_real(na, a, lda, nblk, mpi_comm_rows, mpi_comm_cols, d, e, tau)
extern "C"
#if FTN_UNDERSCORE
void elpa1_mp_tridiag_real_(int const*, double const*, int const*, int const*, MPI_Comm const*, MPI_Comm const*, double*, double*, double*);
#else
void __elpa1_NMOD_tridiag_real(int const*, double const*, int const*, int const*, MPI_Comm const*, MPI_Comm const*, double*, double*, double*);
#endif

void elpa1_mp_tridiag_real(int             na, 
                           double const *  A, 
                           int             lda,
                           int             nblk,
                           MPI_Comm        crow, 
                           MPI_Comm        ccol, 
                           double *        D,
                           double *        E,
                           double *        tau){
#if FTN_UNDERSCORE
  elpa1_mp_tridiag_real_(&na, A, &lda, &nblk, &crow, &ccol, D, E, tau);
#else
  __elpa1_NMOD_tridiag_real(&na, A, &lda, &nblk, &crow, &ccol, D, E, tau);
#endif
}


//subroutine bandred_real(na, a, lda, nblk, nbw, mpi_comm_rows, mpi_comm_cols, tmat)
extern "C"
#if FTN_UNDERSCORE
void elpa2_mp_bandred_real_(int const*, double const*, int const *, int const *, int const*, int const*, int const*, MPI_Comm const*, MPI_Comm const*, double*, int*, int*, int*);
#else
void __elpa2_NMOD_bandred_real(int const*, double const*, int const*, int const*, int const*, MPI_Comm const*, MPI_Comm const*, double*);
#endif

void elpa2_mp_bandred_real(int             na, 
                           double const *  A, 
                           int             lda,
                           int             nblk,
                           int             nbw,
                           MPI_Comm        crow, 
                           MPI_Comm        ccol, 
                           int             useQR){
  int nblks = (na-1)/nbw+1;
  double * tmat = (double*)malloc(nbw*nbw*nblks*sizeof(double));
  int wantDebug=0;
  int success;
#if FTN_UNDERSCORE
  elpa2_mp_bandred_real_(&na, A, &lda, &nblk, &nbw, &lda, &nblks, &crow, &ccol, tmat, &wantDebug, &success, &useQR);
#else
  __elpa2_mp_bandred_real_(&na, A, &lda, &nblk, &nbw, &lda, &nblks, &crow, &ccol, tmat, &wantDebug, &success, &useQR);
#endif
  free(tmat);
}


//subroutine tridiag_band_real(na, nb, nblk, a, lda, d, e, mpi_comm_rows, mpi_comm_cols, mpi_comm)
extern "C"
#if FTN_UNDERSCORE
void elpa2_mp_tridiag_band_real_(int const*, int const*, int const*, double const*, int const *, double*, double*, int const*, MPI_Comm const*, MPI_Comm const*, MPI_Comm const*);
#else
void __elpa2_NMOD_tridiag_band_real(int const*, int const*, int const*, double const*, int const*, double*, double*, int const*, MPI_Comm const*, MPI_Comm const*, MPI_Comm const*);
#endif

void elpa2_mp_tridiag_band_real(int            na, 
                                int            nbw,
                                int            nblk,
                                double const * A, 
                                int            lda,
                                double *       D,
                                double *       E,
                                MPI_Comm       crow, 
                                MPI_Comm       ccol, 
                                MPI_Comm       cworld){
#if FTN_UNDERSCORE
  elpa2_mp_tridiag_band_real_(&na, &nbw, &nblk, A, &lda, D, E, &lda, &crow, &ccol, &cworld);
#else
  __elpa2_NMOD_tridiag_band_real(&na, &nbw, &nblk, A, &lda, D, E, &lda, &crow, &ccol, &cworld);
#endif
}


static char* getopt(char ** begin, char ** end, const std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

int main(int argc, char **argv) {
  int myRank, numPes, n, b, bw, niter, pr, pc, ipr, ipc, i, j, loc_off, lda_A, iter, rb, useQR;
  double * A, * D, * E, * T;
  volatile double time;

  //int prov;
  //MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &prov);
  //if (myRank == 0)
  //  printf("Thread support is %d needed is %d\n", prov, MPI_THREAD_MULTIPLE);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  if (myRank == 0){
    printf("Usage: %s -n 'matrix dimension' -bw 'bandwidth' -b 'distribution blocking factor' -pr 'number of processor rows in processor grid' -niter 'number of iterations' -rb '1 if reduce band to tri 0 otherwise'\n", argv[0]);
  }

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
  /*if (niter != 1){
    if (myRank == 0){
      printf("Number of iterations must be or ELPA band to tridiag seems to crash on the second, overriding setting\n");
    }
    niter = 1;
  }*/
  if (     getopt(argv, argv+argc, "-bw") &&
      atoi(getopt(argv, argv+argc, "-bw")) > 0 )
    bw = atoi(getopt(argv, argv+argc, "-bw"));
  else 
    bw = 32;
  if (     getopt(argv, argv+argc, "-rb") &&
      atoi(getopt(argv, argv+argc, "-rb")) > 0 )
    rb = atoi(getopt(argv, argv+argc, "-rb"));
  else 
    rb = 1;
  if (     getopt(argv, argv+argc, "-useQR") &&
      atoi(getopt(argv, argv+argc, "-useQR")) > 0 )
    useQR = atoi(getopt(argv, argv+argc, "-useQR"));
  else 
    useQR = 1;
  if (     getopt(argv, argv+argc, "-b") &&
      atoi(getopt(argv, argv+argc, "-b")) > 0 )
    b = atoi(getopt(argv, argv+argc, "-b"));
  else 
    b = 8;
  if (     getopt(argv, argv+argc, "-n") &&
      atoi(getopt(argv, argv+argc, "-n")) > 0 )
    n = atoi(getopt(argv, argv+argc, "-n"));
  else 
    n = 4*bw*pr;

  if (myRank == 0)
    printf("Executed as '%s -n %d -bw = %d -b %d -pr %d -niter %d -rb %d -useQR %d'\n", 
            argv[0], n, bw, b, pr, niter, rb, useQR);

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
    printf("Benchmarking ELPA symmetric eigensolve of ");
    printf("%d-by-%d matrix with block size %d ",n,n,b);
    printf("using %d processors in %d-by-%d grid.\n", numPes, pr, pc);
  }

  lda_A = n/pr;
  A = (double*)malloc(n*n*sizeof(double)/numPes);
  D = (double*)malloc(3*n*sizeof(double));
  E = (double*)malloc(3*n*sizeof(double));
  if (bw >= n)
    T = (double*)malloc(3*n*sizeof(double));
  else
    T = (double*)malloc(3*n*sizeof(double)+bw*n*sizeof(double));

  MPI_Comm comm_rows, comm_cols;
  MPI_Comm_split(MPI_COMM_WORLD, ipc, ipr, &comm_rows); 
  MPI_Comm_split(MPI_COMM_WORLD, ipr, ipc, &comm_cols); 

  srand48(66*myRank);

  if (bw>= n){
    time = MPI_Wtime();
    for (iter=0; iter<niter; iter++){
      for (i=0; i<n*n/numPes; i++){
        A[i] = drand48();
      }
      elpa1_mp_tridiag_real(n, A, n/pr, b, comm_rows, comm_cols, D, E, T);
    }
    time = MPI_Wtime()-time;
    
    if(myRank == 0){
      printf("Completed %u iterations of ELPA1 tridiagonal reduction\n", iter);
      printf("ELPA1 full to tridiagonal n=%d b=%d: sec/iterations: %f ", 
              n,b, time/niter);
      printf("Gigaflops: %f\n", ((4./3.)*n*b*b)/(time/niter)*1E-9);
    }
  } else {

//  MPI_Comm_split(MPI_COMM_WORLD, ipc, ipr, &comm_rows); 
  ///MPI_Comm_split(MPI_COMM_WORLD, ipr, ipc, &comm_cols); 
    double time_br = MPI_Wtime();
    for (iter=0; iter<niter; iter++){
      for (i=0; i<n*n/numPes; i++){
        A[i] = drand48();
      }
      elpa2_mp_bandred_real(n, A, lda_A, b, bw, comm_rows, comm_cols, useQR);
    }
    
    time_br = MPI_Wtime()-time_br;
    if(myRank == 0){
      printf("Completed %u iterations of ELPA2 full to band reduction\n", iter);
      printf("ELPA2 full to banded n=%d bw=%d b=%d: sec/iterations: %f ", 
              n,bw,b, time_br/niter);
      printf("Upscaled band reduction gigaflops: %f\n", ((4./3.)*n*b*b)/(time_br/niter)*1E-9);
    }

    if (rb){
      double time_bt = MPI_Wtime();
      for (iter=0; iter<niter; iter++){
        for (i=0; i<n*n/numPes; i++){
          A[i] = drand48();
        }
        elpa2_mp_tridiag_band_real(n, bw, b, A, lda_A, D, E, comm_rows, comm_cols, MPI_COMM_WORLD);
      }
      time_bt = MPI_Wtime()-time_bt;
      
      if(myRank == 0){
        printf("Completed %u iterations of ELPA2 banded to tridiagonal reduction\n", iter);
        printf("ELPA2 banded to tridiagonal n=%d bw=%d b=%d: sec/iterations: %f \n", 
                n,bw,b, time_bt/niter);
      } 
      if(myRank == 0){
        printf("Completed %u iterations of ELPA2 banded to tridiagonal reduction\n", iter);
        printf("ELPA2 total full to tridiagonal n=%d bw=%d b=%d: sec/iterations: %f \n", 
                n,bw,b, (time_bt+time_br)/niter);
      } 
    }

  }
  MPI_Finalize();
  return 0;
} /* end function main */


