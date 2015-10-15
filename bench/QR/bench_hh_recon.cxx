/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <string>
#include <ios>
#include <fstream>
#include <iostream>

#include "CANDMC.h"

using namespace std;

/**
 * \brief Benchmark TSQR and HH reconstruction 
 *
 * \param[in] m number of rows in A
 * \param[in] b number of columns in A
 * \param[in] niter number of iterations
 * \param[in] myRank rank in communicator column
 * \param[in] numPes number of processes in column
 * \param[in] req_id request id to use for send/recv
 * \param[in] comm MPI communicator for column
 **/
void hh_recon_bench(int64_t const m,
                    int64_t const b, 
                    int64_t const niter, 
                    int64_t const myRank, 
                    int64_t const numPes, 
                    int64_t const req_id, 
                    CommData_t  cdt){
  if (myRank == 0)  
    printf("benchmarking parallel TSQR with YT reconstruction...\n");
  double *A;
  double time;
  int64_t i,mb,iter;

  int64_t seed_offset = 99900;
  assert(m%numPes == 0);
  mb = m / numPes;

  assert(0==(posix_memalign((void**)&A,
          ALIGN_BYTES,
          mb*b*sizeof(double))));
  double * W;
  assert(0==(posix_memalign((void**)&W,
          ALIGN_BYTES,
          b*b*sizeof(double))));
  srand48(seed_offset);

  time = MPI_Wtime();
  for (iter=0; iter<niter; iter++){
    for (i=0; i<mb*b; i++){
      A[i] = drand48();
    }
    hh_recon_qr(A,mb,m,b,W,myRank,numPes,0,req_id,cdt);
  }
  MPI_Barrier(cdt.cm);
  time = MPI_Wtime()-time;

  if (myRank == 0)
    printf("TSQR with Householder reconstruction on a " PRId64 "-by-" PRId64 " matrix took %lf seconds/iteration\n",
            m, b, time/niter);

  free(A);
}

int main(int argc, char **argv) {
  int myRank, numPes;
  int64_t m, b, niter;

  CommData_t cdt_glb;
  INIT_COMM(numPes, myRank, 1, cdt_glb);

  if (argc == 2) {
    printf("Usage: mpirun -np <num procs> ./exe <number of rows> <number of columns> <number of iterations>\n");
    ABORT;
  }
  niter = 10;

  if (argc == 1) {
    b = 17;
    m = 39*numPes;
  }
  if (argc >= 3) {
    m = atoi(argv[1]);
    b = atoi(argv[2]);
    assert(m > 0);
    assert(b > 0);
    assert(m % numPes == 0);
    assert(m / numPes >= b);
  }
  if (argc >= 4) 
    niter = atoi(argv[3]);
  TAU_PROFILE_TIMER(timer, "main", "int (int, char**)", TAU_USER);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_SET_NODE(myRank);
  TAU_PROFILE_SET_CONTEXT(0);


  hh_recon_bench(m, b, niter, myRank, numPes, 0, cdt_glb);
  TAU_PROFILE_STOP(timer);

  COMM_EXIT;
  return 0;
}
