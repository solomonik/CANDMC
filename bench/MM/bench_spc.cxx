/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */
/* Copyright (c) 2011, Edgar Solomonik>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following 
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL EDGAR SOLOMONIK BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY 
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
 * SUCH DAMAGE. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <algorithm>
#include <assert.h>
#include "CANDMC.h"

#define ALIGN   16

static
void bench_spc(int const        rank,
               int const        np,
               int const        ndim,
               int const        bidir,
               int const        kary,
               int const        seed,
               int const        nwarm,
               int const        niter,
               int const        n,
               int const        m,
               int const        k,
               double const     alpha,
               double const     beta){
  int i, j;
  double st, end;
  double * A, * B, * C;

  assert(ndim >= 2 && ndim%2 == 0);
  assert(k%ndim == 0);

  posix_memalign((void**)&A, ALIGN, m*k*sizeof(double));
  posix_memalign((void**)&B, ALIGN, n*k*sizeof(double));
  posix_memalign((void**)&C, ALIGN, m*n*sizeof(double));
  
  srand48(seed);

  for (i=0; i<m*k; i++){
    A[i] = drand48();
  }
  for (i=0; i<k*n; i++){
    B[i] = drand48();
  }
  for (i=0; i<m*n; i++){
    C[i] = drand48();
  }

  for (i=0; i<nwarm; i++){
    if (bidir)
      kput_cannon(rank, kary, ndim, MPI_COMM_WORLD, n, m, k, 
                  'N', alpha, A,
                  'T', beta,  B, C);
    else
      kuni_cannon(rank, kary, ndim, MPI_COMM_WORLD, n, m, k, 
                  'N', alpha, A,
                  'T', beta,  B, C);
  }

  if (rank == 0)
    printf("Performed %d warm-up iterations\n", nwarm);
  
  MPI_Barrier(MPI_COMM_WORLD);

 
  st = MPI_Wtime(); 
  for (i=0; i<niter; i++){
    if (bidir)
      kput_cannon(rank, kary, ndim, MPI_COMM_WORLD, n, m, k, 
                  'N', alpha, A,
                  'T', beta,  B, C);
    else
      kuni_cannon(rank, kary, ndim, MPI_COMM_WORLD, n, m, k, 
                  'N', alpha, A,
                  'T', beta,  B, C);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime(); 

  if (rank == 0){
    printf("Performed %d timed iterations\n",niter);
    printf("Avg time per iteration: %lf\n",(end-st)/niter);
    printf("Achieved flop rate: %lf GF\n", (2.*n*m*k*np*sqrt(np)*1.E-9)/((end-st)/niter));
  }
}

char* getCmdOption(char ** begin, 
                   char ** end, 
                   const std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}


int main(int argc, char ** argv){
  int seed, kary, ndim, n, m, k, rank, np, nwarm, niter, bidir;
  double alpha, beta;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);


  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 10;
  } else niter = 10;
  if (getCmdOption(input_str, input_str+in_num, "-nwarm")){
    nwarm = atoi(getCmdOption(input_str, input_str+in_num, "-nwarm"));
    if (nwarm < 0) nwarm = 0;
  } else nwarm = 1;
  if (getCmdOption(input_str, input_str+in_num, "-bidir")){
    bidir = atoi(getCmdOption(input_str, input_str+in_num, "-bidir"));
    if (bidir < 0) bidir = 1;
  } else bidir = 1;
  if (getCmdOption(input_str, input_str+in_num, "-ndim")){
    ndim = atoi(getCmdOption(input_str, input_str+in_num, "-ndim"));
    if (ndim < 0) ndim = 2;
  } else ndim = 2;
  if (getCmdOption(input_str, input_str+in_num, "-kary")){
    kary = atoi(getCmdOption(input_str, input_str+in_num, "-kary"));
    if (kary < 0) kary = 1;
  } else {
    kary = 1;
    for (;pow(kary,ndim)<np;kary++){}
  }
  if (getCmdOption(input_str, input_str+in_num, "-seed")){
    seed = atoi(getCmdOption(input_str, input_str+in_num, "-seed"));
    if (seed < 0) seed = 3;
  } else seed = 3;
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n <= 0) n = 64;
  } else n = 64;
  if (getCmdOption(input_str, input_str+in_num, "-m")){
    m = atoi(getCmdOption(input_str, input_str+in_num, "-m"));
    if (m <= 0) m = 64;
  } else m = 64;
  if (getCmdOption(input_str, input_str+in_num, "-k")){
    k = atoi(getCmdOption(input_str, input_str+in_num, "-k"));
    if (k <= 0) k = 64;
  } else k = 64;
  if (getCmdOption(input_str, input_str+in_num, "-alpha")){
    alpha = atof(getCmdOption(input_str, input_str+in_num, "-alpha"));
    if (alpha < 0) alpha = 1.2;
  } else alpha = 1.2;
  if (getCmdOption(input_str, input_str+in_num, "-beta")){
    beta = atof(getCmdOption(input_str, input_str+in_num, "-beta"));
    if (beta < 0) beta = .8;
  } else beta = .8;

  int p = 1;
  for (int l=0; l<ndim; l++){
    p*=kary;
  }
  if (p != np){
    if (rank == 0){
      printf("Need square processor grid for Cannon test, exiting\n");
    }
    return 0;
  }


  if (rank == 0) {
    if (bidir)
      printf("Topology is a bidirectional %d-ary %d-cube\n", kary, ndim);
    else
      printf("Topology is a unidirectional %d-ary %d-cube\n", kary, ndim);
    printf("Benchmarking multiply of block size %d-by-%d A and %d-by-%d B\n", 
           m, k, k, n);
  }

#ifdef TAU
  TAU_PROFILE_TIMER(timer, "main", "int (int, char**)", TAU_USER);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(rank);
  TAU_PROFILE_SET_CONTEXT(0);
#endif
  
  bench_spc(rank, np, ndim, bidir, kary, seed, nwarm, niter, n, m, k, alpha, beta);

  TAU_PROFILE_STOP(timer);
  MPI_Finalize();
  return 0;
}
