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

static
void test_spc(int const rank,
              int const ndim,
              int const bidir,
              int const kary,
              int const seed,
              int const n,
              int const m,
              int const k,
              double const alpha,
              double const beta){
  int i, j, s, khalf, px, py, tr, pass, allpass;
  double * A, * B, * C;
  double * full_A, * full_B, * full_C;

  assert(ndim >= 2 && ndim%2 == 0);
  assert(k%ndim == 0);

  khalf = 1;
  for (i=0; i<ndim/2; i++){
    khalf *= kary;
  }

  full_A = (double*)malloc(m*khalf*k*khalf*sizeof(double));
  full_B = (double*)malloc(k*khalf*n*khalf*sizeof(double));
  full_C = (double*)malloc(m*khalf*n*khalf*sizeof(double));
  
  A = (double*)malloc(m*k*sizeof(double));
  B = (double*)malloc(k*n*sizeof(double));
  C = (double*)malloc(m*n*sizeof(double));
  
  srand48(seed);

  for (i=0; i<m*khalf*k*khalf; i++){
    full_A[i] = drand48();
  }
  for (i=0; i<k*khalf*n*khalf; i++){
    full_B[i] = drand48();
  }
  for (i=0; i<m*khalf*n*khalf; i++){
    full_C[i] = drand48();
  }

  px=0, py=0, s=1;
  tr=rank;
  for (i=0; i<ndim/2; i++){
    px += (tr%kary)*s;
    tr  = tr/kary;
    py += (tr%kary)*s;
    tr  = tr/kary;
    s = s*kary;
  }
  
  for (i=0; i<k; i++){
    for (j=0; j<m; j++){
      A[i*m+j] = full_A[(px*k+i)*khalf*m + (py*m+j)];
    }
  }
  for (i=0; i<n; i++){
    for (j=0; j<k; j++){
      B[i*k+j] = full_B[(px*n+i)*khalf*k + (py*k+j)];
    }
  }
  for (i=0; i<n; i++){
    for (j=0; j<m; j++){
      C[i*m+j] = full_C[(px*n+i)*khalf*m + (py*m+j)];
    }
  }

  cdgemm('N','N',khalf*m,khalf*n,khalf*k,alpha,full_A,khalf*m,full_B,
        khalf*k,beta,full_C,khalf*m);

  if (bidir)
    kput_cannon(rank, kary, ndim, MPI_COMM_WORLD, n, m, k, 
                'N', alpha, A,
                'N', beta,  B, C);
  else
    kuni_cannon(rank, kary, ndim, MPI_COMM_WORLD, n, m, k, 
                'N', alpha, A,
                'N', beta,  B, C);
 
  pass = 1; 
  for (i=0; i<n; i++){
    for (j=0; j<m; j++){
      if (fabs(C[i*m+j] - full_C[(px*n+i)*khalf*m + (py*m+j)])>1.E-6){
        pass = 0;
        printf("[%d] C[%d,%d] = %lf != %lf\n",rank,(py*m+j),(px*n+i),
                full_C[(px*n+i)*khalf*m + (py*m+j)],
                C[i*m+j]);
      }
    }
  }
  MPI_Reduce(&pass, &allpass, 1, MPI_INT, MPI_BAND, 0, MPI_COMM_WORLD);
  if (rank == 0){
    if (allpass == 1) printf("Test passed.\n");
    else printf("Test FAILED.\n");
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
  int seed, kary, ndim, n, m, k, rank, bidir, numPes;
  double alpha, beta;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);

  if (getCmdOption(input_str, input_str+in_num, "-bidir")){
    bidir = atoi(getCmdOption(input_str, input_str+in_num, "-bidir"));
    if (bidir < 0) bidir = 1;
  } else bidir = 1;
  if (getCmdOption(input_str, input_str+in_num, "-ndim")){
    ndim = atoi(getCmdOption(input_str, input_str+in_num, "-ndim"));
    if (ndim < 2) ndim = 2;
  } else ndim = 2;
  if (getCmdOption(input_str, input_str+in_num, "-kary")){
    kary = atoi(getCmdOption(input_str, input_str+in_num, "-kary"));
    if (kary < 0) kary = 1;
  } else {
    kary = 1;
    for (;pow(kary,ndim)<numPes;kary++){}
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
  if (p != numPes){
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
    printf("Testing multiply of block size %d-by-%d A and %d-by-%d B\n", 
           m, k, k, n);
  }
  
  test_spc(rank, ndim, bidir, kary, seed, n, m, k, alpha, beta);

  MPI_Finalize();
  return 0;
}
