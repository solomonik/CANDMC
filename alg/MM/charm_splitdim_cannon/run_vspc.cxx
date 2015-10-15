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
#include "vspcannon.decl.h"
#include "vspcannon.h"
#include "spcannon_internal.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_VPblock vpgrid;
/*readonly*/ int kary;
/*readonly*/ int ndim;
/*readonly*/ int bidir;

void 
Main::run_spc(){
  int nv, i;
  nv = 1;
  for (i=0; i<ndim; i++) nv*=kary;
  CProxy_Mapper map = CProxy_Mapper::ckNew(ndim, kary, dim_len);
  CkArrayOptions opts(nv);
  opts.setMap(map);
  vpgrid = CProxy_VPblock::ckNew(opts);

  warmup = -1;
  witer = 0;
}

#ifdef BENCH
void Main::done(){
  if (warmup == -1){
    warmup = 1;
    witer = 0;
    vpgrid.init_rand(n, m, k);
    return;
  }
  if (warmup){
    witer++;
    if (witer == nwarm){
      CkPrintf("Performed %d warm-up iterations\n", nwarm);
      warmup = 0;
      witer = 0;
      iter = 0;
      st = CmiWallTimer();
    }
    vpgrid.contract(alpha, beta);
  } else {
    iter++;
    if (iter == niter){
      int i;
      int khalf = 1;
      for (i=0; i<ndim/2; i++){
        khalf *= kary;
      }
      end = CmiWallTimer();
      CkPrintf("Performed %d timed iterations\n",niter);
      CkPrintf("Avg time per iteration: %lf\n",(end-st)/niter);
      CkPrintf("Achieved flop rate: %lf GF\n", 
                (2.*n*m*k*khalf*khalf*khalf*1.E-9)/((end-st)/niter));
      CkExit();
    } else
      vpgrid.contract(alpha, beta);
  }
}
#endif
#ifdef TEST
void Main::done(){
  if (warmup == -1){
    warmup = 1;
    CkPrintf("Initializing chare data\n");
    vpgrid.init_sindex(n, m, k);
    return;
  }
  if (warmup){
    warmup = 0;
    CkPrintf("Starting multiply\n");
    vpgrid.contract(alpha,beta);
  } else {
    CkPrintf("Gathering data\n");
    vpgrid.gatherC();
  }
}
#endif

void Main::reduceC(CkReductionMsg * msg){
  int khalf, i, pass;
  double * full_A, * full_B, * full_C, * C;

  C = (double*)msg->getData();

  khalf = 1;
  for (i=0; i<ndim/2; i++){
    khalf *= kary;
  }
  CkPrintf("Confirming data\n");

  full_A = (double*)malloc(m*khalf*k*khalf*sizeof(double));
  full_B = (double*)malloc(k*khalf*n*khalf*sizeof(double));
  full_C = (double*)malloc(m*khalf*n*khalf*sizeof(double));

  for (i=0; i<m*khalf*k*khalf; i++){
    srand48(i);
    full_A[i] = drand48();
  }
  for (i=0; i<k*khalf*n*khalf; i++){
    srand48(i);
    full_B[i] = drand48();
  }
  for (i=0; i<m*khalf*n*khalf; i++){
    srand48(i);
    full_C[i] = drand48();
  }

  DGEMM('N','N',khalf*m,khalf*n,khalf*k,alpha,full_A,khalf*m,full_B,
        khalf*k,beta,full_C,khalf*m);

  pass = 1;
  for (i=0; i<m*khalf*n*khalf; i++){
    if (fabs(full_C[i] - C[i]) > 1.E-6){
      pass = 0;
      CkPrintf("Computed C[%d] = %lf, answer is C[%d] = %lf\n", i, C[i], i, full_C[i]);
      break;
    }
  }
  if (pass)
    CkPrintf("Correctness test successful.\n");
  else
    CkPrintf("Correctness test UNsuccessful!!!\n");
  CkExit();
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


/**
 * \brief reads network topology data from input string
 */
void read_topology(char **              input_str, 
                   int const            in_num,
                   int const            numPes,
                   int *                nd,
                   int **               lens){
  int i, ndim, stat, np;
  int * dim_len;

  char str_dim_len[80];
  char str_ndim[80];
  char str_tmp[80];

  strcpy(str_ndim,"-ndim");
  
  if (getCmdOption(input_str, input_str+in_num, str_ndim)){
    ndim = atoi(getCmdOption(input_str, input_str+in_num, str_ndim));
    if (ndim <= 0) ndim = 2;
  } else ndim = 2;
  
  dim_len = (int*)malloc(sizeof(int)*ndim);

  np = 1;
  for (i=0; i<ndim; i++){
    strcpy(str_dim_len,"-p");
    sprintf(str_tmp, "%d", i);
    strcat(str_dim_len,str_tmp);
    if (getCmdOption(input_str, input_str+in_num, str_dim_len)){
      dim_len[i] = atoi(getCmdOption(input_str, input_str+in_num, str_dim_len));
    } else {
      dim_len[i] = 1;
    }
    np *= dim_len[i];
  }
  if (np != numPes){
    printf("%s physical grid is incorrect (grid has %d pes ",  np);
    printf("Number of acutal pes is %d\n", numPes);
    CkExit();
  }
  *nd = ndim;
  *lens = dim_len;
}

Main::Main(CkArgMsg* msg) {
  int i;
  int const  argc = msg->argc;
  char ** argv = msg->argv;
  int const in_num = argc;
  char ** input_str = argv;

  np = CkNumPes();

  // store the main proxy
  mainProxy = thisProxy;

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 10;
  } else niter = 10;
  if (getCmdOption(input_str, input_str+in_num, "-nwarm")){
    nwarm = atoi(getCmdOption(input_str, input_str+in_num, "-nwarm"));
    if (nwarm < 0) nwarm = 1;
  } else nwarm = 1;
  if (getCmdOption(input_str, input_str+in_num, "-bidir")){
    bidir = atoi(getCmdOption(input_str, input_str+in_num, "-bidir"));
    if (bidir < 0) bidir = 1;
  } else bidir = 1;
  if (getCmdOption(input_str, input_str+in_num, "-kary")){
    kary = atoi(getCmdOption(input_str, input_str+in_num, "-kary"));
    if (kary < 0) kary = 1;
  } else kary = 1;
/*  if (getCmdOption(input_str, input_str+in_num, "-seed")){
    seed = atoi(getCmdOption(input_str, input_str+in_num, "-seed"));
    if (seed < 0) seed = 3;
  } else seed = 3;*/
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
  
  read_topology(input_str, in_num, np, &ndim, &dim_len);
  
  CkAssert(ndim >= 2 && ndim%2 == 0);
  CkAssert(k%ndim == 0);


  if (bidir)
    CkPrintf("Virtual topology is a bidirectional %d-ary %d-cube\n", kary, ndim);
  else
    CkPrintf("Virtual topology is a unidirectional %d-ary %d-cube\n", kary, ndim);
  CkPrintf("Physical topology is %d", dim_len[0]);
  for (i=1; i<ndim; i++){
    CkPrintf(" by %d",dim_len[i]);
  }
  CkPrintf(".\n");

#ifdef TEST
  CkPrintf("Testing multiply of block size %d-by-%d A and %d-by-%d B\n", 
         m, k, k, n);
#endif


#ifdef BENCH
  CkPrintf("Benchmarking multiply of block size %d-by-%d A and %d-by-%d B\n", 
         m, k, k, n);
#endif

  run_spc();

}
