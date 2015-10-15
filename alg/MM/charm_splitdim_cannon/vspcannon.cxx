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
//#include "vspcannon.decl.h"

extern /*readonly*/ CProxy_Main mainProxy;
extern /*readonly*/ CProxy_VPblock vpgrid;
extern /*readonly*/ int kary;
extern /*readonly*/ int ndim;
extern /*readonly*/ int bidir;


int get_idx(int const ndim,
            int const kary,
            int const my_idx,
            int const dir){
  int i, idx;
  idx = my_idx;
  for (i=0; i<dir; i++){
    idx = idx / kary;
  }
  return idx % kary;
}

int get_neighbor(int const ndim, 
                 int const kary, 
                 int const my_idx, 
                 int const dim, 
                 int const dir){
  int i, nidx, r, s;
  nidx = my_idx;
  s = 1;
  for (i=0; i<=dim; i++){
    s = s*kary;
    r = nidx%kary;
    nidx = nidx/kary;
  }
  nidx = my_idx - r*(s/kary) + WRAP((r+dir),kary)*(s/kary);
  return nidx;
}


Mapper::Mapper(int ndim, int kary, int * dim_len){
  int nv, i, j, idx, lda;
  nv = 1;
  for (i=0; i<ndim; i++) {
    nv*=kary;
    CkAssert(kary%dim_len[i] == 0);
  }

  mapping = (int*)malloc(nv*sizeof(int));

  for (i=0; i<nv; i++){
    idx = 0;
    lda = 1;
    for (j=0; j<ndim; j++){
      idx += lda*(get_idx(ndim, kary, i, j)/(kary/dim_len[j]));
      lda = lda*dim_len[j];
    }
    mapping[i] = idx;
  }
}
    
Mapper::~Mapper(){
  delete [] mapping;
}

int Mapper::procNum(int, const CkArrayIndex &idx){
  int *index = (int *)idx.data();
  return mapping[index[0]];
}

VPblock::VPblock(){
  stgr_set = 0, shft_set = 0;
  if (bidir) nmsg = 2*ndim;
  else nmsg = ndim;
  contribute(CkCallback(CkIndex_Main::done(), mainProxy));
}
    
    
VPblock::VPblock(CkMigrateMessage *msg){

}

void VPblock::init_sindex(int nb, int mb, int kb){
  int i, j, mem, px, py, s, tr, khalf;
  double * buf_B;

  n=nb, m=mb, k=kb;
  
  mem = posix_memalign((void**)&A, ALIGN, m*k*sizeof(double));
  mem = posix_memalign((void**)&buf_B, ALIGN, n*k*sizeof(double));
  mem = posix_memalign((void**)&B, ALIGN, n*k*sizeof(double));
  mem = posix_memalign((void**)&C, ALIGN, m*n*sizeof(double));
  CkAssert(mem==0);
  
  khalf = 1;
  for (i=0; i<ndim/2; i++){
    khalf *= kary;
  }
  
  px=0, py=0, s=1;
  tr=thisIndex;
  for (i=0; i<ndim/2; i++){
    px += (tr%kary)*s;
    tr  = tr/kary;
    py += (tr%kary)*s;
    tr  = tr/kary;
    s = s*kary;
  }
  
  for (i=0; i<k; i++){
    for (j=0; j<m; j++){
      srand48((px*k+i)*khalf*m + (py*m+j));
      A[i*m+j] = drand48();
    }
  }
  for (i=0; i<n; i++){
    for (j=0; j<k; j++){
      srand48((px*n+i)*khalf*k + (py*k+j));
      B[i*k+j] = drand48();
    }
  }
  TRANSPOSE(k,n,B,buf_B);
  free(buf_B);
  for (i=0; i<n; i++){
    for (j=0; j<m; j++){
      srand48((px*n+i)*khalf*m + (py*m+j));
      C[i*m+j] = drand48();
    }
  }
//  CkPrintf("[%d][%d] finished init_rand\n", CkMyPe(), thisIndex);
  contribute(CkCallback(CkIndex_Main::done(), mainProxy));
  
}

void VPblock::init_rand(int nb, int mb, int kb){
  int i, mem;
  int seed = 100*thisIndex;

  n=nb, m=mb, k=kb;
  
  mem = posix_memalign((void**)&A, ALIGN, m*k*sizeof(double));
  mem = posix_memalign((void**)&B, ALIGN, n*k*sizeof(double));
  mem = posix_memalign((void**)&C, ALIGN, m*n*sizeof(double));
  CkAssert(mem==0);
  
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
//  CkPrintf("[%d][%d] finished init_rand\n", CkMyPe(), thisIndex);
  contribute(CkCallback(CkIndex_Main::done(), mainProxy));
  
}
    
void VPblock::gatherC(){
  int i, j, mem, px, py, s, tr, khalf;
  double * full_C;
  
  khalf = 1;
  for (i=0; i<ndim/2; i++){
    khalf *= kary;
  }
  
  mem = posix_memalign((void**)&full_C, ALIGN, m*khalf*n*khalf*sizeof(double));
  CkAssert(mem==0);
  
  px=0, py=0, s=1;
  tr=thisIndex;
  for (i=0; i<ndim/2; i++){
    px += (tr%kary)*s;
    tr  = tr/kary;
    py += (tr%kary)*s;
    tr  = tr/kary;
    s = s*kary;
  }

  std::fill(full_C, full_C+m*khalf*n*khalf, 0.0);
  
  for (i=0; i<n; i++){
    for (j=0; j<m; j++){
      full_C[(px*n+i)*khalf*m + (py*m+j)] = C[i*m+j];
    }
  }
  CkReductionMsg* redmsg = CkReductionMsg::buildNew(m*khalf*n*khalf*sizeof(double), full_C, CkReduction::sum_double);
  redmsg->setCallback(CkCallback(CkIndex_Main::reduceC(NULL), mainProxy));
  this->contribute(redmsg);
}

void VPblock::contract(double a, double b){
  int i;
//  CkPrintf("[%d][%d] in contract\n", CkMyPe(), thisIndex);
  alpha = a, beta = b;

  level = -1;
  stgr_acc_table = (int*)calloc(sizeof(int)*ndim,1);

  stagger();
}

void VPblock::staggerA(StgrMsg * msg){
  int im;
  im = stgr_queue_A.size();
  stgr_queue_A.push_back(msg);
  loc_staggerA(im);
}
    
void VPblock::loc_staggerA(int im){
  int i, pass;
  StgrMsg * msg = stgr_queue_A[im];
  if (stgr_set == 0 || level != msg->level || stgr_acc_table[msg->dim] != 0){
    thisProxy[thisIndex].loc_staggerA(im);
    //Printf("[%d] cycle on stagger A, shft_set = %d\n", thisIndex, shft_set);
  } else {
    stgr_acc_table[msg->dim] = 1;
    
    memcpy(A+(msg->dim/2)*(2*m*k/ndim), msg->data, 2*m*k/ndim*sizeof(double));
    delete msg;

    pass = 1;
    for (i=0; i<ndim; i++)
      if (stgr_acc_table[i] == 0) pass = 0;
    if (pass) stagger();
  }
}

void VPblock::staggerB(StgrMsg * msg){
  int im;
  im = stgr_queue_B.size();
  stgr_queue_B.push_back(msg);
  loc_staggerB(im);
}

void VPblock::loc_staggerB(int im){
  int i, pass;
  StgrMsg * msg = stgr_queue_B[im];
  if (stgr_set == 0 || level != msg->level || stgr_acc_table[msg->dim] != 0){
    thisProxy[thisIndex].loc_staggerB(im);
    //Printf("[%d] cycle on stagger B, shft_set = %d\n", thisIndex, shft_set);
  } else {
    stgr_acc_table[msg->dim] = 1;
    
    memcpy(B+(msg->dim/2)*(2*k*n/ndim), msg->data, 2*k*n/ndim*sizeof(double));
    delete msg;

    pass = 1;
    for (i=0; i<ndim; i++)
      if (stgr_acc_table[i] == 0) pass = 0;
    if (pass) stagger();
  }
}

void VPblock::stagger(){
  int i,j,l;
  stgr_set = 1, shft_set = 0;
  memset(stgr_acc_table, 0, sizeof(int)*ndim);
  level++;

  if (level == ndim/2){
    start_shift();
    return;
  }
  //Printf("[%d] stagger level %d\n",thisIndex,level);

  for (l=0; l<ndim/2; l++){
    j = (l+thisIndex)%(ndim/2);
    i = (j+level)%(ndim/2);
    StgrMsg * mA = new (2*m*k/ndim) StgrMsg;
    mA->dim = 2*i;
    mA->level = level;
    memcpy(mA->data, A+i*(2*m*k/ndim),  2*m*k/ndim*sizeof(double));
    /*CkPrintf("[%d] sending A to -%d, %d\n", thisIndex,
                           get_idx(ndim,kary,thisIndex,2*i+1),
              get_neighbor(ndim, kary, thisIndex, 2*i, 
                           -get_idx(ndim,kary,thisIndex,2*i+1)));*/
    thisProxy[get_neighbor(ndim, kary, thisIndex, 2*j, 
                           -get_idx(ndim,kary,thisIndex,2*j+1))].staggerA(mA);
    
    StgrMsg * mB = new (2*k*n/ndim) StgrMsg;
    mB->dim = 2*i+1;
    mB->level = level;
    memcpy(mB->data, B+i*(2*k*n/ndim),  2*k*n/ndim*sizeof(double));
    /*CkPrintf("[%d] sending B to -%d, %d\n", thisIndex,
                           get_idx(ndim,kary,thisIndex,2*i),
              get_neighbor(ndim, kary, thisIndex, 2*i+1, 
                           -get_idx(ndim,kary,thisIndex,2*i)));*/
    thisProxy[get_neighbor(ndim, kary, thisIndex, 2*j+1, 
                           -get_idx(ndim,kary,thisIndex,2*j))].staggerB(mB);
  }
}

void VPblock::start_shift(){
  int i;
  free(stgr_acc_table);
  shft_acc_table = (int*)malloc(sizeof(int)*nmsg);
  shft_pidx = (int*)calloc(sizeof(int)*ndim,1);
  level = 0;

  shift();
}

void VPblock::shiftA(ShftMsg * msg){
  int im;
  im = shft_queue_A.size();
  shft_queue_A.push_back(msg);
  loc_shiftA(im);
}

void VPblock::loc_shiftA(int im){
  int i, pass;
  ShftMsg * msg = shft_queue_A[im];
  if (shft_set == 0 || level != msg->level || shft_pidx[level] != msg->pidx || shft_acc_table[msg->dim] != 0){
    thisProxy[thisIndex].loc_shiftA(im);
    //if (shft_set)
      //CkPrintf("cycle on A %d %d %d %d\n", level, msg->level, shft_pidx[level], msg->pidx);
  } else {
    shft_acc_table[msg->dim] = 1;
    
    memcpy(A+(msg->dim/2)*(2*m*k/nmsg), msg->data, 2*m*k/nmsg*sizeof(double));

    delete msg;

    pass = 1;
    for (i=0; i<nmsg; i++)
      if (shft_acc_table[i] == 0) pass = 0;
    if (pass) shift();
  }
}

void VPblock::shiftB(ShftMsg * msg){
  int im;
  im = shft_queue_B.size();
  shft_queue_B.push_back(msg);
  loc_shiftB(im);
}

void VPblock::loc_shiftB(int im){
  int i, pass;
  ShftMsg * msg = shft_queue_B[im];
  if (shft_set == 0 || level != msg->level || shft_pidx[level] != msg->pidx || shft_acc_table[msg->dim] != 0){
//    CkPrintf("[%d] cycle on B, shft_set = %d\n", thisIndex, shft_set);
    thisProxy[thisIndex].loc_shiftB(im);
  } else {
    shft_acc_table[msg->dim] = 1;
    
    memcpy(B+(msg->dim/2)*(2*k*n/nmsg), msg->data, 2*k*n/nmsg*sizeof(double));
    delete msg;

    pass = 1;
    for (i=0; i<nmsg; i++)
      if (shft_acc_table[i] == 0) pass = 0;
    if (pass) shift();
  }
}


void VPblock::shift(){
  int i,j,l,ii,ll;
  memset(shft_acc_table, 0, sizeof(int)*nmsg);
  
  if (shft_pidx[level] == kary){
    shft_pidx[level] = 0;
    level++;
    if (level == ndim/2){
      contribute(CkCallback(CkIndex_Main::done(), mainProxy));
      return;
    }
  } else {
    level = 0;
    DGEMM('N','T',m,n,k,alpha,A,m,B,n,beta,C,m);
    beta=1.0;
  }
  shft_pidx[level]++;
  //CkPrintf("shift level = %d\n", level);
    
      
    
  stgr_set = 0, shft_set = 1;

  for (l=0; l<nmsg/2; l++){
    ll = (l+thisIndex)%(nmsg/2);
    j = (l+thisIndex)%(ndim/2);
    i = (j+level)%(ndim/2);
    if (bidir && ll >= ndim/2) ii = i+ndim/2;
    else ii = i;
    //DGEMM
    ShftMsg * mA = new (2*m*k/nmsg) ShftMsg;
    mA->dim = 2*ii;
    mA->level = level;
    mA->pidx = shft_pidx[level];
    memcpy(mA->data, A+ii*(2*m*k/nmsg),  2*m*k/nmsg*sizeof(double));
    if (bidir && ll >= ndim/2)
      thisProxy[get_neighbor(ndim, kary, thisIndex, 2*j, -1)].shiftA(mA);
    else
      thisProxy[get_neighbor(ndim, kary, thisIndex, 2*j, 1)].shiftA(mA);
    
    ShftMsg * mB = new (2*k*n/nmsg) ShftMsg;
    mB->dim = 2*ii+1;
    mB->level = level;
    mB->pidx = shft_pidx[level];
    memcpy(mB->data, B+ii*(2*k*n/nmsg),  2*k*n/nmsg*sizeof(double));
    if (bidir && ll>= ndim/2) {
      thisProxy[get_neighbor(ndim, kary, thisIndex, 2*j+1, -1)].shiftB(mB);
    } else
      thisProxy[get_neighbor(ndim, kary, thisIndex, 2*j+1, 1)].shiftB(mB);
  }

}
#include "vspcannon.def.h"
