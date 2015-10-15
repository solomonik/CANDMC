/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */
#include <mpi.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "lu_offload.h"
#include "../shared/util.h"
#include <algorithm>
#ifdef DGEMM_RING
#include "../shared/omp_dgemm_ring.h"
#endif
#ifdef OFFLOAD
#include "lu_offload.h"
#define HUGE_PAGE_SIZE (2*1024*1024)
#ifdef USE_MIC
__declspec(target(mic:mic_rank))
double * offload_A;
__declspec(target(mic:mic_rank))
double * offload_L;
__declspec(target(mic:mic_rank))
double * offload_U;
__declspec(target(mic:mic_rank))
double * offload_transfer;
__declspec(target(mic:mic_rank))
int * offsets_transfer;
#else
double * offload_A;
double * offload_L;
double * offload_U;
double * offload_transfer;
int * offsets_transfer;
#endif
int64_t size_offload_A;
int64_t size_offload_L;
int64_t size_offload_U;
int64_t size_offload_transfer;
static int gemm_signal=13;
int mic_rank = 0;

#ifdef USE_MIC
__declspec(target(mic:mic_rank))
void MIC_gemm(char       tA,
              char       tB,
              int        m,
              int        n,
              int        k,
              double     alpha,
              int        offset_A,
              OFF_MAT    omat_A,
              int        lda_A,
              int        offset_B,
              OFF_MAT    omat_B,
              int        lda_B,
              double     beta,
              int        offset_C,
              OFF_MAT    omat_C,
              int        lda_C);

__declspec(target(mic:mic_rank))
void MIC_cpy(int nrow,
             double const * __restrict__ from,
             double * __restrict__ to);

__declspec(target(mic:mic_rank))
void MIC_download_lda(int nrow,
                      int ncol, 
                      int lda_A,
                      int offset_A,
                      OFF_MAT omat_A);

__declspec(target(mic:mic_rank))
void MIC_upload_lda(int nrow,
                    int ncol, 
                    int lda_B,
                    int offset_B,
                    OFF_MAT omat_B);

#endif


/* Copies submatrix to submatrix */
#ifdef USE_MIC
__declspec(target(mic:mic_rank))
#endif
void mlda_cpy(const int nrow,  const int ncol,
              const int lda_A, const int lda_B,
              const double *A, double *B){
  if (lda_A == nrow && lda_B == nrow){
    memcpy(B,A,nrow*ncol*sizeof(double));
    return;
  }
  int i;
  for (i=0; i<ncol; i++){
    memcpy(B+lda_B*i,A+lda_A*i,nrow*sizeof(double));
  }

}


#ifndef __C_SRC
extern "C"
#endif
#ifdef BGP
void dgemm
#else
#ifdef USE_MIC
__declspec(target(mic:mic_rank))
#endif
void dgemm_
#endif
            (const char *,      const char *,
            const int *,        const int *,
            const int *,        const double *,
            const double *,     const int *,
            const double *,     const int *,
            const double *,     double *,
                                const int *);

void set_mic_rank(int mic_rank_){
  mic_rank = mic_rank_;
}

void wait_gemm(){
#ifdef ASYNC_GEMM
   #pragma offload_wait target(mic:mic_rank) wait(gemm_signal)
#endif
}

#ifdef USE_MIC
__declspec(target(mic:mic_rank))
void mcdgemm(const char transa, const char transb,
       const int m,   const int n,
       const int k,   const double a,
       const double * A,  const int lda,
       const double * B,  const int ldb,
       const double b,  double * C,
          const int ldc){
#ifdef DGEMM_RING
  omp_dgemm_ring
#else
#ifdef BGP
  dgemm
#else
  dgemm_
#endif
#endif
  (&transa, &transb, &m,
   &n, &k, &a,
   A, &lda, B,
   &ldb, &b, C,
      &ldc);
}
#endif

#ifdef USE_MIC
__declspec(target(mic:mic_rank))
#endif
double * get_mat_handle(OFF_MAT omat){
  double * offload_mat;
  switch  (omat){
    case OFF_A:
      offload_mat = offload_A;
      break;
    case OFF_L:
      offload_mat = offload_L;
      break;
    case OFF_U:
      offload_mat = offload_U;
      break;
  }
  return offload_mat;

}

#ifdef USE_MIC
__declspec(target(mic:mic_rank))
void MIC_gemm(char       tA,
                    char       tB,
                    int        m,
                    int        n,
                    int        k,
                    double     alpha,
                    int        offset_A,
                    OFF_MAT    omat_A,
                    int        lda_A,
                    int        offset_B,
                    OFF_MAT    omat_B,
                    int        lda_B,
                    double     beta,
                    int        offset_C,
                    OFF_MAT    omat_C,
                    int        lda_C){

  double * offload_mat_A = get_mat_handle(omat_A);
  double * offload_mat_B = get_mat_handle(omat_B);
  double * offload_mat_C = get_mat_handle(omat_C);
  DEBUG_PRINTF("offload_L is at %p, offload_U is at %p, offload_A is at %p\n",
                offload_L, offload_U, offload_A);
  DEBUG_PRINTF("multiplying A[%d] = %lf by B[%d] = %lf into C[%d] = %lf\n",
                offset_A, offload_mat_A[offset_A],
                offset_B, offload_mat_B[offset_B],
                offset_C, offload_mat_C[offset_C]);
  mcdgemm(tA, tB, m, n, k, alpha, offload_mat_A+offset_A, lda_A, 
                                  offload_mat_B+offset_B, lda_B, 
                           beta,  offload_mat_C+offset_C, lda_C);
  DEBUG_PRINTF("multiplying A[%d] = %lf by B[%d] = %lf computed C[%d] = %lf\n",
                offset_A, offload_mat_A[offset_A],
                offset_B, offload_mat_B[offset_B],
                offset_C, offload_mat_C[offset_C]);
}
#endif

void offload_gemm_A(char       tA,
                    char       tB,
                    int        m,
                    int        n,
                    int        k,
                    double     alpha,
                    int        offset_A,
                    OFF_MAT    omat_A,
                    int        lda_A,
                    int        offset_B,
                    OFF_MAT    omat_B,
                    int        lda_B,
                    double     beta,
                    int        offset_C,
                    OFF_MAT    omat_C,
                    int        lda_C){
#ifdef USE_MIC
  #pragma offload target(mic:mic_rank) signal(gemm_signal)
  MIC_gemm(tA, tB, m, n, k, alpha, offset_A, omat_A, lda_A,
                                   offset_B, omat_B, lda_B,  
                             beta, offset_C, omat_C, lda_C);
#else
  double * offload_mat_A = get_mat_handle(omat_A);
  double * offload_mat_B = get_mat_handle(omat_B);
  double * offload_mat_C = get_mat_handle(omat_C);
  DEBUG_PRINTF("offload_L is at %p, offload_U is at %p, offload_A is at %p\n",
                offload_L, offload_U, offload_A);
  DEBUG_PRINTF("multiplying A[%d] = %lf by B[%d] = %lf into C[%d] = %lf\n",
                offset_A, offload_mat_A[offset_A],
                offset_B, offload_mat_B[offset_B],
                offset_C, offload_mat_C[offset_C]);
  cdgemm(tA, tB, m, n, k, alpha, offload_mat_A+offset_A, lda_A, 
                                 offload_mat_B+offset_B, lda_B, 
                          beta,  offload_mat_C+offset_C, lda_C);
  DEBUG_PRINTF("multiplying A[%d] = %lf by B[%d] = %lf computed C[%d] = %lf\n",
                offset_A, offload_mat_A[offset_A],
                offset_B, offload_mat_B[offset_B],
                offset_C, offload_mat_C[offset_C]);
#endif
}

#ifdef USE_MIC
__declspec(target(mic:mic_rank))
void MIC_cpy(int nrow,
             double const * __restrict__ from,
             double * __restrict__ to){
  memcpy(to, from, nrow*sizeof(double));

}

__declspec(target(mic:mic_rank))
void MIC_download_lda(int nrow,
                      int ncol, 
                      int lda_A,
                      int offset_A,
                      OFF_MAT omat_A){
  int i;
  double * offload_mat_A = get_mat_handle(omat_A);
  DEBUG_PRINTF("On the MIC data requested from offset %d is %lf\n",
                offset_A,offload_mat_A[offset_A]);
  mlda_cpy(nrow, ncol, lda_A, nrow, offload_mat_A+offset_A, offload_transfer);
  //mkl_domatcopy('C', 'N', nrow, ncol, 1.0, offload_mat_A+offset_A, lda_A, offload_transfer, nrow);
  /*if (lda_A == nrow){
    memcpy(offload_transfer,offload_mat_A+offset_A,nrow*ncol*sizeof(double));
  } else*/

/*  double * __restrict__ * __restrict__ offload_t = (double * __restrict__ * __restrict__)malloc(sizeof(double*)*ncol);
  for (i=0; i<ncol; i++){
    offload_t[i] = offload_transfer + nrow*i;
  }

  #pragma omp parallel private(i)
  {
    int tid = omp_get_thread_num();
    int ntd = omp_get_num_threads();
    if (tid%4 == 0){
      ntd = ntd/4;
      if (ntd%4 > 0) ntd++;
      tid = tid/4;
      int st_col = tid*(ncol/ntd);
      st_col+=MIN(tid,(ncol%ntd));
      int end_col = st_col + (ncol/ntd);
      if (tid < ncol%ntd) end_col++;
      end_col = MIN(ncol, end_col);
      //printf("tid = %d, ntd = %d st_col = %d end_col = %d\n", tid, ntd, st_col, end_col);
      for (i=st_col; i<end_col; i++){
        MIC_cpy(nrow, offload_mat_A+offset_A+lda_A*i, offload_t[i]);
      }
    }
  }*/
  //mkl_domatcopy('C', 'N', nrow, ncol, 1.0, offload_mat_A+offset_A, lda_A, offload_transfer, nrow);
  /*#pragma omp parallel for private (i)
  for (i=0; i<ncol; i++){
    memcpy(offload_transfer+nrow*i,
           offload_mat_A+offset_A+lda_A*i,
           nrow*sizeof(double));
  }*/
}
__declspec(target(mic:mic_rank))
void MIC_upload_lda(int nrow,
                    int ncol, 
                    int lda_B,
                    int offset_B,
                    OFF_MAT omat_B){
  int i;
  double * offload_mat_B = get_mat_handle(omat_B);
  DEBUG_PRINTF("On the MIC data set to offset %d is %lf omat_B = %d offload_mat_B = %p\n",
                offset_B, offload_transfer[0], omat_B, offload_mat_B);
  mlda_cpy(nrow, ncol, nrow, lda_B, offload_transfer, offload_mat_B+offset_B);
/*  if (lda_B == nrow){
    memcpy(offload_mat_B+offset_B,
           offload_transfer,nrow*ncol*sizeof(double));
  }else */
  /*#pragma omp parallel for private (i)
  for (i=0; i<ncol; i++){
    memcpy(offload_mat_B+offset_B+lda_B*i,
           offload_transfer+nrow*i,nrow*sizeof(double));
  }*/
  //mkl_domatcopy('C', 'N', nrow, ncol, 1.0, offload_transfer,  nrow, offload_mat_B+offset_B, lda_B);
}
#endif

void download_lda_cpy(int nrow,
                      int ncol, 
                      int lda_A,
                      int lda_B,
                      int offset_A,
                      double * B,
                      OFF_MAT omat_A){
#ifdef USE_MIC
  TAU_FSTART(MIC_download_lda);
  #pragma offload target(mic:mic_rank) \
    out(offload_transfer:length(nrow*ncol) alloc_if(0) free_if(0)) 
  MIC_download_lda(nrow, ncol, lda_A, offset_A, omat_A);
  TAU_FSTOP(MIC_download_lda);
  //printf("Downloaded %lf on the cpu side\n",offload_transfer[0]);
  mlda_cpy(nrow, ncol, nrow, lda_B, offload_transfer, B);
#else
  int i;
  double * offload_mat_A = get_mat_handle(omat_A);
  lda_cpy(nrow, ncol, lda_A, lda_B, offload_mat_A+offset_A, B);
  /*if (lda_A == nrow && lda_B == nrow){
    memcpy(B,offload_mat_A+offset_A,nrow*ncol*sizeof(double));
    return;
  }
  for (i=0; i<ncol; i++){
    memcpy(B+lda_B*i,offload_mat_A+offset_A+lda_A*i,nrow*sizeof(double));
  }*/
#endif
}

void upload_lda_cpy(int nrow,
                    int ncol, 
                    int lda_A,
                    int lda_B,
                    double const * A,
                    int offset_B,
                    OFF_MAT omat_B){
#ifdef USE_MIC
  double * contig_A;
  mlda_cpy(nrow, ncol, lda_A, nrow, A, offload_transfer);
  TAU_FSTART(MIC_upload_lda);
  #pragma offload target(mic:mic_rank) \
    in(offload_transfer:length(nrow*ncol) alloc_if(0) free_if(0) ) 
  MIC_upload_lda(nrow, ncol, lda_B, offset_B, omat_B);
  TAU_FSTOP(MIC_upload_lda);
#else
  double * offload_mat_B = get_mat_handle(omat_B);
  lda_cpy(nrow, ncol, lda_A, lda_B, A, offload_mat_B+offset_B);
  /*if (lda_A == nrow && lda_B == nrow){
    memcpy(offload_mat_B,A,nrow*ncol*sizeof(double));
    return;
  }
  for (int i=0; i<ncol; i++){
    memcpy(offload_mat_B+offset_B+lda_B*i,A+lda_A*i,nrow*sizeof(double));
  }*/
#endif
}

#ifdef USE_MIC
__declspec(target(mic:mic_rank))
void MIC_offload_sparse_rw(int nrow,
                           int ncol, 
                           int lda_B,
                           OFF_MAT omat_B,
                           char rw){
  double * offload_mat_B = get_mat_handle(omat_B);
  if (rw == 's'){
    #pragma omp parallel for 
    for (int i=0; i<nrow; i++){
      double rbuf[ncol];
      mlda_cpy(1,ncol,lda_B,1,offload_mat_B+offsets_transfer[i],rbuf);
      mlda_cpy(1,ncol,1,lda_B,offload_transfer+i*ncol,offload_mat_B+offsets_transfer[i]);
      memcpy(offload_transfer+i*ncol,rbuf,ncol*sizeof(double));
    }
  } else if (rw == 'r'){
    #pragma omp parallel for 
    for (int i=0; i<nrow; i++){
      mlda_cpy(1,ncol,lda_B,1,offload_mat_B+offsets_transfer[i],offload_transfer+i*ncol);
    }
  } else if (rw == 'w'){
    #pragma omp parallel for 
    for (int i=0; i<nrow; i++){
      mlda_cpy(1,ncol,1,lda_B,offload_transfer+i*ncol,offload_mat_B+offsets_transfer[i]);
    }
  }
}
#endif


void offload_sparse_rw(int nrow,
                       int ncol, 
                       int lda_B,
                       double * A,
                       int lda_A,
                       int * offsets_transfer_,
                       OFF_MAT omat_B,
                       char rw){
  if (ncol == 0 || nrow == 0) return;
  offsets_transfer = offsets_transfer_;
#ifdef USE_MIC
  double * contig_A;
  TAU_FSTART(MIC_sparse_rw);
  if (rw == 's'){
    mlda_cpy(ncol, nrow, lda_A, ncol, A, offload_transfer);
    #pragma offload target(mic:mic_rank) \
      inout ( offload_transfer:length(nrow*ncol) alloc_if(0) free_if(0) ) \
      in ( offsets_transfer:length(nrow) alloc_if(1) free_if(1) ) 
    MIC_offload_sparse_rw(nrow, ncol, lda_B, omat_B, rw);
    mlda_cpy(ncol, nrow, ncol, lda_A, offload_transfer, A);
  } else if (rw == 'r'){
    #pragma offload target(mic:mic_rank) \
      out ( offload_transfer:length(nrow*ncol) alloc_if(0) free_if(0) ) \
      in ( offsets_transfer:length(nrow) alloc_if(1) free_if(1) ) 
    MIC_offload_sparse_rw(nrow, ncol, lda_B, omat_B, rw);
    mlda_cpy(ncol, nrow, ncol, lda_A, offload_transfer, A);
  } else if (rw == 'w'){
    mlda_cpy(ncol, nrow, lda_A, ncol, A, offload_transfer);
    #pragma offload target(mic:mic_rank) \
      in ( offload_transfer:length(nrow*ncol) alloc_if(0) free_if(0) ) \
      in ( offsets_transfer:length(nrow) alloc_if(1) free_if(1) ) 
    MIC_offload_sparse_rw(nrow, ncol, lda_B, omat_B, rw);
  } else ABORT;
  TAU_FSTOP(MIC_sparse_rw);
#else
  double * offload_mat_B = get_mat_handle(omat_B);
  if (rw == 's'){
    double * rbuf = (double*)malloc(sizeof(double)*ncol);
    for (int i=0; i<nrow; i++){
      lda_cpy(1,ncol,lda_B,1,offload_mat_B+offsets_transfer[i],rbuf);
      lda_cpy(1,ncol,1,lda_B,A+i*lda_A,offload_mat_B+offsets_transfer[i]);
      memcpy(A+i*lda_A,rbuf,ncol*sizeof(double));
    }
    free(rbuf);
  } else if (rw == 'r'){
    for (int i=0; i<nrow; i++){
      lda_cpy(1,ncol,lda_B,1,offload_mat_B+offsets_transfer[i],A+i*lda_A);
    }
  } else if (rw == 'w'){
    for (int i=0; i<nrow; i++){
      lda_cpy(1,ncol,1,lda_B,A+i*lda_A,offload_mat_B+offsets_transfer[i]);
    }
  } else ABORT;
#endif
}

void alloc_A(int64_t size, double * ptr){
#ifdef USE_MIC
  offload_A = ptr;
  #pragma offload target(mic:mic_rank) \
    in (offload_A:length(size) alloc_if(1) free_if(0) ) 
  {
  }
#else
  assert(0==posix_memalign((void**)&offload_A, ALIGN_BYTES, sizeof(double)*size)); 
#endif
  size_offload_A = size;
}

/*
void upload_A(){
#ifdef USE_MIC
  #pragma offload target(mic:mic_rank) \
    in (offload_A:length(size_offload_A) alloc_if(0) free_if(0) ) 
  {
  }
#else
  memcpy(offload_A,ptr,size_offload_A*sizeof(double));
#endif
}*/


void alloc_L(int64_t size){
#ifdef USE_MIC
  assert(0==posix_memalign((void**)&offload_L, HUGE_PAGE_SIZE, sizeof(double)*size)); 
  #pragma offload target(mic:mic_rank) \
    nocopy(offload_L:length(size) alloc_if(1) free_if(0) ) 
  {
  }
#else
  assert(0==posix_memalign((void**)&offload_L, ALIGN_BYTES, sizeof(double)*size)); 
#endif
  size_offload_L = size;
}
void alloc_U(int64_t size){
#ifdef USE_MIC
  assert(0==posix_memalign((void**)&offload_U, HUGE_PAGE_SIZE, sizeof(double)*size)); 
  #pragma offload target(mic:mic_rank) \
    nocopy(offload_U:length(size) alloc_if(1) free_if(0) ) 
  {
  }
#else
  assert(0==posix_memalign((void**)&offload_U, ALIGN_BYTES, sizeof(double)*size)); 
#endif
  size_offload_U = size;
}

void alloc_transfer(int64_t size){
#ifdef USE_MIC
  assert(0==posix_memalign((void**)&offload_transfer, HUGE_PAGE_SIZE, sizeof(double)*size)); 
  #pragma offload target(mic:mic_rank) \
    nocopy(offload_transfer:length(size) alloc_if(1) free_if(0) ) 
  {
    DEBUG_PRINTF("offload_transfer allocated %p\n", offload_transfer);
//    assert(0==posix_memalign((void**)&offload_transfer, HUGE_PAGE_SIZE, sizeof(double)*size)); 
  }
#endif
  size_offload_transfer = size;
}


void free_offload_A(){//int64_t size){
#ifdef USE_MIC
  #pragma offload target(mic:mic_rank) \
    nocopy(offload_A:length(size_offload_A) alloc_if(0) free_if(1)) 
  {
  }
#else
  free(offload_A);
#endif
}

void free_offload_L(){//int64_t size){
#ifdef USE_MIC
  #pragma offload target(mic:mic_rank) \
    nocopy(offload_L:length(size_offload_L) alloc_if(0) free_if(1)) 
  {
  }
#else
  free(offload_L);
#endif
}
void free_offload_U(){//int64_t size){
#ifdef USE_MIC
  #pragma offload target(mic:mic_rank) \
    nocopy(offload_U:length(size_offload_U) alloc_if(0) free_if(1)) 
  {
  }
#else
  free(offload_U);
#endif
}

void free_offload_transfer(){//int64_t size){
#ifdef USE_MIC
  #pragma offload target(mic:mic_rank) \
    nocopy(offload_transfer:length(size_offload_transfer) alloc_if(0) free_if(1)) 
  {
  }
#else
  free(offload_transfer);
#endif
}
#endif
