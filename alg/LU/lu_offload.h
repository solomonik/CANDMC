#ifndef __LU_OFFLOAD_H__
#define __LU_OFFLOAD_H__

#include <mpi.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../shared/util.h"


#ifdef USE_MIC
#include "mkl.h"
#include "omp.h"
#define ASYNC_GEMM
#define MIC_PER_NODE 2
#endif

enum OFF_MAT { OFF_A, OFF_L, OFF_U };

#ifdef USE_MIC
__declspec(target(mic:mic_rank))
#endif
double * get_mat_handle(OFF_MAT omat);

void set_mic_rank(int mic_rank);

void wait_gemm();

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
                    int        lda_C);

void download_lda_cpy(int nrow,
                      int ncol, 
                      int lda_A,
                      int lda_B,
                      int offset_A,
                      double * B,
                      OFF_MAT omat_A);

void download_lda_cpy(int nrow,
                      int ncol, 
                      int lda_A,
                      int lda_B,
                      int offset_A,
                      double * B,
                      OFF_MAT omat_A);


void upload_lda_cpy(int nrow,
                    int ncol, 
                    int lda_A,
                    int lda_B,
                    double const * A,
                    int offset_B,
                    OFF_MAT omat_B);


void upload_lda_cpy(int nrow,
                    int ncol, 
                    int lda_A,
                    int lda_B,
                    double const * A,
                    int offset_B,
                    OFF_MAT omat_B);

void offload_sparse_rw(int nrow,
                       int ncol, 
                       int lda_B,
                       double * A,
                       int lda_A,
                       int * offsets_transfer,
                       OFF_MAT omat_B,
                       char rw);

void upload_A();
void alloc_A(int64_t size, double * ptr);
void alloc_L(int64_t size);
void alloc_U(int64_t size);
void alloc_transfer(int64_t size);
void free_offload_A();//int64_t size);
void free_offload_L();//int64_t size);
void free_offload_U();//int64_t size);
void free_offload_transfer();//int64_t size);


#endif
