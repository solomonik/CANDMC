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

#ifndef __SPCANNON_INTERNAL_H__
#define __SPCANNON_INTERNAL_H__

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define ALIGN           16
#define MALLOC(s)       malloc(s)
#define DGEMM           cdgemm
#define BLAS_DGEMM      dgemm_
#define TRANSPOSE       naive_transp
#ifndef WRAP
#define WRAP(a,b)       ((a + b)%b)
#endif


extern "C"
void dgemm_(const char *,       const char *,
                const int *,    const int *,
                const int *,    const double *,
                const double *, const int *,
                const double *, const int *,
                const double *, double *,
                                const int *);

inline
void DGEMM(const char transa,   const char transb,
           const int m,         const int n,
           const int k,         const double a,
           const double * A,    const int lda,
           const double * B,    const int ldb,
           const double b,      double * C,
                                const int ldc){
  BLAS_DGEMM(&transa, &transb, &m, &n, &k, &a, A,
             &lda, B, &ldb, &b, C, &ldc);

}

inline 
void TRANSPOSE(int const lda_fr, int const lda_to, double * A, double * buf){
  int i,j;
  for (i=0; i<lda_fr; i++){
    for (j=0; j<lda_to; j++){
      buf[i*lda_to+j] = A[j*lda_fr + i];
    }
  }
  memcpy(A, buf, lda_to*lda_fr*sizeof(double));
}

#endif
