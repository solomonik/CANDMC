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

#include <assert.h>
#include "spcannon.h"
#include "spcannon_internal.h"

#define ALIGN   16

static
void uni_stagger(int const      rank,
                 int const      kary,
                 int const      ndim,
                 int const      level,
                 MPI_Comm const comm,
                 int const      n,
                 int const      m,
                 int const      k,
                 double *       A,
                 double *       buf_A,
                 MPI_Win        win_A,
                 double *       B,
                 double *       buf_B,
                 MPI_Win        win_B){
  int i,j,tA,tr,tB,s,bA,bB;

  bA = 2*m*k/ndim;
  bB = 2*k*n/ndim;
  
  tr = rank;
  s=1;

  MPI_Win_fence(0, win_A);
  MPI_Win_fence(0, win_B);
  for (j=0; j<ndim/2; j++){
    i = (j+level)%(ndim/2);
    tA = tr % kary;
    tr = tr / kary;
    tB = tr % kary;
    tr = tr / kary;

    MPI_Put(A+i*bA, bA, MPI_DOUBLE, 
            rank+((WRAP((tA-tB),kary))-tA)*s,
              i*bA, bA, MPI_DOUBLE, win_A); 
    s = s*kary;
    
    MPI_Put(B+i*bB, bB, MPI_DOUBLE, 
            rank+((WRAP((tB-tA),kary))-tB)*s,
              i*bB, bB, MPI_DOUBLE, win_B); 
    s = s*kary;
  }
  MPI_Win_fence(0, win_A);
  MPI_Win_fence(0, win_B);
  memcpy(A, buf_A, m*k*sizeof(double));
  memcpy(B, buf_B, k*n*sizeof(double));

  if (level<ndim/2-1){
    uni_stagger(rank, kary, ndim, level+1, comm, n, m, k, 
                A, buf_A, win_A,
                B, buf_B, win_B);
  } 
}

static
void bdr_shift(int const        rank,
               int const        kary,
               int const        ndim,
               int const        level,
               MPI_Comm const   comm,
               int const        n,
               int const        m,
               int const        k,
               double const     alpha,
               double *         A,
               double *         buf_A,
               MPI_Win          win_A,
               double const     beta,
               double *         B,
               double *         buf_B,
               MPI_Win          win_B,
               double *         C){
  int i,ka,j,tA,tr,tB,s,bA,bB;
  double dbeta = beta;

  bA = m*k/ndim;
  bB = k*n/ndim;
  
  for (ka=0; ka<kary; ka++){
    if (level<ndim/2-1){
      bdr_shift(rank, kary, ndim, level+1, comm, n, m, k, 
                alpha, A, buf_A, win_A,
                dbeta,  B, buf_B, win_B, C);
    } else {
      TAU_FSTART(DGEMM);
      DGEMM('N','T',m,n,k,alpha,A,m,B,n,dbeta,C,m);
      TAU_FSTOP(DGEMM);
    }
    dbeta=1.0;

    MPI_Win_fence(0, win_A);
    MPI_Win_fence(0, win_B);
    tr = rank;
    s = 1;
    for (j=0; j<ndim/2; j++){
      i = (j+level)%(ndim/2);
      tA = tr % kary;
      tr = tr / kary;
      tB = tr % kary;
      tr = tr / kary;

      /*printf("[%d] shifting to %d %d %d %d\n", rank,
              rank+((WRAP((tA+1),kary))-tA)*s,
              rank+((WRAP((tA-1),kary))-tA)*s,
              rank+((WRAP((tB+1),kary))-tB)*s*kary,
              rank+((WRAP((tB-1),kary))-tB)*s*kary);*/

      MPI_Put(A+2*i*bA, bA, MPI_DOUBLE, 
              rank+((WRAP((tA+1),kary))-tA)*s,
                2*i*bA, bA, MPI_DOUBLE, win_A); 
      MPI_Put(A+(2*i+1)*bA, bA, MPI_DOUBLE, 
              rank+((WRAP((tA-1),kary))-tA)*s,
                (2*i+1)*bA, bA, MPI_DOUBLE, win_A); 
      s = s*kary;

      MPI_Put(B+2*i*bB, bB, MPI_DOUBLE, 
              rank+((WRAP((tB+1),kary))-tB)*s,
                2*i*bB, bB, MPI_DOUBLE, win_B); 
      MPI_Put(B+(2*i+1)*bB, bB, MPI_DOUBLE, 
              rank+((WRAP((tB-1),kary))-tB)*s,
                (2*i+1)*bB, bB, MPI_DOUBLE, win_B); 
      s = s*kary;

    }
    MPI_Win_fence(0, win_A);
    MPI_Win_fence(0, win_B);
      
    memcpy(A, buf_A, m*k*sizeof(double));
    memcpy(B, buf_B, k*n*sizeof(double));
  }
}

static
void uni_shift(int const        rank,
               int const        kary,
               int const        ndim,
               int const        level,
               MPI_Comm const   comm,
               int const        n,
               int const        m,
               int const        k,
               double const     alpha,
               double *         A,
               double *         buf_A,
               MPI_Win          win_A,
               double const     beta,
               double *         B,
               double *         buf_B,
               MPI_Win          win_B,
               double *         C){
  int i,ka,j,tA,tr,tB,s,bA,bB;
  double dbeta = beta;

  bA = 2*m*k/ndim;
  bB = 2*k*n/ndim;
  
  for (ka=0; ka<kary; ka++){
    if (level<ndim/2-1){
      uni_shift(rank, kary, ndim, level+1, comm, n, m, k, 
                alpha, A, buf_A, win_A,
                dbeta,  B, buf_B, win_B, C);
    } else {
      TAU_FSTART(DGEMM);
      DGEMM('N','T',m,n,k,alpha,A,m,B,n,dbeta,C,m);
      TAU_FSTOP(DGEMM);
    }
    dbeta=1.0;

    MPI_Win_fence(0, win_A);
    MPI_Win_fence(0, win_B);
    tr = rank;
    s = 1;
    for (j=0; j<ndim/2; j++){
      i = (j+level)%(ndim/2);
      tA = tr % kary;
      tr = tr / kary;
      tB = tr % kary;
      tr = tr / kary;

      /*printf("[%d] shifting to %d %d %d %d\n", rank,
              rank+((WRAP((tA+1),kary))-tA)*s,
              rank+((WRAP((tA-1),kary))-tA)*s,
              rank+((WRAP((tB+1),kary))-tB)*s*kary,
              rank+((WRAP((tB-1),kary))-tB)*s*kary);*/

      MPI_Put(A+i*bA, bA, MPI_DOUBLE, 
              rank+((WRAP((tA+1),kary))-tA)*s,
                i*bA, bA, MPI_DOUBLE, win_A); 
      s = s*kary;

      MPI_Put(B+i*bB, bB, MPI_DOUBLE, 
              rank+((WRAP((tB+1),kary))-tB)*s,
                i*bB, bB, MPI_DOUBLE, win_B); 
      s = s*kary;

    }
    MPI_Win_fence(0, win_A);
    MPI_Win_fence(0, win_B);
      
    memcpy(A, buf_A, m*k*sizeof(double));
    memcpy(B, buf_B, k*n*sizeof(double));
  }
}


void kput_cannon(int const      rank,
                 int const      kary,
                 int const      ndim,
                 MPI_Comm const comm,
                 int const      n,
                 int const      m,
                 int const      k,
                 char const     transp_A,
                 double const   alpha,
                 double *       A,
                 char const     transp_B,
                 double const   beta,
                 double *       B,
                 double *       C){
  double * buf_A, * buf_B;
  MPI_Win win_A, win_B;

  assert(k%ndim == 0);
  
  /* Allocate buffer space */
  posix_memalign((void**)&buf_A, ALIGN, m*k*sizeof(double));
  posix_memalign((void**)&buf_B, ALIGN, n*k*sizeof(double));

  
  /* Transpose to make buffer easily divisible */
  if (transp_A == 'T' || transp_A == 't'){
    TRANSPOSE(k,m,A,buf_A);
  }
  if (transp_B == 'N' || transp_B == 'n'){
    TRANSPOSE(k,n,B,buf_B);
  }

  /* Set up MPI windows */
  MPI_Win_create(buf_A, m*k*sizeof(double), sizeof(double),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &win_A);
  MPI_Win_create(buf_B, n*k*sizeof(double), sizeof(double),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &win_B);
  MPI_Win_fence(0, win_A);
  MPI_Win_fence(0, win_B);
  
  /* Stagger */
  TAU_FSTART(uni_stagger);
  uni_stagger(rank, kary, ndim, 0, comm, n, m, k,
              A, buf_A, win_A,
              B, buf_B, win_B);
  TAU_FSTOP(uni_stagger);
  
  /* Shift and multiply */
  TAU_FSTART(bdr_shift);
  bdr_shift(rank, kary, ndim, 0, comm, n, m, k,
            alpha, A, buf_A, win_A,
            beta,  B, buf_B, win_B, C);
  TAU_FSTOP(bdr_shift);

}

void kuni_cannon(int const      rank,
                 int const      kary,
                 int const      ndim,
                 MPI_Comm const comm,
                 int const      n,
                 int const      m,
                 int const      k,
                 char const     transp_A,
                 double const   alpha,
                 double *       A,
                 char const     transp_B,
                 double const   beta,
                 double *       B,
                 double *       C){
  double * buf_A, * buf_B;
  MPI_Win win_A, win_B;

  assert(k%ndim == 0);
  
  /* Allocate buffer space */
  posix_memalign((void**)&buf_A, ALIGN, m*k*sizeof(double));
  posix_memalign((void**)&buf_B, ALIGN, n*k*sizeof(double));

  
  /* Transpose to make buffer easily divisible */
  if (transp_A == 'T' || transp_A == 't'){
    TRANSPOSE(k,m,A,buf_A);
  }
  if (transp_B == 'N' || transp_B == 'n'){
    TRANSPOSE(k,n,B,buf_B);
  }

  /* Set up MPI windows */
  MPI_Win_create(buf_A, m*k*sizeof(double), sizeof(double),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &win_A);
  MPI_Win_create(buf_B, n*k*sizeof(double), sizeof(double),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &win_B);
  MPI_Win_fence(0, win_A);
  MPI_Win_fence(0, win_B);
  
  /* Stagger */
  TAU_FSTART(uni_stagger);
  uni_stagger(rank, kary, ndim, 0, comm, n, m, k,
              A, buf_A, win_A,
              B, buf_B, win_B);
  TAU_FSTOP(uni_stagger);
  
  /* Shift and multiply */
  TAU_FSTART(uni_shift);
  uni_shift(rank, kary, ndim, 0, comm, n, m, k,
            alpha, A, buf_A, win_A,
            beta,  B, buf_B, win_B, C);
  TAU_FSTOP(uni_shift);

}
