/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "../../shared/util.h"
#include "../tsqr/bitree_tsqr.h"
#include "../tsqr/butterfly_tsqr.h"
#include "yamamoto.h"

#define BSIZE 64

/**
 * \brief Perform signed LU
 *
 * \param[in,out] A dense lda_A-by-b matrix, 
 *                  on output the top b-by-b matrix contains L and U factors of itself minus S
 * \param[in,out] sign matrix S
 * \param[in] b number of rows/columns in A
 * \param[in] lda_A leading dimension (number of buffer rows) in A
 **/
void signed_YLU(double  * A,
                int64_t   b,
                int64_t   lda_A,
                int *     signs){
  int64_t info, i, j, pos;

  for (i=0; i<b; i++){
    if (A[i*lda_A+i] > 0)
      signs[i] = -1;
    else 
      signs[i] = 1;
    //assert(signs[i]==1);
    A[i*lda_A+i] = A[i*lda_A+i] - signs[i];
    //A[i*lda_A+i] = 1 - signs[i]*A[i*lda_A+i];

    for (j=1; j<b-i; j++){
      A[i*lda_A+i+j] = A[i*lda_A+i+j]/A[i*lda_A+i];
    }
    cdger(b-i-1, b-i-1, -1.0, A+i*lda_A+i+1, 1, 
          A+(i+1)*lda_A+i, lda_A,
          A+(i+1)*lda_A+(i+1), lda_A);
  } 
}


/**
 * \brief Perform recursive signed LU 
 *
 * \param[in,out] A b-by-b dense square matrix, L\U on output
 * \param[in] b number of rows/columns in A
 * \param[in] lda_A leading dimension (number of buffer rows) in A
 * \param[in] signs of R rows to use
 **/
void recursive_YLU(double * A,
                   int64_t  b,
                   int64_t  lda_A,
                   int *    signs){
  int64_t i, b1, b2;
  int info;
 
  if (b<=BSIZE){
    signed_YLU(A, b, lda_A, signs);
    return;
  }
  
  b1 = b/2;
  b2 = b-b1;

  recursive_YLU(A, b1, lda_A, signs);

  cdtrsm('L', 'L', 'N', 'U', b1, b2, 1.0, A, lda_A, A+b1*lda_A, lda_A);
  cdtrsm('R', 'U', 'N', 'N', b2, b1, 1.0, A, lda_A, A+b1,       lda_A);

  cdgemm('N', 'N', b2, b2, b1, -1.0, A+b1, lda_A, A+b1*lda_A, lda_A, 1.0, A+b1*(lda_A+1), lda_A);
  recursive_YLU(A+b1*(lda_A+1), b2, lda_A, signs+b1);
}

/**
 * \brief Perform TSQR and construct Yamamoto's basis kernel representation [Q1-S; Q2][Q1-S]^-1[Q1-S; Q2] 
 *          on a (sub)-column of processors 
 *
 * \param[in,out] A m-by-b dense tall-skinny matrix [tree_Y\R] on output
 * \param[in] lda_A lda of A
 * \param[in,out] Qm m-by-b dense tall-skinny matrix [Q1-S; Q2] on output
 * \param[in] lda_Qm lda of Qm
 * \param[in] m number of rows in A
 * \param[in] b number of columns in A
 * \param[in,out] W b-by-b matrix (must be preallocated) containing (Q1-S)^-1
 * \param[in] myRank rank in communicator column
 * \param[in] numPes number of processes in column
 * \param[in] root the root of the tree (who will own R at the end)
 * \param[in] req_id request id to use for send/recv
 * \param[in] cdt MPI communicator for column
 **/
void Yamamoto(double *   A,
              int64_t    lda_A,
              double *   Qm,
              int64_t    lda_Qm,
              int64_t    m,
              int64_t    b,
              double *   W,
              int64_t    myRank,
              int64_t    numPes,
              int64_t    root,
              int64_t    req_id,
              CommData_t cdt){
  int64_t mb;
  int info;
  double * R, * tau, * tree_data;

  mb = ((m/b)/numPes)*b;
  if ((m/b) % numPes > (myRank + numPes - root) % numPes) mb+=b;

  R             = (double*)malloc(sizeof(double)*b*b);
  tau           = (double*)malloc(sizeof(double)*b);
  tree_data     = (double*)malloc(sizeof(double)*2*b*b*(log(numPes)+2));


  TAU_FSTART(TSQR);
#ifdef BUTTERFLY_QR
  butterfly_tsqr(A, lda_A, tau, m, b, myRank, numPes, root, req_id, cdt, tree_data);
#else
  bitree_tsqr(A, lda_A, R, tau, m, b, myRank, numPes, root, req_id, cdt, 1, tree_data);
  if (myRank == root)
    copy_upper(R, A, b, b, lda_A, 0);
#endif
  MPI_Barrier(cdt.cm);
  TAU_FSTOP(TSQR);

  MPI_Barrier(cdt.cm);
  TAU_FSTART(Construct_Q1);
#ifdef BUTTERFLY_QR
  butterfly_construct_Q1(A, lda_A, tau, Qm, lda_Qm, m, b,  myRank, numPes, root, cdt, tree_data);
  //construct_Q1(A, lda_A, tau, Q1, mb, m, b, b, myRank, numPes, root, cdt, tree_data);
#else
  construct_Q1(A, lda_A, tau, Qm, lda_Qm, m, b, b, myRank, numPes, root, cdt, tree_data);
#endif
  
  MPI_Barrier(cdt.cm);
  TAU_FSTOP(Construct_Q1);
  if (myRank == root){
    TAU_FSTART(LU_of_Q1_minus_I);
    int * signs = (int*)malloc(sizeof(int)*b);
    int * pivs = (int*)malloc(sizeof(int)*b);
    double * wk = (double*)malloc(sizeof(double)*4*b);
    //compute LU(Q-S)
    lda_cpy(b, b, lda_Qm, b, Qm, W);
    recursive_YLU(W, b, b, signs);
    for (int i=0; i<b; i++){
      pivs[i] = i+1;
    }
    cdgetri(b, W, b, pivs, wk, 4*b, &info);
    for (int i=0; i<b; i++){
      Qm[i+lda_Qm*i] -= signs[i];
      //both seem to work
    //  cdscal(b,(double)signs[i],W+i,b);
      cdscal(b,(double)signs[i],W+i*b,1);
      cdscal(b-i,(double)signs[i],A+i+i*lda_A,lda_A);
    }
    free(wk);
    free(pivs);
    free(signs);
    TAU_FSTOP(LU_of_Q1_minus_I);
  }

  MPI_Bcast(W, b*b, MPI_DOUBLE, root, cdt.cm);
 
  free(R);
  free(tau);
  free(tree_data);
}


