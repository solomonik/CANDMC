/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "../../shared/util.h"
#include "../tsqr/bitree_tsqr.h"
#include "../tsqr/butterfly_tsqr.h"
#include "hh_recon.h"

#define BSIZE 64

//#define BUTTERFLY_QR
//#define CARRY_UINV

/**
 * \brief perform sequential b-by-b 
 *        TRSM to compute invT from W matrix (output of hh_recon QR)
 * \param[in] W b-by-b triangular factor -T*Y1'
 * \param[in] b dimension of W and T
 * \param[in,out] invT preallcative space for T^-1
 */

void compute_invT_from_W(double const * W,
                         int64_t        b, 
                         double *       invT){
  /* solve W^T T^-T = -Y1*/
  cdtrsm( 'L', 'U', 'T', 'N', b, b, -1.0, W, b, invT, b );
}

/**
 * \brief Perform signed LU
 *
 * \param[in,out] A b-by-b dense square matrix, L\U on output
 * \param[in,out] R b-by-b upper-triangular matrix, gets multiplied by signs
 * \param[in] b number of rows/columns in A
 * \param[in] lda_A leading dimension (number of buffer rows) in A
 * \param[in] lda_R leading dimension (number of buffer rows) in R
 * \param[out] signs of R, filled if not NULL
 **/
void signed_NLU(double *      A,
                double *      R,
                int64_t       b,
                int64_t       lda_A,
                int64_t       lda_R,
                int64_t *     signs){
  int64_t info, i, j, pos;
 
  for (i=0; i<b; i++){
    if (A[i*lda_A+i] > 0)
      signs[i] = -1;
    else 
      signs[i] = 1;
    A[i*lda_A+i] -= 1*signs[i];
    if (signs[i] == -1){
      cdscal(b-i, -1, R+i*lda_R+i, lda_R);
    } 

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
 * \param[in,out] R b-by-b upper-triangular matrix, gets multiplied by signs
 * \param[in] b number of rows/columns in A
 * \param[in] lda_A leading dimension (number of buffer rows) in A
 * \param[in] lda_R leading dimension (number of buffer rows) in R
 * \param[in] signs of R rows to use
 **/
void recursive_NLU( double *      A,
                    double *      R,
                    int64_t       b,
                    int64_t       lda_A,
                    int64_t       lda_R,
                    int64_t *     signs){
  int64_t i, b1, b2;
  int info;
 
  if (b<=BSIZE){
    signed_NLU(A, R, b, lda_A, lda_R, signs);
    return;
  }
  
  b1 = b/2;
  b2 = b-b1;

  recursive_NLU(A, R, b1, lda_A, lda_R, signs);
 
  for (i=0; i<b1; i++){
    if (signs[i] == -1)
      cdscal(b2, -1.0, R+b1*lda_R+i, lda_R);
  //  cdaxpy(b2, -1.0, R+b1*lda_R+i, lda_R, A+b1*lda_A+i, lda_A);
  }

  cdtrsm('L', 'L', 'N', 'U', b1, b2, 1.0, A, lda_A, A+b1*lda_A, lda_A);
  cdtrsm('R', 'U', 'N', 'N', b2, b1, 1.0, A, lda_A, A+b1,       lda_A);

  cdgemm('N', 'N', b2, b2, b1, -1.0, A+b1, lda_A, A+b1*lda_A, lda_A, 1.0, A+b1*(lda_A+1), lda_A);
  recursive_NLU(A+b1*(lda_A+1), R+b1*(lda_R+1), b2, lda_A, lda_R, signs+b1);
}



/**
 * \brief Perform TSQR and reconstruct YT on a (sub)-column of processors 
 *
 * \param[in,out] A m-by-b dense tall-skinny matrix, Y\R on output
 * \param[in] lda_A lda of A
 * \param[in] m number of rows in A
 * \param[in] b number of columns in A
 * \param[in,out] W b-by-b upper triangular matrix -T*Y1'
 * \param[in,out] preallocated buffer for upper triangular W matrix
 * \param[in] myRank rank in communicator column
 * \param[in] numPes number of processes in column
 * \param[in] root the root of the tree (who will own R at the end)
 * \param[in] req_id request id to use for send/recv
 * \param[in] cdt MPI communicator for column
 **/
void hh_recon_qr(double *       A,
                 int64_t        lda_A,
                 int64_t        m,
                 int64_t        b,
                 double *       W,
                 int64_t        myRank,
                 int64_t        numPes,
                 int64_t        root,
                 int64_t        req_id,
                 CommData_t     cdt){
  int64_t i, j, mb;
  int info;
  double * R, * Q1, * tau, * tree_data;
  int64_t * signs;

  mb = ((m/b)/numPes)*b;
  if ((m/b) % numPes > (myRank + numPes - root) % numPes) mb+=b;

  R = (double*)malloc(sizeof(double)*b*b);
  double * Rbuf = (double*)malloc(sizeof(double)*psz_upr(b));
  Q1 = (double*)malloc(sizeof(double)*mb*b);
  tau = (double*)malloc(sizeof(double)*b);
  tree_data = (double*)malloc(sizeof(double)*2*b*b*(log(numPes)+2));
  TAU_FSTART(TSQR);
#ifdef CARRY_UINV
  double * Y12Uinv = (double*)malloc(sizeof(double)*b*b);
  int croot = (root+1)%numPes;
  if (numPes == 1 || m == b){
    bitree_tsqr(A, lda_A, R, tau, m, b, myRank, numPes, root, req_id, cdt, 1, tree_data);
  } else {
    if (myRank == root){
      //TSQR on n-b by b minor of A
      bitree_tsqr(A+b, lda_A, R, tau, m-b, b, myRank, numPes, croot, req_id, cdt, 1, tree_data);
      MPI_Recv(Rbuf, psz_upr(b), MPI_DOUBLE, croot, req_id, cdt.cm, MPI_STATUS_IGNORE);
      std::fill(R,R+b*b,0.0);
      unpack_upper(Rbuf, R, b, b);
      double * toptau = (double*)malloc(sizeof(double)*b);
      double * topT = (double*)malloc(sizeof(double)*b*b);
      double * Uinv = (double*)malloc(sizeof(double)*b*b);
      double * buf = (double*)malloc(sizeof(double)*b*b);
      //QR on top b-by-b block [Y1,R]=QR(A1)
      cdgeqrf(b, b, A, lda_A, toptau, buf, b*b, &info);
      //TSQR on [R; TSQR_R] forming dense T, Y12
      //std::fill(topT,topT+b*b,0.0);
      cdtpqrt(b, b, 0, b, A, lda_A, R, b, topT, b, buf, &info);
      //Uinv = T-I
      copy_upper(topT, Uinv, b, b, b, 1);
      for (i=0; i<b; i++){
      //  cdscal(i, -1., Uinv+i*b, 1);
        Uinv[i*b+i] = Uinv[i*b+i] -1.;
      }
      //Uinv := Q1 = dormqr(Y1,T-I)
      //cdormqr('L', 'N', b, b, b, A, lda_A, toptau, Uinv, b, buf, b*b, &info);
      int vecs, nb = TAU_BLK;//16;
      for (j=b; j>=0; j=j-b){
        vecs = b-j;
        cdormqr('L', 'N', b-j, b-j, vecs, A+lda_A*j+j, lda_A, toptau+j, Uinv+b*j+j, b, buf, b*b, &info);
      }
     
      signs = (int64_t*)malloc(sizeof(int64_t)*b);
      //Uinv := LU(I-dormqr(Y1,T-I));
      signed_NLU(Uinv, A, b, b, lda_A, signs);
      //write lower part of Uinv (L from LU) back to A
      copy_lower(Uinv, A, b, b, b, lda_A, 0);
      //W = uppertri(Uinv)*signs
      for (i=0; i<b; i++){
  //      if (signs[i] == -1)
        std::fill(Uinv+i*b+i+1, Uinv+(i+1)*b, 0.0);
        cdscal(i+1, -1.0, Uinv+i*b, 1);
        cdscal(i+1, -1.0, A+i*lda_A, 1);
      }
      copy_upper(Uinv, W, b, b, b, 0);
      for (i=0; i<b; i++){
        if (signs[i] == 1.0)
          cdscal(i+1, -1.0, W+i*b, 1);
      }
      //Uinv := inv(uppertri(LU(I-dormqr(Y1,T-I)))*signs)
      cdtrtri('U', 'N', b, Uinv, b, &info);
      //Y12Uinv=apply(Y12,[I*Uinv; 0])
      std::fill(Y12Uinv, Y12Uinv+b*b, 0.0);
      cdtpmqrt('L','N', b, b, b, 0, b, R, b, topT, b, Uinv, b, Y12Uinv, b, buf, &info);
        
      pack_upper(Y12Uinv, Rbuf, b, b);
      MPI_Send(Rbuf, psz_upr(b), MPI_DOUBLE, croot, req_id, cdt.cm);

      free(toptau);
      free(topT);
      free(Uinv);
      free(buf);
      free(signs);
    } else {
      bitree_tsqr(A, lda_A, R, tau, m-b, b, myRank, numPes, croot, req_id, cdt, 1, tree_data);
      if (myRank == croot){
        pack_upper(R, Rbuf, b, b);
        MPI_Send(Rbuf, psz_upr(b), MPI_DOUBLE, root, req_id, cdt.cm);
        MPI_Recv(Rbuf, psz_upr(b), MPI_DOUBLE, root, req_id, cdt.cm, MPI_STATUS_IGNORE);
        std::fill(Y12Uinv, Y12Uinv+b*b, 0.0);
        unpack_upper(Rbuf, Y12Uinv, b, b);
      }
    }
  }
#else
#ifdef BUTTERFLY_QR
  butterfly_tsqr(A, lda_A, tau, m, b, myRank, numPes, root, req_id, cdt, tree_data);
  copy_upper(A, R, b, lda_A, b, 1);
#else
  bitree_tsqr(A, lda_A, R, tau, m, b, myRank, numPes, root, req_id, cdt, 1, tree_data);
#endif
#endif
#ifdef PROFILE
  MPI_Barrier(cdt.cm);
#endif
  TAU_FSTOP(TSQR);


#ifdef PROFILE
  MPI_Barrier(cdt.cm);
#endif
  TAU_FSTART(Construct_Q1);
#ifdef CARRY_UINV
  if (numPes ==  1 || m == b){
    construct_Q1(A, lda_A, tau, Q1, mb, m, b, b, myRank, numPes, root, cdt, tree_data);
  } else {
    if (m-b > 0){
      if (myRank == root){
        construct_Q1(A+b, lda_A, tau, Q1, mb-b, m-b, b, b, myRank, numPes, croot, cdt, tree_data, Y12Uinv);
        lda_cpy(mb-b,b,mb-b,lda_A,Q1,A+b);
      } else {
        construct_Q1(A, lda_A, tau, Q1, mb, m-b, b, b, myRank, numPes, croot, cdt, tree_data, Y12Uinv);
        lda_cpy(mb,b,mb,lda_A,Q1,A);
      }
    }
  }
#else
#ifdef BUTTERFLY_QR
  butterfly_construct_Q1(A, lda_A, tau, Q1, mb, m, b,  myRank, numPes, root, cdt, tree_data);
  //construct_Q1(A, lda_A, tau, Q1, mb, m, b, b, myRank, numPes, root, cdt, tree_data);
#else
  construct_Q1(A, lda_A, tau, Q1, mb, m, b, b, myRank, numPes, root, cdt, tree_data);
#endif
#endif

#ifdef PROFILE
  MPI_Barrier(cdt.cm);
#endif
  TAU_FSTOP(Construct_Q1);
#ifdef CARRY_UINV
  if (numPes == 1 || m == b)
#endif
  {
    if (myRank == root){
      signs = (int64_t*)malloc(sizeof(int64_t)*b);
      TAU_FSTART(LU_of_Q1_minus_I);
      recursive_NLU(Q1, R, b, mb, b, signs);
      TAU_FSTOP(LU_of_Q1_minus_I);
      copy_upper(Q1, W, b, mb, b, 0);
    }
    if (myRank == root){
      copy_upper(R, A, b, b, lda_A, 0);
    }
    MPI_Barrier(cdt.cm);
    TAU_FSTART(Form_Y2);
    MPI_Bcast(W, b*b, MPI_DOUBLE, root, cdt.cm);
    if (myRank != root){
      if (mb > 0){
        cdtrsm('R', 'U', 'N', 'N', mb, b, 1.0, W, b, Q1, mb);
        lda_cpy(mb, b, mb, lda_A, Q1, A);
      }
    } else {
      if (mb-b > 0){
        cdtrsm('R', 'U', 'N', 'N', mb-b, b, 1.0, W, b, Q1+b, mb);
      }
      copy_lower(Q1, A, b, mb, mb, lda_A, 0);
    }
    TAU_FSTOP(Form_Y2);
    if (myRank == root){
      for (i=0; i<b; i++){
        if (signs[i] == -1)
          cdscal(i+1, -1.0, W+i*b, 1);
      }
      free(signs);
    }
  }

#ifdef CARRY_UINV
  free(Y12Uinv);
#endif
  free(R);
  free(Rbuf);
  free(Q1);
  free(tau);
  free(tree_data);
}


