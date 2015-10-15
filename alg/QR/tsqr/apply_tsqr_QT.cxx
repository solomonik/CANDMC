/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "../../shared/util.h"
#include "bitree_tsqr.h"


/**
 * \brief Apply Q^T to B, where Q^T is represented implicitly by Y and
 * tree_data, obtained by running binary tree TSQR
 *
 * \param[in] Y m-by-b matrix of Householder vectors from TSQR
 * \param[in] lda_Y lda of Y
 * \param[in] tau TAU values associated with the first level of TSQR HH vecs
 * \param[in,out] B m-by-k matrix to apply QT to
 * \param[in] lda_B length of leading dimension of B
 * \param[in] m number of rows in Y
 * \param[in] b number of columns in Y
 * \param[in] k number of columns of B to apply Y to
 * \param[in] myRank rank in communicator column
 * \param[in] numPes number of processes in column
 * \param[in] root the root of the tree (who will own R at the end)
 * \param[in] cdt MPI communicator for column
 * \param[in] tree_data TAU and Y data for the TSQR tree, 
 *                must be of size ((log2(p)+1)*b+b)/2-by-b
 **/
void apply_tsqr_QT( double const *  Y,
                    int64_t const   lda_Y,
                    double const *  tau,
                    double *        B,
                    int64_t const   lda_B,
                    int64_t const   m,
                    int64_t const   b,
                    int64_t const   k,
                    int64_t const   myRank,
                    int64_t const   numPes,
                    int64_t const   root,
                    CommData_t      cdt,
                    double *        tree_data){
  int req_id = 0;
  MPI_Request req;
  MPI_Status stat;
  int info;
  int64_t comm_pe, np, myr, offset, mb, bmb, i, buf_sz, j, np_work, coff;
  double * T, * Y2, * buffer, * B_buf;
  
  mb = (m+root*b)/numPes;
  bmb = (m+root*b)/numPes;
  if (myRank < root){
    offset = (numPes-root)*mb+myRank*(mb-b);
    mb-=b;
  } else
    offset = (myRank-root)*mb;

  if (mb <= 0){
    return;
  } else {
    myr = myRank-root;
    if (myr < 0)
      myr += numPes;
  }
  if (numPes * b > m)
    np_work = m/b;
  else 
    np_work = numPes;
  buf_sz = mb*MAX(k,b);
  
  buffer = (double*)malloc(sizeof(double)*buf_sz);

  // exit early if only one process involved
  if (np_work == 1){
    TAU_FSTART(apply_tsqr_QT_local);
    if (myr >= 0){
      cdormqr('L', 'T', m, k, b, Y, lda_Y, tau, B, lda_B, buffer, buf_sz, &info);
    }
    TAU_FSTOP(apply_tsqr_QT_local);
    free(buffer);
    return;
  }
  
  Y2 = (double*)malloc(sizeof(double)*b*b*2);
  B_buf = (double*)malloc(sizeof(double)*b*k*2);
  T = (double*)malloc(sizeof(double)*b*b);
  std::fill(Y2, Y2+b*b*2, 0.0);
  
  TAU_FSTART(apply_tsqr_QT_local);
  if (myr >= 0){
    //cdormqr('L', 'T', mb, k, b, Y, lda_Y, tau, B, lda_B, buffer, buf_sz, &info);
    cdlarft('F', 'C', mb, b, Y, lda_Y, tau, T, b);
    cdlarfb('L', 'T', 'F', 'C', mb, k, b, Y, lda_Y, T, b, B, lda_B, buffer, k);
  }
  TAU_FSTOP(apply_tsqr_QT_local);

  lda_cpy(b,k,lda_B,2*b,B,B_buf);


  TAU_FSTART(apply_tsqr_QT_tree);
  for (np = np_work; np > 1; np = np/2+(np%2)){
    /* If I am in second half of processor list send my data to lower half */
    if ((myr > np/2 || myr*2 == np)  && myr < np ){
      comm_pe = myr-(np+1)/2;
      comm_pe = comm_pe + root;
      if (comm_pe >= numPes)
        comm_pe = comm_pe - numPes;

      if ((np%2 == 0 || myr != np/2)){
        if (tree_data == NULL){
          pack_upper(Y, buffer, b, lda_Y);
          MPI_Send(buffer, psz_upr(b), MPI_DOUBLE, comm_pe, req_id, cdt.cm);
        }
        lda_cpy(b, k, 2*b, b, B_buf, buffer);
        MPI_Send(buffer, k*b, MPI_DOUBLE, comm_pe, req_id, cdt.cm);
        MPI_Irecv(buffer, k*b, MPI_DOUBLE, comm_pe, req_id, cdt.cm, &req);
        MPI_Wait(&req, &stat);
        lda_cpy(b, k, b, 2*b, buffer, B_buf);
      }
    } else if (myr < np/2 && myr >= 0) {
      TAU_FSTART(ctQ_tree_worker);
      comm_pe = myr+(np+1)/2;
      comm_pe = comm_pe + root;
      if (comm_pe >= numPes){
        comm_pe = comm_pe - numPes;
        coff = comm_pe*(bmb-b)+(numPes-root)*bmb;
      } else
        coff = (comm_pe-root)*bmb;
      MPI_Irecv(buffer, k*b, MPI_DOUBLE, comm_pe, req_id, cdt.cm, &req);
      MPI_Wait(&req, &stat);
      lda_cpy(b, k, b, 2*b, buffer, B_buf+b);
      if (tree_data == NULL){
        MPI_Irecv(buffer, psz_upr(b), MPI_DOUBLE, comm_pe, req_id, cdt.cm, &req);
        MPI_Wait(&req, &stat);
        unpack_upper(buffer, Y2+b, b, 2*b);
        tau_recon('U', b, b, 2*b, Y2+b, T);
        cdormqr('L', 'N', 2*b, k, b, Y2, 2*b, T, B_buf, 2*b, 
                buffer, buf_sz, &info);
      } else {
        memcpy(T, tree_data, b*MIN(b,TAU_BLK)*sizeof(double));
        tree_data += b*MIN(b,TAU_BLK);
        unpack_upper(tree_data, Y2+b, b, 2*b);
        tree_data += psz_upr(b);
        TAU_FSTART(cdtpmqrt);
        cdtpmqrt('L', 'T', b, k, b, b, MIN(b,TAU_BLK), Y2+b, 2*b, T, MIN(b,TAU_BLK), B_buf, 2*b,
                     B_buf+b, 2*b, buffer, &info);
        TAU_FSTOP(cdtpmqrt);
      }
      lda_cpy(b,k,2*b,b,B_buf+b,buffer);
      MPI_Send(buffer, k*b, MPI_DOUBLE, comm_pe, req_id, cdt.cm);
      TAU_FSTOP(ctQ_tree_worker);
    }
  }
  TAU_FSTOP(apply_tsqr_QT_tree);
  lda_cpy(b,k,2*b,lda_B,B_buf,B);


  free(T);
  free(B_buf);
  free(buffer);
  free(Y2);
}


