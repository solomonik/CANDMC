/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "../../shared/util.h"
#include "../tsqr/bitree_tsqr.h"
#include "../tsqr/butterfly_tsqr.h"
#include "../hh_recon/hh_recon.h"

//#define USE_BINARY_TREE

/**
 * \param[in,out] A m-by-k dense matrix on input, YR where (I-YTY^T)R 
 * \param[in] lda_A lda of A
 * \param[in] m number of rows in A
 * \param[in] k number of columns in A
 * \param[in] b number of Householder vectors
 * \param[in] myRank my global processorrank
 * \param[in] numPes number of processors 
 * \param[in] root_row current row root
 * \param[in] root_col current column root
 * \param[in] cdt_row MPI communicator for row
 * \param[in] cdt_col MPI communicator for column
 * \param[in] cdt_world MPI communicator for world
 * \param[in] stop_at if negative ignored, if positive, stop QR after this many
 *            cols/rows
 **/
void QR_butterfly_2D( double *        A,
                      int64_t const   lda_A,
                      int64_t const   m,
                      int64_t const   k,
                      int64_t const   b,
                      int64_t const   myRank,
                      int64_t const   numPes,
                      int64_t const   root_row,
                      int64_t const   root_col,
                      CommData_t      cdt_row,
                      CommData_t      cdt_col,
                      CommData_t      cdt_world,
                      int64_t const   _stop_at){
  int64_t i, j, pe_st_new, move_ptr;
  double * R, * Y, * tau, * tree_data;
  int64_t mb = (m+root_row*b)/cdt_col.np;
  if (cdt_col.rank < root_row) mb-=b;
  int64_t kb = (k+root_col*b)/cdt_row.np;
  if (cdt_row.rank <= root_col) kb-=b;
  int64_t tdsz = (psz_upr(b) + MIN(b,TAU_BLK)*b)*(log2(cdt_col.np)+2);

  int64_t stop_at;
  if (_stop_at >= 0){
    stop_at = MIN(MIN(m,k),_stop_at);
  } else {
    stop_at = MIN(m,k);
  }

  R = (double*)malloc(sizeof(double)*b*b);
  Y = (double*)malloc(sizeof(double)*mb*b);
  tau = (double*)malloc(sizeof(double)*b);
  tree_data = (double*)malloc(sizeof(double)*tdsz);
  /* TSQR on block column */
  TAU_FSTART(TSQR);
  if (cdt_row.rank == root_col){
#ifndef USE_BINARY_TREE
    if ( kb < cdt_col.np){
      //if (cdt_col.rank == 0) printf("Binary tree\n");
      bitree_tsqr(A, lda_A, R, tau, m, MIN(b,stop_at), cdt_col.rank, cdt_col.np, 
                  root_row, 0, cdt_col, 1, tree_data);
      if (cdt_col.rank == root_row)
        copy_upper(R, A, b, b, lda_A, 0);
    } else {
      //if (cdt_col.rank == 0) printf("Butterfly tree\n");
      butterfly_tsqr(A, lda_A, tau, m, MIN(b,stop_at), cdt_col.rank, cdt_col.np, 
                     root_row, 13, cdt_col, tree_data);
    }
#else
    bitree_tsqr(A, lda_A, R, tau, m, MIN(b,stop_at), cdt_col.rank, cdt_col.np, 
                root_row, 0, cdt_col, 1, tree_data);
    if (cdt_col.rank == root_row)
      copy_upper(R, A, b, b, lda_A, 0);
#endif
  }
  /* Iterate over panels */
  if (m-b>0 || k-b>0){
    if (cdt_row.rank == root_col)
      lda_cpy(mb, b, lda_A, mb, A, Y);
    MPI_Barrier(cdt_world.cm);
    TAU_FSTOP(TSQR);
    TAU_FSTART(Bcast_update);
    MPI_Bcast(Y, mb*b, MPI_DOUBLE, root_col, cdt_row.cm);
    MPI_Bcast(tree_data, tdsz, MPI_DOUBLE, root_col, cdt_row.cm);
    MPI_Bcast(tau, b, MPI_DOUBLE, root_col, cdt_row.cm);
    TAU_FSTOP(Bcast_update);
    move_ptr = 0;
    /* Update 2D distributed matrix */
/*    if (myRank == 0) printf("before apply Q:\n");
    double * A_ptr = A;
    for (int rr=0; rr<m/b; rr++){
      if (cdt_world.rank == ((rr+root_row)%cdt_world.np)){
        printf("[%d] ",rr); 
        print_matrix(A_ptr+b*lda_A, b, kb-b, lda_A);
        A_ptr+=b;
      }
      fflush(stdout);
      MPI_Barrier(cdt_world.cm);
    }*/
    if (cdt_row.rank == root_col){
      move_ptr = b*lda_A;
    }
    TAU_FSTART(schur_tree_Q);
    if (kb > 0){
#ifndef USE_BINARY_TREE
      if ( kb < cdt_col.np)
        apply_tsqr_QT(Y, mb, tau, A+move_ptr, lda_A, m, b, kb, cdt_col.rank, cdt_col.np, root_row, cdt_col, tree_data);
      else
        apply_butterfly_tsqr_QT(Y, mb, tau, A+move_ptr, lda_A, m, b, kb, cdt_col.rank, cdt_col.np, root_row, cdt_col, tree_data);
#else
      apply_tsqr_QT(Y, mb, tau, A+move_ptr, lda_A, m, b, kb, cdt_col.rank, cdt_col.np, root_row, cdt_col, tree_data);
#endif
    }
/*    if (myRank == 0) printf("after apply Q:\n");
    A_ptr = A;
    for (int rr=0; rr<m/b; rr++){
      if (cdt_world.rank == ((rr+root_row)%cdt_world.np)){
        printf("[%d] ",rr); 
        print_matrix(A_ptr+b*lda_A, b, kb-b, lda_A);
        A_ptr+=b;
      }
      fflush(stdout);
      MPI_Barrier(cdt_world.cm);
    }
    MPI_Barrier(cdt_world.cm);*/
    TAU_FSTOP(schur_tree_Q);
    if (cdt_col.rank == root_row)
      move_ptr += b;
    
    /* Recurse into the next step */
    if (b<stop_at)
      QR_butterfly_2D( A+move_ptr, lda_A, 
                  m-b, k-b, b, myRank, numPes,
                  (root_row+1) % cdt_col.np,
                  (root_col+1) % cdt_row.np,
                  cdt_row, cdt_col, cdt_world, stop_at-b);

  }
}  

