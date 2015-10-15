/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "../../shared/util.h"
#include "bitree_tsqr.h"
#include "butterfly_tsqr.h"
#include "mpi.h"

/**
 * \brief Perform TSQR over a (sub)-column of processors 
 *
 * \param[in,out] A m-by-b dense tall-skinny matrix
 * \param[in] lda_A lda of A
 * \param[in,out] R b-by-b upper-triangular matrix of, for which A=QR
 * \param[in,out] tau length b vector of tau values for the first tree level
 * \param[in] m number of rows in A
 * \param[in] b number of columns in A
 * \param[in] myRank rank in communicator column
 * \param[in] numPes number of processes in column
 * \param[in] root the root of the tree (who will own R at the end)
 * \param[in] req_id request id to use for send/recv
 * \param[in] cdt MPI communicator for column
 * \param[in] tree_data TAU and Y data for the TSQR tree, 
 *            must be of size ((log2(p)+1)*b+b)/2-by-b
 **/
void butterfly_tsqr(double *      A,
                    int64_t const lda_A,
                    double *      tau,
                    int64_t const m,
                    int64_t const b,
                    int64_t const myRank,
                    int64_t const numPes,
                    int64_t const root,
                    int64_t const req_id,
                    CommData_t    cdt,
                    double *      tree_data){
  int64_t comm_pe, mb, i, info, np_work;
  double * R_buf, * S_buf, * work;

  mb = ((m/b)/numPes)*b;
  if ((m/b) % numPes > (myRank + numPes - root) % numPes) mb+=b;
 
  /* This is not an invarient butterfly, there is an up */
  int myr = myRank-root;
  if (myr < 0)
    myr += numPes;

  /* determine the number of processors that have a block involved */
  if (numPes * b > m){
    np_work = m/b;
  } else 
    np_work = numPes;
    
  /* find the smallest power of two less than or equal to number of working processors */
  int inp = np_work;
  int pow2_np_work = 1;
  while (inp > 1){
    pow2_np_work = pow2_np_work*2;
    inp = inp/2;
  }
  assert(pow2_np_work <= np_work);

  work = (double*)malloc(sizeof(double)*mb*b);
  // exit early if only one process involved
  if (np_work == 1){
    /*R_buf = (double*)malloc(sizeof(double)*m*b);
    lda_cpy(mb, b, lda_A, mb, A, R_buf);*/
    if (myr == 0){
      TAU_FSTART(Local_panel_TSQR);
      local_tsqr(A, work, m, b, lda_A, tau);
      TAU_FSTOP(Local_panel_TSQR);
    }
//    free(R_buf);

    return;
  }

  // compute the first R on each m-by-b starting block
  R_buf = (double*)malloc(sizeof(double)*2*b*b);
  double * R = (double*)malloc(sizeof(double)*b*b);

  if (myr < np_work){
    TAU_FSTART(Local_panel_TSQR);
    local_tsqr(A, work, mb, b, lda_A, tau);
    pack_upper(A, R, b, lda_A);
    TAU_FSTOP(Local_panel_TSQR);
  }
  double * R_recv_buf = (double*)malloc(sizeof(double)*b*b);

  // if processor count now power of two do one clipped butterfly level
  TAU_FSTART(TSQR_clipped_wing);
  if (np_work > pow2_np_work){
    if ((myr >= pow2_np_work && myr < np_work) || (myr < np_work - pow2_np_work)){
      int parity = (myr>=pow2_np_work);
      int vcomm_pe = myr - parity*pow2_np_work + (1-parity)*pow2_np_work;
      int comm_pe = (vcomm_pe + root)%numPes;
      MPI_Status stat;
      MPI_Sendrecv(R,          psz_upr(b), MPI_DOUBLE, comm_pe, req_id+parity,
                   R_recv_buf, psz_upr(b), MPI_DOUBLE, comm_pe, req_id+1-parity, 
                   cdt.cm, &stat);
        
      unpack_upper(R, R_buf+parity*b, b, 2*b);
      unpack_upper(R_recv_buf, R_buf+(1-parity)*b, b, 2*b);

      tree_tsqr(R_buf, work, b, tree_data);
      tree_data += b*MIN(b,TAU_BLK);
      pack_upper(R_buf+b,tree_data, b, 2*b);
      tree_data += psz_upr(b);
      pack_upper(R_buf, R, b, 2*b);
    }
  }
#ifdef PROFILE
  MPI_Barrier(cdt.cm);
#endif
  TAU_FSTOP(TSQR_clipped_wing);
  
  TAU_FSTART(TSQR_butterfly);
  if (pow2_np_work > 1 && myr < pow2_np_work){
    /* Tournament tree is a butterfly */
    for (int level=pow2_np_work; level>1; level=level/2){
      /* parity determines which buttefly wing this proc is on */
      int parity = (myr%level)>=level/2;
      /* comm_pe finds the other wing */
      int vcomm_pe = level*(myr/level) + (((myr%level)+(level/2))%level);
      int comm_pe = (vcomm_pe + root)%numPes;

      MPI_Status stat;
      MPI_Sendrecv(R,          psz_upr(b), MPI_DOUBLE, comm_pe, req_id+parity,
                   R_recv_buf, psz_upr(b), MPI_DOUBLE, comm_pe, req_id+1-parity, 
                   cdt.cm, &stat);
        
      unpack_upper(R, R_buf+parity*b, b, 2*b);
      unpack_upper(R_recv_buf, R_buf+(1-parity)*b, b, 2*b);

      tree_tsqr(R_buf, work, b, tree_data);
      tree_data += b*MIN(b,TAU_BLK);
      pack_upper(R_buf+b,tree_data, b, 2*b);
      tree_data += psz_upr(b);
      pack_upper(R_buf, R, b, 2*b);
    }
  }
#ifdef PROFILE
  MPI_Barrier(cdt.cm);
#endif
  TAU_FSTOP(TSQR_butterfly);
  if (myRank == root){
    copy_upper(R_buf, A, b, 2*b, lda_A, 0);
  }

  free(R);
  free(R_buf);
  free(R_recv_buf);
}



