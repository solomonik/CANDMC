#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "../../shared/util.h"
#include "bitree_tsqr.h"


/**
 * \brief Perform TSQR on a matrix of two stacked upper-trinagular Rs
 *
 * \param[in,out] A 2b-by-b [R1; R2] tall-skinny matrix, Y\R on output
 * \param[in,out] buf 2b*b sized buffer
 * \param[in] b number of columns in A
 * \param[in] tau if not NULL put T matrix here
 **/
void tree_tsqr(double *      A,
               double *      buf,
               int64_t const b,
               double *      tau){
  int info, alc;
/*  double * tau2;

  tau2 = (double*)malloc(b*b*sizeof(double));*/

  cdtpqrt(b, b, b, MIN(b,TAU_BLK), A, 2*b, A+b, 2*b, tau, MIN(b,TAU_BLK), buf, &info);
  assert(info == 0);
/*  if (tau != NULL)
    pack_upper(tau2, tau, b, b);*/
//  memcpy(tau, tau2, b*b*sizeof(double));
}



/**
 * \brief Perform TSQR over a (sub)-column of processors 
 *
 * \param[in,out] A m-by-b dense tall-skinny matrix, Y\R on output
 * \param[in,out] buf mb-by-b buffer
 * \param[in] m number of rows in A
 * \param[in] b number of columns in A
 * \param[in] lda leading dimension (number of buffer rows) in A
 * \param[in] tau if not NULL put T matrix here
 **/
void local_tsqr(double *      A,
                double *      buf,
                int64_t const m,
                int64_t const b,
                int64_t const lda_A,
                double *      tau){
  int info, alc;

  alc = 0;
  if (tau == NULL){
    tau = (double*)malloc(b*sizeof(double));
    alc = 1;
  }

  cdgeqrf(m, b, A, lda_A, tau, buf, m*b, &info);

  assert(info == 0);

/*  pack_upper(A, R, b, lda_A);*/

  if (alc){
    free(tau);
  }
}

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
 * \param[in] output_Y if 1 form up a matrix of Y factors in place of A
 * \param[in] tree_data TAU and Y data for the TSQR tree, 
 *            must be of size ((log2(p)+1)*b+b)/2-by-b
 **/
void bitree_tsqr( double *      A,
                  int64_t const lda_A,
                  double *      R,
                  double *      tau,
                  int64_t const m,
                  int64_t const b,
                  int64_t const myRank,
                  int64_t const numPes,
                  int64_t const root,
                  int64_t const req_id,
                  CommData_t    cdt,
                  int64_t const output_Y,
                  double *      tree_data){
  int64_t comm_pe, np, myr, mb, i, info, np_work;
  double * R_buf, * S_buf, * work;

//  mb = (m+root*b)/numPes;
//  if (myRank < root) mb-=b;

  mb = ((m/b)/numPes)*b;
  if ((m/b) % numPes > (myRank + numPes - root) % numPes) mb+=b;

  if (mb <= 0){
    return;
  } else {
    /* Adjust the root of the tree by a cyclic rotation of the processor
        grid column */
    myr = myRank-root;
    if (myr < 0)
      myr += numPes;
  }
  if (numPes * b > m)
    np_work = m/b;
  else 
    np_work = numPes;

  work = (double*)malloc(sizeof(double)*mb*b);
  // exit early if only one process involved
  if (np_work == 1){
    R_buf = (double*)malloc(sizeof(double)*m*b);
    lda_cpy(mb, b, lda_A, mb, A, R_buf);
    TAU_FSTART(Local_panel_TSQR);
    local_tsqr(R_buf, work, m, b, m, tau);
    pack_upper(R_buf, R, b, m);
    TAU_FSTOP(Local_panel_TSQR);
    if(output_Y){
      //Copy R_buf back to A
      lda_cpy(mb, b, mb, lda_A, R_buf, A);
    }
    memcpy(R_buf, R, psz_upr(b)*sizeof(double));
    std::fill(R, R+b*b, 0.0);
    unpack_upper(R_buf, R, b, b);
    free(R_buf);
    free(work);
    return;
  }

  if (output_Y)
    S_buf = (double*)malloc(sizeof(double)*b*b);

  // compute the first R on each m-by-b starting block
  // FIXME: Can we do without this buffer in some cases?
  //R_buf = (double*)malloc(sizeof(double)*MAX(2*b,mb)*b);
  //lda_cpy(mb, b, lda_A, mb, A, R_buf);
  TAU_FSTART(Local_panel_TSQR);
  local_tsqr(A, work, mb, b, lda_A, tau);
  pack_upper(A, R, b, lda_A);
  TAU_FSTOP(Local_panel_TSQR);

  /*if(output_Y){
    //Copy R_buf back to A
    lda_cpy(mb, b, mb, lda_A, R_buf, A);
  }*/

  /*if (mb != 2*b){
    free(R_buf);
  }*/
  R_buf = (double*)malloc(sizeof(double)*2*b*b);

  /* Tournament tree is a butterfly */
  TAU_FSTART(TSQR_tree);
  for (np = np_work; np > 1; np = np/2+(np%2)){

    /* If I am in second half of processor list send my data to lower half */
    if ((myr > np/2 || myr*2 == np)  && myr < np ){
      comm_pe = myr-(np+1)/2;
      comm_pe = comm_pe + root;
      if (comm_pe >= numPes)
        comm_pe = comm_pe - numPes;

      if ((np%2 == 0 || myr != np/2)){
        MPI_Send(R, psz_upr(b), MPI_DOUBLE, comm_pe, req_id, cdt.cm);

        if(tree_data == NULL && output_Y){
          //Wait until I receive my Householder Reflectors
          MPI_Status stat;
          MPI_Recv(R_buf, psz_upr(b), MPI_DOUBLE, comm_pe, req_id, cdt.cm, &stat);
          //now I should copy upper part of R in Upper part (A)
          unpack_upper(R_buf, A, b, lda_A);
        }
      }

    }
    else if (myr < np/2 && myr >= 0) {
      TAU_FSTART(tsqr_tree_inner);
      std::fill(R_buf, R_buf+2*b*b, 0.0);
      comm_pe = myr+(np+1)/2;
      comm_pe = comm_pe + root;
      if (comm_pe >= numPes)
        comm_pe = comm_pe - numPes;
      
      unpack_upper(R, R_buf, b, 2*b);

      /* put received R into the lower half of the buffer */
      MPI_Status stat;
      MPI_Recv(R, psz_upr(b), MPI_DOUBLE, comm_pe, req_id, cdt.cm, &stat);

      unpack_upper(R, R_buf+b, b, 2*b);
      TAU_FSTART(tsqr_tree_work);
      if (tree_data == NULL)
        local_tsqr(R_buf, work, 2*b, b, 2*b);
      else
        tree_tsqr(R_buf, work, b, tree_data);
      if (tree_data != NULL){
        tree_data += b*MIN(b,TAU_BLK);//psz_upr(b);
        pack_upper(R_buf+b,tree_data, b, 2*b);
        tree_data += psz_upr(b);
      }
      TAU_FSTOP(tsqr_tree_work);

      pack_upper(R_buf, R, b, 2*b);

      if(tree_data == NULL && output_Y){
        pack_upper(R_buf, R, b, 2*b);
        unpack_upper(R, A, b, lda_A);
        pack_upper(R_buf+b,S_buf, b, 2*b);
        //send lower part of R_buf to the other processor
        MPI_Send(S_buf, psz_upr(b), MPI_DOUBLE, comm_pe, req_id, cdt.cm);
      }
      TAU_FSTOP(tsqr_tree_inner);
    }
  }   
  TAU_FSTOP(TSQR_tree);

  if (output_Y)
    free(S_buf);

  if (myr == 0){
    memcpy(R_buf, R, psz_upr(b)*sizeof(double));
    std::fill(R, R+b*b, 0.0);
    unpack_upper(R_buf, R, b, b);
  }

  if (myr >= 0){
    free(R_buf);
  }
  free(work);
}



