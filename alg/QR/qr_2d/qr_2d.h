#ifndef __QR_2D_H__
#define __QR_QD_H__

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "../../shared/util.h"
#include "../tsqr/bitree_tsqr.h"
#include "../hh_recon/hh_recon.h"


/**
 * \brief computes triangular T from Y by dsyrk
 *
 * \param[in] Y Househodler vectors stored along root processor column, local size mb-by-b
 * \param[in] lda_Y lda of Y matrix
 * \param[in,out] T computed triangular T corresponding to Y
 * \param[in] lda_T lda of T matrix
 * \param[in] mb length of local piece of Householder vectors
 * \param[in] b number of Householder vectors
 * \param[in] pv current parallel view
 */
void compute_invT_from_Y(double const * Y,
                      int64_t        lda_Y,
                      double       * T,
                      int64_t        lda_T,
                      int64_t        mb,
                      int64_t        b,
                      pview *        pv);

/**
 * \brief computes rectangular T from Y1 and Y2 by gemm
 *
 * \param[in] Y1 Househodler vectors stored along root processor column, local size mb-by-b1
 * \param[in] lda_Y1 lda of Y1 matrix
 * \param[in] Y2 Househodler vectors stored along root processor column, localsize mb-by-b2
 * \param[in] lda_Y2 lda of Y2 matrix
 * \param[in,out] T computed b1-by-b2 T corresponding to Y1^TY2
 * \param[in] lda_T lda of T matrix
 * \param[in] mb length of local piece of Householder vectors
 * \param[in] b1 number of Householder vectors in Y1
 * \param[in] b2 number of Householder vectors in Y2
 * \param[in] pv current parallel view
 */
void compute_invT_from_Y(double const * Y1,
                      int64_t        lda_Y1,
                      double const * Y2,
                      int64_t        lda_Y2,
                      double       * T,
                      int64_t        lda_T,
                      int64_t        mb,
                      int64_t        b1,
                      int64_t        b2,
                      pview *        pv);

/**
 * \brief Perform (I-YTY^T)A 
 *
 * \param[in,out] Y m-by-k dense lower-triangular matrix of HH vecs
 * \param[in] lda_Y lda of Y
 * \param[in,out] A m-by-k dense matrix of HH vecs
 * \param[in] lda_A lda of A
 * \param[in] m number of rows in A
 * \param[in] k number of columns in A
 * \param[in] b number of Householder vectors
 * \param[in] W is either null or Y1^TT
 * \param[in] pv current processor grid view
 * \param[in] aggreg_Y is either null or a buffer into which we should save Ybuf 
 * \param[in] lda_aY lda of aggreg_Y if not null
 * \param[in] W_is_T whether W contains T
 **/
void update_A(double const *  Y,
              int64_t         lda_Y,
              double *        A,
              int64_t         lda_A,
              int64_t         m,
              int64_t         k,
              int64_t         b,
              double const *  W,
              pview *         pv,
              double *        aggreg_Y,
              int64_t         lda_aY,
              bool            W_is_T=false);

/**
 * \brief Perform (I-YTY^T)A  after bcasting Y
 *
 * \param[in,out] Ybuf m-by-k dense lower-triangular matrix of HH vecs owned on
 *                all columns of the 2D grid
 * \param[in,out] A m-by-k dense matrix of HH vecs
 * \param[in] lda_A lda of A
 * \param[in] mb number of rows I own in A
 * \param[in] kb number of columns I own in A
 * \param[in] b number of Householder vectors
 * \param[in] pv current processor grid view
 **/
void upd_A( double const *  Ybuf,
            int64_t         lda_Y,
            double *        A,
            int64_t         lda_A,
            int64_t         mb,
            int64_t         kb,
            int64_t         b,
            double const *  W,
            pview *         pv,
            bool            W_is_T=false);

/**
 * \brief Perform 2D QR with one level of blocking
 *
 * \param[in,out] A m-by-k dense matrix
 * \param[in] lda_A lda of A
 * \param[in] m number of rows in A
 * \param[in] k number of columns in A
 * \param[in] b number of Householder vectors
 * \param[in] W is either null or Y1^TT
 * \param[in] pv current processor grid view
 * \param[in] aggreg_Y is either null or a buffer into which we should aggregate
 *                      broadcasted panels
 * \param[in] lda_aY lda of aggreg_Y if not null
 **/
void QR_2D( double * A,
            int64_t  lda_A,
            int64_t  m,
            int64_t  k,
            int64_t  b,
            pview *  pv,
            double * aggreg_Y,
            int64_t  lda_aY);

/**
 * \brief Perform 2D QR with one level of blocking and with pipelining
 *
 * \param[in,out] A m-by-k dense matrix
 * \param[in] lda_A lda of A
 * \param[in] m number of rows in A
 * \param[in] k number of columns in A
 * \param[in] b number of Householder vectors
 * \param[in] W is either null or Y1^TT
 * \param[in] pv current processor grid view
 * \param[in] last_Y is the Y broadcast at the last step or is null
 * \param[in] lda_lY lda of last_Y if not null
 **/
void QR_2D_pipe(double * A,
                int64_t  lda_A,
                int64_t  m,
                int64_t  k,
                int64_t  b,
                pview *  pv,
                double * last_Y,
                int64_t  lda_lY,
                double * last_W,
                double * my_last_W);


/**
 * \brief Perform 2D QR with two levels of blocking
 *
 * \param[in,out] A m-by-k dense matrix
 * \param[in] lda_A lda of A
 * \param[in] m number of rows in A
 * \param[in] k number of columns in A
 * \param[in] b large block-size to which to aggregate Y, is a multiple of b_sub
 * \param[in] b_sub small block size with which matrix is distributed
 * \param[in] pv current processor grid view
 * \param[in] aggreg_Y is either null or a buffer into which we should aggregate
 *                      broadcasted panels
 * \param[in] lda_aY lda of aggreg_Y if not null
 * \param[in] desc_A descriptor for whole A matrix
 * \param[in] org_A pointer to top left corner of A matrix
 * \param[in] IA row index offset
 * \param[in] JA column index offset
 **/
void QR_2D_2D(double  * A,
              int64_t   lda_A,
              int64_t   m,
              int64_t   k,
              int64_t   b,
              int64_t   b_sub,
              pview   * pv,
              double  * aggreg_Y,
              int64_t   lda_aY,
              int *     desc_A = NULL,
              double *  org_A = NULL,
              int64_t   IA=1,
              int64_t   JA=1);

/**
 * \brief Perform 2D QR with ScaLAPACK panel factorization
 *
 * \param[in,out] A m-by-k dense matrix
 * \param[in] lda_A lda of A
 * \param[in] m number of rows in A
 * \param[in] k number of columns in A
 * \param[in] b number of Householder vectors
 * \param[in] W is either null or Y1^TT
 * \param[in] pv current processor grid view
 * \param[in] aggreg_Y is either null or a buffer into which we should aggregate
 *                      broadcasted panels
 * \param[in] lda_aY lda of aggreg_Y if not null
 * \param[in] desc_A descriptor for whole A matrix
 * \param[in] org_A pointer to top left corner of A matrix
 * \param[in] IA row index offset
 * \param[in] JA column index offset
 **/
void QR_scala_2D( double * A,
                  int64_t  lda_A,
                  int64_t  m,
                  int64_t  k,
                  int64_t  b,
                  pview *  pv,
                  double * aggreg_Y,
                  int64_t  lda_aY,
                  int const * desc_A,
                  double * org_A,
                  int64_t  IA,
                  int64_t  JA);

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
void QR_tree_2D(double *        A,
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
                int64_t const   _stop_at=-1);
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
void QR_butterfly_2D(double *        A,
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
                int64_t const   _stop_at=-1);

/**
 * \brief Perform 2D QR with one level of blocking and with pipelining
 *
 * \param[in,out] A m-by-k dense matrix
 * \param[in] lda_A lda of A
 * \param[in] m number of rows in A
 * \param[in] k number of columns in A
 * \param[in] b number of Householder vectors
 * \param[in] W is either null or Y1^TT
 * \param[in] pv current processor grid view
 * \param[in] last_Y is the Y broadcast at the last step or is null
 * \param[in] lda_lY lda of last_Y if not null
 **/
void QR_2D_pipe(double * A,
                int64_t  lda_A,
                int64_t  m,
                int64_t  k,
                int64_t  b,
                pview *  pv,
                double * last_Y=NULL,
                int64_t  lda_lY=0,
                double * last_W=NULL,
                double * my_last_W=NULL);

#endif
