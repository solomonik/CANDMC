#ifndef __HH_RECON_H__
#define __HH_RECON_H__

#include "../../shared/comm.h"

/**
 * \brief perform sequential b-by-b 
 *        TRSM to compute invT from W matrix (output of hh_recon QR)
 * \param[in] W b-by-b triangular factor -T*Y1'
 * \param[in] b dimension of W and T
 * \param[in,out] invT preallcative space for T^-1
 */
void compute_invT_from_W(double const * W,
                         int64_t        b, 
                         double *       invT);

/**
 * \brief Perform TSQR over a (sub)-column of processors 
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
                int64_t *     signs);

/**
 * \brief Perform TSQR over a (sub)-column of processors 
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
                    int64_t *     signs);

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
                 CommData_t     cdt);
#endif
