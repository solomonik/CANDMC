#ifndef __BUTTERFLY_TSQR_H__
#define __BUTTERFLY_TSQR_H__

#include "../../shared/comm.h"

#ifndef TAU_BLK
#define TAU_BLK 16
#endif

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
 * \param[in,out] tree_data TAU and Y data for the TSQR tree, 
 *                must be of size ((log2(p)+1)*b+b)/2-by-b
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
                    double *      tree_data);

/**
 * \brief Construct Q from butterfly TSQR over a (sub)-column of processors 
 *
 * \param[in] Y m-by-b matrix of Householder vectors from TSQR
 * \param[in] lda_Y lda of Y
 * \param[in] tau TAU values associated with the first level of TSQR HH vecs
 * \param[in,out] Q1 first k columns of Q (buffer should be prealloced and 
 *                                         preset if is_form_q set to 0)
 * \param[in] lda_Q length of leading dimension of Q1
 * \param[in] m number of rows in Y
 * \param[in] b number of columns in Y, number of columns of Q1 to compute
 * \param[in] myRank rank in communicator column
 * \param[in] numPes number of processes in column
 * \param[in] root the root of the tree (who will own R at the end)
 * \param[in] cdt MPI communicator for column
 * \param[in] tree_data TAU and Y data for the TSQR tree, 
 *                must be of size ((log2(p)+1)*b+b)/2-by-b
 **/
void butterfly_construct_Q1(double const *  Y,
                            int64_t         lda_Y,
                            double const *  tau,
                            double *        Q1,
                            int64_t         lda_Q,
                            int64_t         m,
                            int64_t         b,
                            int64_t         myRank,
                            int64_t         numPes,
                            int64_t         root,
                            CommData_t      cdt,
                            double *        tree_data);

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
void apply_butterfly_tsqr_QT( double const *  Y,
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
                              double *        tree_data);

#endif
