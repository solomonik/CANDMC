#ifndef __BITREE_TSQR_H__
#define __BITREE_TSQR_H__

#include "../../shared/comm.h"

#ifndef TAU_BLK
#define TAU_BLK 16
#endif

/**
 * \brief Perform TSQR on a matrix of two stacked upper-trinagular Rs
 *
 * \param[in,out] A 2b-by-b [R1; R2] tall-skinny matrix, Y\R on output
 * \param[in,out] buf 2b-by-b buffer
 * \param[in] b number of columns in A
 * \param[in] tau if not NULL put T matrix here
 **/
void tree_tsqr(double *      A,
               double *      buf,
               int64_t const b,
               double *      tau=NULL);

/**
 * \brief Perform TSQR over a (sub)-column of processors 
 *
 * \param[in,out] A m-by-b dense tall-skinny matrix, Y\R on output
 * \param[in,out] R b-by-b upper-triangular matrix for which A=QR
 * \param[in] m number of rows in A
 * \param[in] b number of columns in A
 * \param[in] lda leading dimension (number of buffer rows) in A
 * \param[in] tau if not NULL put T matrix here
 **/
void local_tsqr(double *      A,
                double *      R,
                int64_t const m,
                int64_t const b,
                int64_t const lda_A,
                double *      tau = NULL);

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
 * \param[in] output_Y whether to overwrite A with Y
 * \param[in,out] tree_data TAU and Y data for the TSQR tree, 
 *                must be of size ((log2(p)+1)*b+b)/2-by-b
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
                  int64_t const   req_id,
                  CommData_t    cdt,
                  int64_t const output_Y=0,
                  double *      tree_data = NULL);

/**
 * \brief Perform TSQR over a (sub)-column of processors 
 *
 * \param[in] Y m-by-b matrix of Householder vectors from TSQR
 * \param[in] lda_Y lda of Y
 * \param[in] tau TAU values associated with the first level of TSQR HH vecs
 * \param[out] Q1 first b columns of Q (buffer should be prealloced)
 * \param[in] lda_Q length of leading dimension of Q1
 * \param[in] m number of rows in Y
 * \param[in] b number of columns in Y
 * \param[in] k number of columns of Q1 to compute
 * \param[in] myRank rank in communicator column
 * \param[in] numPes number of processes in column
 * \param[in] root the root of the tree (who will own R at the end)
 * \param[in] cdt MPI communicator for column
 * \param[in,out] tree_data TAU and Y data for the TSQR tree, 
 *                must be of size ((log2(p)+1)*b+b)/2-by-b
 * \param[in] ID b-by-k matrix which replaces the identity on the tree root if not NULL
 **/
void construct_Q1(double const *  Y,
                  int64_t const   lda_Y,
                  double const *  tau,
                  double *        Q1,
                  int64_t const   lda_Q,
                  int64_t const   m,
                  int64_t const   b,
                  int64_t const   k,
                  int64_t const   myRank,
                  int64_t const   numPes,
                  int64_t const   root,
                  CommData_t      cdt,
                  double *        tree_data = NULL,
                  double *        ID = NULL);

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
 * \param[in] req_id request id to use for send/recv
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
                    double *        tree_data = NULL);

#endif
