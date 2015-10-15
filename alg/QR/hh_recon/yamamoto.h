#ifndef __YAMAMOTO_H__
#define __YAMAMOTO_H__
/**
 * \brief Perform TSQR and construct Yamamoto's basis kernel representation [Q1-S; Q2][Q1-S]^-1[Q1-S; Q2] 
 *          on a (sub)-column of processors 
 *
 * \param[in,out] A m-by-b dense tall-skinny matrix [tree_Y\R] on output
 * \param[in] lda_A lda of A
 * \param[in,out] Qm m-by-b dense tall-skinny matrix [Q1-I; Q2] on output
 * \param[in] lda_Qm lda of Qm
 * \param[in] m number of rows in A
 * \param[in] b number of columns in A
 * \param[in,out] W b-by-b matrix (must be preallocated containing [L \ U] = LU(Q1-S)
 * \param[in] myRank rank in communicator column
 * \param[in] numPes number of processes in column
 * \param[in] root the root of the tree (who will own R at the end)
 * \param[in] req_id request id to use for send/recv
 * \param[in] cdt MPI communicator for column
 * \param[in,out] signs S
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
              CommData_t cdt);

#endif
