#ifndef __QR_YAMAMOTO_H__
#define __QR_YAMAMOTO_H__

class aggregator {
  public:
    int64_t lda_aQm;
    int64_t lda_aT;
    int64_t shift;
    double * aQm;
    double * aT;
    int64_t n;

    /**
     * \brief constructor, arrays are allocated and initialized to zero, n is set to zero
     */
    aggregator(int64_t lda_aQm, int64_t lda_aT);

    /**
     * \brief sets n to zero (does not clear buffers)
     */    
    void reset();
    
    /**
     * \brief shifts row pointer for next Qm
     */    
    void shift_down(int64_t b);


    /**
     * \brief increases n by b, appends Qm to aQm and computes larger aT from Qm, T, aQm, aT
     * \param[in] mb height of Qm
     * \param[in] b width of Qm
     * \param[in] Qm new Q-I matrix to append
     * \param[in] lda_Qm new Q-I matrix to append
     * \param[in] T new T matrix to append
     * \param[in] pv current processor grid view
     */    
    void append(int64_t        mb,
                int64_t        b,
                double const * Qm,
                int64_t        lda_Qm,
                double const * T,
                pview *        pv);
};
/**
 * \brief Perform (I-Qm*T*Qm^T)A 
 *
 * \param[in,out] Qm m-by-k is [Q1-S; Q2]
 * \param[in] lda_Qm lda of Qm
 * \param[in,out] A m-by-k dense trailing matrix
 * \param[in] lda_A lda of A
 * \param[in] m number of rows in A
 * \param[in] k number of columns in A
 * \param[in] b number of Householder vectors
 * \param[in] T is LU of Q1-S owned on root column
 * \param[in] pv current processor grid view
 * \param[in] agg is either null or a class into which we should append Qm and T
 **/
void update_Yamamoto_A(double *     Qm,
                       int64_t      lda_Qm,
                       double *     A,
                       int64_t      lda_A,
                       int64_t      m,
                       int64_t      k,
                       int64_t      b,
                       double *     T,
                       pview *      pv,
                       aggregator * agg);


/**
 * \brief Perform (I-Qm*T*Qm^T)A  after bcasting Y
 *
 * \param[in,out] Qm m-by-k is [Q1-S; Q2] owned on
 *                all columns of the 2D grid
 * \param[in] lda_Qm lda of Qm
 * \param[in,out] A m-by-k dense trailing matrix
 * \param[in] lda_A lda of A
 * \param[in] mb number of rows I own in A
 * \param[in] kb number of columns I own in A
 * \param[in] b number of Householder vectors
 * \param[in] T is LU of Q1-S owned by all processors
 * \param[in] pv current processor grid view
 **/
void upd_Yamamoto_A(double const * Qm,
                    int64_t        lda_Qm,
                    double *       A,
                    int64_t        lda_A,
                    int64_t        mb,
                    int64_t        kb,
                    int64_t        b,
                    double const * T,
                    pview *        pv);


/**
 * \brief Perform 2D QR with one level of blocking
 *
 * \param[in,out] A m-by-k dense matrix
 * \param[in] lda_A lda of A
 * \param[in] m number of rows in A
 * \param[in] k number of columns in A
 * \param[in] b number of Householder vectors
 * \param[in] pv current processor grid view
 * \param[in] agg is either null or a class into which we should append Qm and T
 * \param[in] compute_Y if true, Y is computed and placed in A
 **/
void QR_Yamamoto_2D(double *     A,
                    int64_t      lda_A,
                    int64_t      m,
                    int64_t      k,
                    int64_t      b,
                    pview *      pv,
                    aggregator * agg=NULL,
                    bool         compute_Y=false);


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
 * \param[in] agg is either null or an aggregation class for Y and T
 * \param[in] compute_Y if true, Y is computed and placed in A
 **/
void QR_Yamamoto_2D_2D(double  *    A,
                       int64_t      lda_A,
                       int64_t      m,
                       int64_t      k,
                       int64_t      b,
                       int64_t      b_sub,
                       pview   *    pv,
                       aggregator * agg=NULL,
                       bool         compute_Y=false);
#endif
