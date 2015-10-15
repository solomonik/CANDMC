#ifndef __NuSE_H__
#define __NuSE_H__


/**
 * \brief Perform reduction to banded using 2D QR
 *
 * \param[in,out] A n-by-n dense symmetric matrix, stored unpacked
                    pointer should refer to current working corner of A
 * \param[in] lda_A lda of A
 * \param[in] n number of rows and columns in A
 * \param[in] b is the large block to which we are reducing the band, 
                b must be a multiple of b_sub
 * \param[in] b_sub small block size with which matrix is distributed
 * \param[in] pv current processor grid view oriented at corner of A 
 **/
void sym_full2band(double  * A,
                   int64_t   lda_A,
                   int64_t   n,
                   int64_t   b,
                   int64_t   b_sub,
                   pview   * pv);

/**
 * \brief Perform reduction to banded using a 3D algorithm
 *
 * \param[in,out] A n-by-n dense symmetric matrix, stored unpacked
 *                  pointer should refer to current working corner of A
 *                 blocked accrows crow and ccol pv and replicated across clyr
 * \param[in] lda_A lda of A
 * \param[in] n number of rows and columns in A
 * \param[in] b_agg is the number of U vectors to aggregate before applying to 
 *                  full trailing matrix (must be multiply of b_qr)
 * \param[in] bw is the bandwidth to reduce to (2D QR size)
 * \param[in] b_sub small block size with which matrix is distributed
 * \param[in] pv current processor grid view oriented at corner of A 
 **/
void sym_full2band_3d(double *    A,
                      int64_t     lda_A,
                      int64_t     n,
                      int64_t     b_agg,
                      int64_t     bw,
                      int64_t     b_sub,
                      pview_3d *  pv);

/**
 * \brief Perform reduction to banded using 2D QR
 *
 * \param[in,out] A n-by-n dense symmetric matrix, stored unpacked
                    pointer should refer to current working corner of A
 * \param[in] lda_A lda of A
 * \param[in] n number of rows and columns in A
 * \param[in] b is the large block to which we are reducing the band, 
                b must be a multiple of b_sub
 * \param[in] b_sub small block size with which matrix is distributed
 * \param[in] pv current processor grid view oriented at corner of A 
 * \param[in] desc_A descriptor for whole A matrix
 * \param[in] org_A pointer to top left corner of A matrix
 * \param[in] IA row index offset
 * \param[in] JA column index offset
 **/
void sym_full2band_scala(double  *    A,
                         int64_t      lda_A,
                         int64_t      n,
                         int64_t      b,
                         int64_t      b_sub,
                         pview *      pv,
                         int const *  desc_A,
                         double *     org_A,
                         int64_t      IA=1,
                         int64_t      JA=1);
#endif
