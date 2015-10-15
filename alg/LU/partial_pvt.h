#ifndef __PARTIAL_PVT_H__
#define __PARTIAL_PVT_H__

#include "../shared/comm.h"

/**
 * \brief performs parallel partial pivoting on a tall-skinny matrix
 *
 * \param[in,out] A pointer to nb-by-b block in lda-by-b buffer 
 * \param[in] lda leading dimension of A
 * \param[in,out] P pivot matrix which contains the initial index of each row of A
 * \param[in] nb number of local rows of A
 * \param[in] b number of columns in A
 * \param[in] myRank rank in column
 * \param[in] numPes number of processors in column
 * \param[in] root root process in column
 * \param[in] cdt communicator
 */
void partial_pvt(double *       A, 
                 int const      lda,
                 int *          P,
                 int64_t const  nb,
                 int64_t const  b,
                 int const      myRank,
                 int const      numPes,
                 int const      root,
                 CommData_t     cdt);



#endif
