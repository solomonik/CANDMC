/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */
#include <mpi.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../shared/util.h"
#include "partial_pvt.h"
#include "tnmt_pvt.h"
#include <algorithm>

extern "C"{
int idamax_(int const * N, double const * A, int const * inc_A);
}
//void dscal_(int const * n, double * dA, double * dX,  int const * incX);
//void dger_( int const *          M,
//           int const *          N,
//           double const *       alpha,
//           double const *       X,
//           int const *          incX,
//           double const *       Y,      
//           int const *          incY,
//           double *             A,
//           int const *          lda);
//}
//
int cidamax(int const N, double const * A, int const inc_A){
  return idamax_(&N, A, &inc_A);
}
//
//void cdscal(int const n, double dA, double * dX,  int const incX){
//  dscal_(&n, &dA, dX, &incX);
//}
//
//void cdger(int const      M,
//           int const      N,
//           double const   alpha,
//           double const * X,
//           int const      incX,
//           double const * Y,  
//           int const      incY,
//           double *       A,
//           int const      lda){
//  dger_(&M, &N, &alpha, X, &incX, Y, &incY, A, &lda);
//}



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
                 CommData_t     cdt){
  int i, j, imax, all_imax, npiv;
  double dmax, all_dmax;
  double * buffer     = (double*)malloc(sizeof(double)*b);
  double * col        = (double*)malloc(sizeof(double)*nb);
  int * pivoted_rows  = (int*)malloc(sizeof(int)*b);
  int * P_start       = (int*)malloc(sizeof(int)*nb);
  MPI_Status stat;

  TAU_FSTART(partial_pvt_inner);

  memcpy(P_start, P, nb*sizeof(int));

  npiv = 0;
  for (i=0; i<b; i++){
    TAU_FSTART(select_pivot);
    memcpy(col, A+lda*i, sizeof(double)*nb);
    for (j=0; j<npiv; j++){
      col[pivoted_rows[j]] = 0.0;
    }
    imax = cidamax(nb, col, 1)-1;
    if (imax >= 0) 
      dmax = fabs(col[imax]);
    else
      dmax = 0.0;

    MPI_Allreduce(&dmax, &all_dmax, 1, MPI_DOUBLE, MPI_MAX, cdt.cm);
    assert(all_dmax!=0.0);

    if (dmax == all_dmax) all_imax = myRank;
    else all_imax = -1;

    MPI_Allreduce(MPI_IN_PLACE, &all_imax, 1, MPI_INT, MPI_MAX, cdt.cm);

    if (myRank == all_imax){
      pivoted_rows[npiv] = imax;
      npiv++;
      lda_cpy(1, b, lda, 1, A+imax, buffer);
      
      if (myRank != root)
        MPI_Sendrecv_replace(P+imax, 1, MPI_INT, root, 2*i, 
                             root, 2*i+1, cdt.cm, &stat);
      else {
        P[i] = P_start[imax];
      }
      col[imax] = 0.0;
        
    }
    if (myRank == root && all_imax != root){
      MPI_Sendrecv_replace(P+i, 1, MPI_INT, all_imax, 2*i+1, 
                           all_imax, 2*i, cdt.cm, &stat);
    }
    TAU_FSTOP(select_pivot);
    TAU_FSTART(update_thin_panel);
    MPI_Bcast(buffer, b, MPI_DOUBLE, all_imax, cdt.cm);
    
    cdscal(nb, 1.0/buffer[i], col, 1);

    cdger(nb, b-i-1, -1.0, 
          col, 1,
          buffer+i+1, 1,
          A+(i+1)*lda, lda);
    for (j=0; j<npiv; j++){
      col[pivoted_rows[j]] =A[i*lda+pivoted_rows[j]];
    }
    memcpy(A+lda*i, col, sizeof(double)*nb);
    TAU_FSTOP(update_thin_panel);
  }
  TAU_FSTOP(partial_pvt_inner);

  free(buffer);
  free(col);
  free(pivoted_rows);
  free(P_start);
  
}


