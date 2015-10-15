#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "CANDMC.h"
#include "unit_test.h"

/* confirms LU factorization of a matrix
 * with arbitrary pivoting
 * assumes the matrix was creating with 
 * A[row,col] = rand48() with seed48(seed+col*dim + row) */
void pvt_con_lu(int const       nrows,
                int const       ncols,
                int const       seed,
                double const*   LU,
                int const*      P){


  int row, col, i, j, k;

  double *A, *A_piv, div;

  assert(ncols > 0);
  assert(nrows >= ncols);
  assert(seed >= 0);

  assert(0==(posix_memalign((void**)&A,
                            ALIGN_BYTES,
                            nrows*ncols*sizeof(double))));
  assert(0==(posix_memalign((void**)&A_piv,
                            ALIGN_BYTES,
                            nrows*ncols*sizeof(double))));
  for (col=0; col<ncols; col++){
    for (row=0; row<nrows; row++){
      srand48(seed + row+col*nrows);
      A[row+col*nrows] = drand48();
    }
  }
  pivot_mat(nrows,ncols,P,A,A_piv);
  for (i=0; i<MIN(nrows,ncols); i++){
    div = A_piv[i*nrows+i];
    if (div != 0.0){
      for (j=i+1; j<nrows; j++){
        A_piv[i*nrows+j] = A_piv[i*nrows+j]/div;
      }
    } else printf("DIAGONAL WAS EXACTLY 0\n");
    for (j=i+1; j<nrows; j++){
      for (k=i+1; k<ncols; k++){
        A_piv[k*nrows+j] = A_piv[k*nrows+j] 
                                - A_piv[i*nrows+j]*A_piv[k*nrows+i];
      }
    }
  }
#ifdef VERBOSE
  print_matrix(A_piv,nrows,ncols);
#endif

  bool correct = true;
  for (i=0; i<ncols; i++){
    for (j=0; j<nrows; j++){
      if (fabs(LU[i*nrows+j] - A_piv[i*nrows+j]) > 1E-6 &&
          fabs((LU[i*nrows+j] - A_piv[i*nrows+j])/A_piv[i*nrows+j]) > 1E-6){
        DEBUG_PRINTF("LU[%d][%d] = %lf, should have been %lf\n",
                     j,i,LU[i*nrows+j],A_piv[i*nrows+j]);
        correct = false;
      }
    }
  }
  if (correct) printf("given the pivot matrix, the answer is CORRECT\n");
  else printf("given the pivot matrix, the answer is INCORRECT\n");
      
  double * max_norm_tnmt = (double*)malloc(sizeof(double)); 
  double * frb_norm_tnmt = (double*)malloc(sizeof(double)); 
  backerr_lu(nrows,ncols,seed,A_piv,
             P,frb_norm_tnmt,max_norm_tnmt,
             0,0,nrows,ncols);
  printf("with this pivot matrix blas 2 LU gets backward norms |(A-LU)|, frobenius = %E, max = %E\n",
                frb_norm_tnmt[0],max_norm_tnmt[0]);
}
/* confirms LU factorization of a matrix
 * by computing the backward error norm
 * assumes the matrix was creating with 
 * A[row,col] = rand48() with seed48(seed+col*dim + row) */
void backerr_lu(int const       nrows,
                int const       ncols,
                int const       seed,
                double const*   LU,
                int const*      P,
                double*         frb_norm,
                double*         max_norm,
                int const       row_st,
                int const       col_st,
                int const       num_row_chk,
                int const       num_col_chk){


  int row, col, i;

  double *A, *A_piv, val, err;

  assert(ncols > 0);
  assert(nrows >= ncols);
  assert(seed >= 0);

  assert(0==(posix_memalign((void**)&A,
                            ALIGN_BYTES,
                            nrows*ncols*sizeof(double))));
  assert(0==(posix_memalign((void**)&A_piv,
                            ALIGN_BYTES,
                            nrows*ncols*sizeof(double))));
  for (col=0; col<ncols; col++){
    for (row=0; row<nrows; row++){
      srand48(seed + row+col*nrows);
      A[row+col*nrows] = drand48();
    }
  }
  pivot_mat(nrows,ncols,P,A,A_piv);
  //print_matrix(A_piv,nrows,ncols);
  *frb_norm = 0.0;
  *max_norm = 0.0;
  for (col=col_st; col<col_st+num_col_chk; col++){
    for (row=row_st; row<row_st+num_row_chk; row++){    
      val = 0.0;        
      for (i=0; i<MIN(row+1,col+1); i++){
        if (i == row){
          val += 1.0*LU[i+col*nrows];
        } else {
          val += LU[row+i*nrows]*LU[i+col*nrows];
        }
      }
      err       = A_piv[row+col*nrows]-val;
      *frb_norm += err*err;
      *max_norm  = MAX((*max_norm),fabs(err));
    }
  }
  *frb_norm = sqrtf(*frb_norm);
//  free(A);
//  free(A_piv);
}
void backerr_lu(int const       nrows,
                int const       ncols,
                int const       seed,
                double const*   LU,
                int const*      P,
                double*         frb_norm,
                double*         max_norm){
  backerr_lu(nrows,ncols,seed,LU,P,frb_norm,max_norm,0,0,nrows,ncols);
}

#ifdef UNIT_TEST
int main(int argc, char **argv) {
  int myRank, numPes, b_sm, b_lrg, n, c_rep, test_mask;

  CommData_t *cdt_glb = (CommData_t*)malloc(sizeof(CommData_t));
  INIT_COMM(numPes, myRank, 1, cdt_glb);

  if (argc > 1) b_sm = atoi(argv[1]);
  else b_sm = 16;

  if (argc > 2) b_lrg = atoi(argv[2]);
  else b_lrg = b_sm*sqrt(numPes)*2;

  if (argc > 3) n = atoi(argv[3]);
  else n = b_lrg*2;

  if (argc > 4) c_rep = atoi(argv[4]);
  else c_rep = 1;
  
  if (argc > 5) test_mask = atoi(argv[5]);
  else test_mask = 0x1F;

  if (myRank == 0) {
    printf("starting unit tests for 2.5D LU.");
    printf("b_sm=%d, b_lrg=%d, n=%d\n",b_sm,b_lrg,n);
  }

  if (test_mask&0x1){
    if (myRank == 0)  {
      seq_tnmt_unit_test(b_sm);
    }
    GLOBAL_BARRIER(cdt_glb);
  }
  if (test_mask&0x2){
    par_tnmt_unit_test(b_sm, myRank, numPes, 0, cdt_glb);
    GLOBAL_BARRIER(cdt_glb);
  }
  if (test_mask&0x4){
    par_pivot_unit_test(b_sm, n, myRank, numPes, 0, cdt_glb);
    GLOBAL_BARRIER(cdt_glb);
  }
  if (test_mask&0x8){
    lu_25d_unit_test(n, b_sm, b_lrg, myRank, numPes, c_rep, cdt_glb);
    GLOBAL_BARRIER(cdt_glb);
  }
  if (test_mask&0x10){
    lu_25d_pvt_unit_test(n, b_sm, b_lrg, myRank, numPes, c_rep, cdt_glb);
    GLOBAL_BARRIER(cdt_glb);
  }
  COMM_EXIT;
  return 0;
}
#endif
