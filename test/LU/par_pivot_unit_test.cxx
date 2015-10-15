/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tnmt_pvt.h"
//#include "../shared/comm.h"
#include "unit_test.h"
#include "../shared/util.h"
#include "../shared/seq_lu.h"

/* test parallel tournament pivoting parallel swap function */
void par_pivot_unit_test(int  b_sm, 
       int  mat_dim,
       int  myRank, 
       int  numPes, 
       int  req_id, 
       CommData cdt){
  if (myRank == 0)  
    printf("unit testing block cyclic parallel tournament pivoting...\n");
  double *A,*A_buf,*R,*R_out;
  int *P, *P_br, *P_I;
  int i,j,row,col,info;
  double * max_norm_tnmt = (double*)malloc(sizeof(double)); 
  double * frb_norm_tnmt = (double*)malloc(sizeof(double)); 
  double * tot_max_norm_tnmt = (double*)malloc(sizeof(double)); 
  double * tot_frb_norm_tnmt = (double*)malloc(sizeof(double)); 
  int idx_off;
  bool passed = true;
  double val;

  const int mat_subdim  = mat_dim/numPes;
  
  int seed_offset = 1000;
    
  assert(0==(posix_memalign((void**)&A,
          ALIGN_BYTES,
          2*mat_subdim*b_sm*sizeof(double))));
  assert(0==(posix_memalign((void**)&A_buf,
          ALIGN_BYTES,
          2*mat_subdim*b_sm*sizeof(double))));
  assert(0==(posix_memalign((void**)&R,
          ALIGN_BYTES,
          2*mat_subdim*b_sm*sizeof(double))));
  assert(0==(posix_memalign((void**)&R_out,
          ALIGN_BYTES,
          2*mat_subdim*b_sm*sizeof(double))));
  assert(0==(posix_memalign((void**)&P_br,
          ALIGN_BYTES,
          4*mat_subdim*sizeof(int))));
  assert(0==(posix_memalign((void**)&P,
          ALIGN_BYTES,
          4*mat_subdim*sizeof(int))));
  assert(0==(posix_memalign((void**)&P_I,
          ALIGN_BYTES,
          4*mat_dim*sizeof(int))));


  for (idx_off=0; idx_off < mat_subdim; idx_off+=b_sm){
    RANK_PRINTF(myRank,0,"idx_off=%d\n",idx_off);
    for (i=0; i<mat_subdim/b_sm; i++){
      for (j=0; j<b_sm; j++){
  row = i*b_sm*numPes + myRank*b_sm + j;
  for (col=0; col<b_sm; col++){
    srand48(seed_offset + row + col*mat_dim);
    A[i*b_sm+j+col*mat_subdim] = drand48();
  }
  P[i*b_sm+j] = row;
  P_br[i*b_sm+j] = row;
  P_I[i*b_sm+j] = row;
      }
    }

    //get best rows
    local_tournament_col_maj(A+idx_off,R,P_br,mat_subdim-idx_off,b_sm,mat_subdim);
    pivot_conv(b_sm, P_br, P+idx_off);
    tnmt_pvt_1d(R,R_out,P+idx_off,A_buf,P_br+idx_off,b_sm,myRank,0,numPes,0,req_id,cdt);

    par_pivot(A+idx_off, 
        A_buf, 
        b_sm,
        b_sm,
        b_sm,
        mat_subdim,
        idx_off,
        idx_off*numPes,
        P_I+idx_off,
        P_br+idx_off,
        myRank,
        0,
        numPes,
        1,
        0,
        0,
        mat_subdim/b_sm,
        cdt,
        cdt);

    memcpy(P_br+idx_off,P_I+idx_off,b_sm*sizeof(int));
   
    par_pivot(A+idx_off, 
        A_buf, 
        b_sm,
        b_sm,
        b_sm,
        mat_subdim,
        idx_off,
        idx_off*numPes,
        P_I+idx_off,
        P_br+idx_off,
        myRank,
        0,
        numPes,
        1,
        0,
        0,
        mat_subdim/b_sm,
        cdt,
        cdt);
        

    for (i=0; i<mat_subdim/b_sm; i++){
      for (j=0; j<b_sm; j++){
  row = i*b_sm*numPes + myRank*b_sm + j;
  for (col=0; col<b_sm; col++){
    srand48(seed_offset + row + col*mat_dim);
    val = drand48();
    if (A[i*b_sm+j+col*mat_subdim] != val){
      passed = false;
      RANK_PRINTF(myRank,myRank,"ERROR: A[%d,%d] = %lf != %lf\n",
      row, col, A[i*b_sm+j+col*mat_subdim], val);
    }
  }
      }
    }
    if (!passed){
      break;
    }
  }
  if (!passed){
    printf("[%d] pivot unit test failed, idx_off = %d\n",myRank,idx_off);
  } else if (myRank == 0){
    printf("[%d] pivot unit test passed\n",myRank);
  }


  free(P_br);
  free(A);
  free(A_buf);
  free(R);
  free(P_I);
}
