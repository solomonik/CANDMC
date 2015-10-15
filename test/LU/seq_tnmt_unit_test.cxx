#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tnmt_pvt.h"
#include "unit_test.h"
#include "../shared/util.h"
#include "../shared/seq_lu.h"

/* test sequetnial tournament pivoting
 * b is the size of the panel */
void seq_tnmt_unit_test(int b){
  printf("unit testing local tournament pivoting...\n");
  double *A,*B;
  int *P, *P_br, *P_I;
  int i,row,col,info;
  double frb_norm_tnmt, max_norm_tnmt;
  double frb_norm_gepp, max_norm_gepp;

  int seed_offset = 1000;
    
  assert(0==(posix_memalign((void**)&A,
          ALIGN_BYTES,
          2*b*b*sizeof(double))));
  assert(0==(posix_memalign((void**)&B,
          ALIGN_BYTES,
          2*b*b*sizeof(double))));
  assert(0==(posix_memalign((void**)&P_br,
          ALIGN_BYTES,
          b*sizeof(int))));
  assert(0==(posix_memalign((void**)&P,
          ALIGN_BYTES,
          2*b*sizeof(int))));
  assert(0==(posix_memalign((void**)&P_I,
          ALIGN_BYTES,
          2*b*sizeof(int))));

  for (col=0; col<b; col++){
    for (row=0; row<2*b; row++){
      srand48(seed_offset + row+col*2*b);
      A[row+col*2*b] = drand48();
    }
  }
  //get best rows
  local_tournament_col_maj(A,B,P_br,2*b,b,2*b);

  memcpy(B,A,2*b*b*sizeof(double));
  cdlaswp(b,B,2*b,1,b,P_br,1);

  //perform LU on the top part
  cdgetrf(b,b,B,2*b,P,&info);
  
  //trsm on the bottom part
  cdtrsm('R','U','N','N',b,b,1.0,B,2*b,B+b,2*b);
  

  //get tournament pivoting permuted original
  for (i=0; i<2*b; i++) P_I[i] = i;
  pivot_conv(b,P_br,P_I);
  pivot_conv(b,P,P_I);
  //pivot_mat(2*b,b,P_I,A,A_piv);
  backerr_lu(2*b,b,seed_offset,B,P_I,&frb_norm_tnmt,&max_norm_tnmt);

  //get partial pivoting answer
  cdgetrf(2*b,b,A,2*b,P,&info);
  for (i=0; i<2*b; i++) P_I[i] = i;
  pivot_conv(b,P,P_I);
  backerr_lu(2*b,b,seed_offset,A,P_I,&frb_norm_gepp,&max_norm_gepp);

  //get partial pivoting permuted original
  //cdlaswp(b,A,2*b,1,b,P,1);

  printf("with tournmanet pivoting, backward norms |(A-LU)|, frobenius = %E, max = %E\n",
        frb_norm_tnmt,max_norm_tnmt);
  printf("with GEPP pivoting, backward norms |(A-LU)|, frobenius = %E, max = %E\n",
        frb_norm_gepp,max_norm_gepp);
  if (frb_norm_gepp >= frb_norm_tnmt/100. && max_norm_gepp >= max_norm_tnmt/100.){
    printf("test passed (local tournament pivoting test)\n");
  } else { 
    printf("TEST FAILED (local tournament pivoting test)\n");
    printf("diff between frb = %E\n",frb_norm_tnmt-frb_norm_gepp);
    printf("diff between max = %E\n",max_norm_tnmt-max_norm_gepp);
  }

  free(A);
  free(B);
  free(P_br);
  free(P);
  free(P_I);
}
