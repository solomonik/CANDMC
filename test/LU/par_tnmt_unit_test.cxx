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

/* test parallel tournament pivoting
 * b is the size of the panel */
void par_tnmt_unit_test(int b, int myRank, int numPes, int req_id, CommData *cdt){
  if (myRank == 0)  printf("unit testing parallel tournament pivoting...\n");
  double *A,*A_buf,*R,*whole_A;
  int *P, *P_br, *P_I;
  int i,j,row,col,info;
  //double frb_norm_tnmt[1], max_norm_tnmt[1];
  double * max_norm_tnmt = (double*)malloc(sizeof(double)); 
  double * frb_norm_tnmt = (double*)malloc(sizeof(double)); 
//  double frb_norm_gepp, max_norm_gepp;
  double * tot_max_norm_tnmt = (double*)malloc(sizeof(double)); 
  double * tot_frb_norm_tnmt = (double*)malloc(sizeof(double)); 
//  double tot_frb_norm_gepp, tot_max_norm_gepp;

  int seed_offset = 1000;
    
  assert(0==(posix_memalign((void**)&A,
          ALIGN_BYTES,
          2*b*b*sizeof(double))));
  assert(0==(posix_memalign((void**)&A_buf,
          ALIGN_BYTES,
          2*b*b*sizeof(double))));
  assert(0==(posix_memalign((void**)&R,
          ALIGN_BYTES,
          2*b*b*sizeof(double))));
  assert(0==(posix_memalign((void**)&whole_A,
          ALIGN_BYTES,
          numPes*b*b*sizeof(double))));
  assert(0==(posix_memalign((void**)&P_br,
          ALIGN_BYTES,
          3*b*sizeof(int))));
  assert(0==(posix_memalign((void**)&P,
          ALIGN_BYTES,
          2*b*sizeof(int))));
  assert(0==(posix_memalign((void**)&P_I,
          ALIGN_BYTES,
          numPes*b*sizeof(int))));


  for (col=0; col<b; col++){
    for (row=0; row<b; row++){
      srand48(seed_offset + (myRank*b + row)+(col*numPes*b));
      A[row+col*b] = drand48();
    }
  }
  for (i=0; i<b; i++) P[i] = i+myRank*b;
  //get best rows
  tnmt_pvt_1d(A,R,P,A_buf,P_br,b,myRank,0,numPes,0,req_id,cdt);

  if (myRank == 0){
    //print_matrix(R,b,b);
    cdgetrf(b,b,R,b,P,&info);
    for (i=0; i<b; i++){
      memcpy(whole_A+i*b*numPes,R+i*b,b*sizeof(double));
    }
    pivot_conv(b,P,P_br);
    inv_br(b,0,P_br,P);
    DBCAST(P,b,COMM_INT_T,0,cdt,myRank);
    DBCAST(A,b*b,COMM_DOUBLE_T,0,cdt,myRank);
    DBCAST(R,b*b,COMM_DOUBLE_T,0,cdt,myRank);
  } else {
    DBCAST(P,b,COMM_INT_T,0,cdt,myRank);
    DBCAST(A_buf,b*b,COMM_DOUBLE_T,0,cdt,myRank);
    for (i=0; i<b; i++){
      if (P[i] >= myRank*b && P[i] < (myRank+1)*b){
  for (j=0; j<b; j++){
    A[j*b+P[i]-myRank*b] = A_buf[j*b+i];
  }
      }
    }
    DBCAST(A_buf,b*b,COMM_DOUBLE_T,0,cdt,myRank);
    cdtrsm('R','U','N','N',b,b,1.0,A_buf,b,A,b);
    for (i=0; i<b; i++){
      memcpy(whole_A+i*b*numPes   ,A_buf+i*b,b*sizeof(double));
      memcpy(whole_A+i*b*numPes+b*myRank,A+i*b    ,b*sizeof(double));
    }
  }
  br_to_pivot(b,b*numPes,P_br,P_I);
  DBCAST(P_I,b*numPes,COMM_INT_T,0,cdt,myRank);
  backerr_lu(numPes*b,b,seed_offset,whole_A,P_I,frb_norm_tnmt,max_norm_tnmt,
       myRank*b,0,b,b);
  DEBUG_PRINTF("[%d] my norms are %E and %E\n", myRank, frb_norm_tnmt[0], max_norm_tnmt[0]);
  frb_norm_tnmt[0] = frb_norm_tnmt[0]*frb_norm_tnmt[0];
  ALLREDUCE(frb_norm_tnmt,tot_frb_norm_tnmt,1,COMM_DOUBLE_T,COMM_OP_SUM,cdt);
  tot_frb_norm_tnmt[0] = sqrtf(tot_frb_norm_tnmt[0]);
  ALLREDUCE(max_norm_tnmt,tot_max_norm_tnmt,1,COMM_DOUBLE_T,COMM_OP_MAX,cdt);
  frb_norm_tnmt[0] = sqrtf(frb_norm_tnmt[0]);
  COMM_BARRIER(cdt);
  DEBUG_PRINTF("[%d] reductions complete\n", myRank);




  if (myRank == 0) {
    printf("with tournmanet pivoting, backward norms |(A-LU)|, frobenius = %E, max = %E\n",
          tot_frb_norm_tnmt[0],tot_max_norm_tnmt[0]);
    if (tot_frb_norm_tnmt[0] < 1.E-12 && tot_max_norm_tnmt[0] <1.E-12){
      printf("test passed (parallel tournament pivoting test)\n");
    } else { 
      printf("TEST FAILED (parallel tournament pivoting test)\n");
    }
  }

  free(P_br);
  free(A);
  free(A_buf);
  free(R);
  free(whole_A);
  free(P_I);
}
