/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tnmt_pvt.h"
#include "partial_pvt.h"
//#include "../shared/comm.h"
#include "../shared/util.h"
#include "../shared/seq_lu.h"

/* test parallel tournament pivoting
 * b is the size of the panel */
void par_tnmt_bench(int n, int b, int myRank, int numPes, 
                    int req_id, CommData *cdt, int num_iter){
//  if (myRank == 0)  printf("benchmarking parallel tournament pivoting with b=%d...\n",b);
  double *A,*A_buf,*R;
  int *P, *P_br;
  int i,j,row,col,info,it;
  //double frb_norm_tnmt[1], max_norm_tnmt[1];
//  double * max_norm_tnmt = (double*)malloc(sizeof(double)); 
//  double * frb_norm_tnmt = (double*)malloc(sizeof(double)); 
////  double frb_norm_gepp, max_norm_gepp;
//  double * tot_max_norm_tnmt = (double*)malloc(sizeof(double)); 
//  double * tot_frb_norm_tnmt = (double*)malloc(sizeof(double)); 
//  double tot_frb_norm_gepp, tot_max_norm_gepp;
  double start_time, tnmt_time, out_pivot_time, barrier_time, partial_time;

  int seed_offset = 1000;
  assert(n%numPes==0);
  int nb = n / numPes;
    
  assert(0==(posix_memalign((void**)&A,
                            ALIGN_BYTES,
                            nb*b*sizeof(double))));
  assert(0==(posix_memalign((void**)&A_buf,
                            ALIGN_BYTES,
                            nb*b*sizeof(double))));
  assert(0==(posix_memalign((void**)&R,
                            ALIGN_BYTES,
                            nb*b*sizeof(double))));
  assert(0==(posix_memalign((void**)&P_br,
                            ALIGN_BYTES,
                            3*nb*sizeof(int))));
  assert(0==(posix_memalign((void**)&P,
                            ALIGN_BYTES,
                            nb*sizeof(int))));



  COMM_BARRIER(cdt);
  start_time = TIME_SEC();
  for (i=0; i<num_iter; i++){
    COMM_BARRIER(cdt);
  }
  barrier_time = TIME_SEC();
  barrier_time = (barrier_time-start_time)/num_iter;


  COMM_BARRIER(cdt);
  tnmt_time = 0.0;
  for (i=0; i<num_iter; i++){
    srand48(myRank);
    for (col=0; col<b; col++){
      for (row=0; row<nb; row++){
//        srand48(seed_offset + (myRank*b + row)+(col*numPes*b));
        A[row+col*nb] = drand48();
      }
    }
    for (row=0; row<nb; row++) P[row] = row+myRank*nb;
    COMM_BARRIER(cdt);
  
    start_time = TIME_SEC();
    local_tournament<0>(A, R, P, nb, b, nb);
    tnmt_pvt_1d(R,A,P,A_buf,P_br,b,myRank,0,numPes,0,req_id,cdt);
    COMM_BARRIER(cdt);
    tnmt_time += TIME_SEC()-start_time;
  }
  tnmt_time = tnmt_time/num_iter;
  tnmt_time = tnmt_time-barrier_time;

  COMM_BARRIER(cdt);
  start_time = TIME_SEC();
  for (it=0; it<num_iter; it++){
    if (myRank == 0){
      inv_br(b,0,P_br,P);
      DBCAST(P,b,COMM_INT_T,0,cdt,myRank);
      DBCAST(A,b*b,COMM_DOUBLE_T,0,cdt,myRank);
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
    }
    COMM_BARRIER(cdt);
  }
  out_pivot_time = TIME_SEC();
  out_pivot_time = (out_pivot_time-start_time)/num_iter;
  out_pivot_time = out_pivot_time-barrier_time;
  
  partial_time = 0.0;
  for (i=0; i<num_iter; i++){
    srand48(myRank);
    for (col=0; col<b; col++){
      for (row=0; row<nb; row++){
//        srand48(seed_offset + (myRank*b + row)+(col*numPes*b));
        A[row+col*nb] = drand48();
      }
    }
    for (row=0; row<nb; row++) P[row] = row+myRank*nb;
    COMM_BARRIER(cdt);
  
    start_time = TIME_SEC();
    partial_pvt(A,nb,P,nb,b,myRank,numPes,0,req_id,cdt);
    COMM_BARRIER(cdt);
    partial_time += TIME_SEC() - start_time;;
  }
  partial_time = partial_time/num_iter;
//  partial_time = partial_time-barrier_time;


  if (myRank == 0) {
    printf("%d\t%lf\t%lf\t%lf\t%lf\t%lf\n",b,tnmt_time*1.E3,out_pivot_time*1.E3,partial_time*1.E3,
                            barrier_time*1.E3,(tnmt_time+out_pivot_time)*1.E3);
  }

  free(P_br);
  free(A);
  free(A_buf);
  free(R);
}

int main(int argc, char **argv) {
  int myRank, numPes, n, b_min, b_max, b, num_iter;

  CommData_t *cdt_glb = (CommData_t*)malloc(sizeof(CommData_t));
  INIT_COMM(numPes, myRank, 1, cdt_glb);

  if (argc > 1) n = atoi(argv[1]);
  else n = 64;
  if (argc > 2) b_min = atoi(argv[2]);
  else b_min = 16;
  if (argc > 3) b_max = atoi(argv[3]);
  else b_max = 32;
  if (argc > 4) num_iter = atoi(argv[4]);
  else num_iter = 25;

  if (myRank == 0) {
    printf("benchmarking tournament pivoting panel of length %d for block sizes from b_min = %d to b_max = %d (p=%d)\n",n,b_min,b_max,numPes);
    printf("performing %d iterations\n", num_iter);
    printf("b\ttnmt(ms)\toutp(ms)\tpartial(ms)\tbarr(ms)\ttotal(ms)\n");
  }

  GLOBAL_BARRIER(cdt_glb);
  for (b = b_min; b <= b_max; b = b*2){
    par_tnmt_bench(n, b, myRank, numPes, 0, cdt_glb, num_iter);
  }
  GLOBAL_BARRIER(cdt_glb);
  COMM_EXIT;
  return 0;
}
