#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include "topo_pdgemm_algs.h"
#include "../../shared/util.h"

#define __STGR_TAG_X    101
#define __STGR_TAG_Y    102
#define __SHFT_TAG_X    201
#define __SHFT_TAG_Y    202

#ifndef ASSERT
#define ASSERT(...)             \
  do{                           \
    assert(__VA_ARGS__);        \
  } while (0)
#endif

#ifdef DEBUG
int my_pe;
#endif


/* returns the number of bytes of buffer space
   we need */
static 
int64_t buffer_space_req(int64_t b, int64_t overlap){
  if (overlap)
    return 5*b*b*sizeof(double);
  else
    return 3*b*b*sizeof(double);
}

//inline static 
void bcast_cannon_4d(ctb_args_t const   * args,
                     double             * mat_A,
                     double             * mat_B,
                     double             * mat_C,
                     double             * buffer,
                     CommData_t         cdt_x1,
                     CommData_t         cdt_y1,
                     CommData_t         cdt_x2,
                     CommData_t         cdt_y2){
  int i1, i2;

  const int x1_rank = cdt_x1.rank;
  const int y1_rank = cdt_y1.rank;
  const int x2_rank = cdt_x2.rank;
  const int y2_rank = cdt_y2.rank;
  const int x1_np = cdt_x1.np;
  const int y1_np = cdt_y1.np;
  const int x2_np = cdt_x2.np;
  const int y2_np = cdt_y2.np;

  const int ovp = args->ovp;

  const int64_t n      = args->n;
  const int64_t b      = n / (x1_np*x2_np);

#ifdef DEBUG
  my_pe = x1_rank + x1_np*(y1_rank+y1_np*(x2_rank+x2_np*y2_rank));
#endif

  /* make sure we have enough buffer space */
  ASSERT(args->buffer_size >= buffer_space_req(b,ovp));
  ASSERT(x1_np == y1_np);
  ASSERT(x2_np == y2_np);
  ASSERT(n % (x1_np*x2_np) == 0);
 
  MPI_Request breq1, breq2; 
  MPI_Request req_x0, req_x1; 
  MPI_Request req_y0, req_y1; 
  MPI_Status stat;

  double * loc_A, * loc_B, * buf_A, * buf_B, * tmp_X; 
  double * mul_A, * mul_B, * swp_A, * swp_B;
  loc_A   = buffer;
  loc_B   = loc_A+b*b;
  if (ovp){
    swp_A = loc_B+b*b;
    swp_B = swp_A+b*b;
  }

  /* Get the matrices out of their lda, because we need to 
     send them first anyway */
  if (args->lda_A != b) {
    lda_cpy(b,b,args->lda_A,b,mat_A,loc_A);
    buf_A = mat_A;
  } else {
    buf_A = loc_A;
    loc_A = mat_A;
  }
  if (args->lda_B != b) {
    lda_cpy(b,b,args->lda_B,b,mat_B,loc_B);
    buf_B = mat_B;
  } else {
    buf_B = loc_B;
    loc_B = mat_B;
  }

  /* stagger the matrix */
  const int x2_stgr_tgt = WRAP((x2_rank-y2_rank),x2_np);
  const int y2_stgr_tgt = WRAP((y2_rank-x2_rank),y2_np);
  
  const int x2_stgr_src = WRAP((x2_rank+y2_rank),x2_np);
  const int y2_stgr_src = WRAP((y2_rank+x2_rank),y2_np);

  if (y2_np > 1){
    if (x2_rank != x2_stgr_tgt){
      //POST_RECV(buf_A, b*b, MPI_DOUBLE, x2_stgr_src, req_x0, cdt_x2, __STGR_TAG_X);
      MPI_Irecv(buf_A, b*b, MPI_DOUBLE, x2_stgr_src, __STGR_TAG_X, cdt_x2.cm, &req_x0);
    } 
    if (y2_rank != y2_stgr_tgt){
      //POST_RECV(buf_B, b*b, MPI_DOUBLE, y2_stgr_src, req_y0, cdt_y2, __STGR_TAG_Y);
      MPI_Irecv(buf_B, b*b, MPI_DOUBLE, y2_stgr_src, __STGR_TAG_X, cdt_y2.cm, &req_y0);
    }

    if (y2_rank != y2_stgr_tgt){
      //ISEND(loc_B, b*b, MPI_DOUBLE, y2_stgr_tgt, req_y1, cdt_y2, __STGR_TAG_Y);
      MPI_Isend(loc_B, b*b, MPI_DOUBLE, y2_stgr_tgt, __STGR_TAG_Y, cdt_y2.cm, &req_y1);
      MPI_Wait(&req_y0, &stat); 
      MPI_Wait(&req_y1, &stat);
      tmp_X = buf_B, buf_B = loc_B, loc_B = tmp_X;
    }
    if (x2_rank != x2_stgr_tgt){
      //ISEND(loc_A, b*b, MPI_DOUBLE, x2_stgr_tgt, req_x1, cdt_x2, __STGR_TAG_X);
      MPI_Isend(loc_A, b*b, MPI_DOUBLE, x2_stgr_tgt, __STGR_TAG_Y, cdt_x2.cm, &req_x1);
      MPI_Wait(&req_y0, &stat);
      MPI_Wait(&req_y1, &stat);
      tmp_X = buf_A, buf_A = loc_A, loc_A = tmp_X;
    }
  }

  MPI_Barrier(cdt_x2.cm); 
  MPI_Barrier(cdt_y2.cm); 

  const int x2_shft_tgt = WRAP((x2_rank-1),x2_np);
  const int y2_shft_tgt = WRAP((y2_rank-1),y2_np);
  
  const int x2_shft_src = WRAP((x2_rank+1),x2_np);
  const int y2_shft_src = WRAP((y2_rank+1),y2_np);
  

  for (i2=0; i2<y2_np; i2++){
    for (i1=0; i1<y1_np; i1++){
      MPI_Barrier(cdt_x1.cm); 
      MPI_Barrier(cdt_y1.cm); 
      if (x1_rank == i1){
        POST_BCAST(loc_A, b*b, MPI_DOUBLE, i1, cdt_x1, breq1);
      } else {
        POST_BCAST(buf_A, b*b, MPI_DOUBLE, i1, cdt_x1, breq1);
      }
      if (y1_rank == i1){
        POST_BCAST(loc_B, b*b, MPI_DOUBLE, i1, cdt_y1, breq2);
      } else {
        POST_BCAST(buf_B, b*b, MPI_DOUBLE, i1, cdt_y1, breq2);
      }
      if (ovp && i1 > 0){
        cdgemm(args->trans_A, args->trans_B, b, b, b, 
               1.0, mul_A, b, mul_B, b, (i1>1 || i2>0)*1.0, mat_C, args->lda_C); 
      }
      if (x1_rank == i1)
        mul_A = loc_A;
      else
        mul_A = buf_A;
      if (y1_rank == i1)
        mul_B = loc_B;
      else 
        mul_B = buf_B;
      if (ovp){
        tmp_X = buf_A, buf_A = swp_A, swp_A = tmp_X;
        tmp_X = buf_B, buf_B = swp_B, swp_B = tmp_X;
      }

      WAIT_BCAST(cdt_x1, breq1);
      WAIT_BCAST(cdt_y1, breq2);
      DEBUG_PRINTF("[%d][%d][%d][%d] multiplying %lf by %lf\n",
             cdt_x2.rank, cdt_y2.rank,
             cdt_x1.rank, cdt_y1.rank,
             mul_A[0], mul_B[0]);
      if (!ovp){
        cdgemm(args->trans_A, args->trans_B, b, b, b, 
               1.0, mul_A, b, mul_B, b, (i1>0 || i2>0)*1.0, mat_C, args->lda_C); 
      }
    }
    if (ovp){
      cdgemm(args->trans_A, args->trans_B, b, b, b, 
             1.0, mul_A, b, mul_B, b, (i1>1 || i2>0)*1.0, mat_C, args->lda_C); 
    }
    if (i2<y2_np-1){
      //POST_RECV(buf_A, b*b, MPI_DOUBLE, x2_shft_src, req_x0, cdt_x2, __SHFT_TAG_X);
      //POST_RECV(buf_B, b*b, MPI_DOUBLE, y2_shft_src, req_y0, cdt_y2, __SHFT_TAG_Y);
      MPI_Irecv(buf_A, b*b, MPI_DOUBLE, x2_shft_src, __SHFT_TAG_X, cdt_x2.cm, &req_x0);
      MPI_Irecv(buf_B, b*b, MPI_DOUBLE, y2_shft_src, __SHFT_TAG_Y, cdt_y2.cm, &req_y0);
      
      MPI_Barrier(cdt_x2.cm); 
      MPI_Barrier(cdt_y2.cm); 
      
      MPI_Isend(loc_A, b*b, MPI_DOUBLE, x2_shft_tgt, __SHFT_TAG_X, cdt_x2.cm, &req_x1);
      MPI_Isend(loc_B, b*b, MPI_DOUBLE, y2_shft_tgt, __SHFT_TAG_Y, cdt_y2.cm, &req_y1);
      MPI_Wait(&req_x0, &stat);
      MPI_Wait(&req_x1, &stat);
      MPI_Wait(&req_y0, &stat); 
      MPI_Wait(&req_y1, &stat); 
    
      tmp_X = buf_A, buf_A = loc_A, loc_A = tmp_X;
      tmp_X = buf_B, buf_B = loc_B, loc_B = tmp_X;
    }
  }
}

/*
template void d25_topo_bcast_tmp<0>(ctb_args_t const*, double *, double *,
                                    double *, double *, CommData_t,
                                    CommData_t, CommData_t);

template void d25_topo_bcast_tmp<1>(ctb_args_t const*, double *, double *,
                                    double *, double *, CommData_t,
                                    CommData_t, CommData_t);

void d25_topo_bcast(ctb_args_t const    * args,
                    double              * mat_A,
                    double              * mat_B,
                    double              * mat_C,
                    double              * buffer,
                    CommData_t          cdt_row,
                    CommData_t          cdt_col,
                    CommData_t          cdt_kdir){
  d25_topo_bcast_tmp<0>(args,mat_A,mat_B,mat_C,buffer,cdt_row,cdt_col,cdt_kdir);
}

void d25_topo_bcast_ovp(ctb_args_t const        * args,
                        double                  * mat_A,
                        double                  * mat_B,
                        double                  * mat_C,
                        double                  * buffer,
                        CommData_t              cdt_row,
                        CommData_t              cdt_col,
                        CommData_t              cdt_kdir){
  d25_topo_bcast_tmp<1>(args,mat_A,mat_B,mat_C,buffer,cdt_row,cdt_col,cdt_kdir);
}

*/




