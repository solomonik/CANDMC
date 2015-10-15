/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include "topo_pdgemm_algs.h"
#include "../../shared/util.h"

#ifndef ASSERT
#define ASSERT(...)             \
  do{                           \
    assert(__VA_ARGS__);        \
  } while (0)
#endif

/* returns the number of bytes of buffer space
   we need */
static 
int64_t buffer_space_req(int64_t b){
  return 4*b*b*sizeof(double);
}

void summa(ctb_args_t const * args,
                       double const     * mat_A,
                       double const     * mat_B,
                       double           * mat_C,
                       double           * buffer,
                       CommData_t       cdt_row,
                       CommData_t       cdt_col){
  int64_t i;

  const int np_row = cdt_col.np;
  const int np_col = cdt_row.np;
  const int my_row = cdt_col.rank;
  const int my_col = cdt_row.rank;
  
  const int64_t n      = args->n;
  const int64_t b      = n / np_col;

  /* make sure we have enough buffer space */
  ASSERT(args->buffer_size >= buffer_space_req(b));
  ASSERT(np_row == np_col);
  ASSERT(n % np_row == 0);

  double * loc_A   = buffer;
  double * loc_B   = loc_A+b*b;
  double * buf_A   = loc_B+b*b;
  double * buf_B   = buf_A+b*b;

  lda_cpy(b,b,args->lda_A,b,mat_A,loc_A);
  lda_cpy(b,b,args->lda_B,b,mat_B,loc_B);

  MPI_Request req1, req2;

  TAU_FSTART(d2_topo_bcast_gemm);
  for (i=0; i<np_col; i++){
    if (my_col == i){
      RANK_PRINTF((my_row*np_col+my_col), (my_row*np_col+my_col),
                  "POSTING BCAST\n");
      POST_BCAST(loc_A, b*b, MPI_DOUBLE, i, cdt_row, req1);
      RANK_PRINTF((my_row*np_col+my_col), (my_row*np_col+my_col),
                  "POSTED BCAST\n");
      buf_A = loc_A;
    } else {
      RANK_PRINTF((my_row*np_col+my_col), (my_row*np_col+my_col),
                  "POSTING BCAST\n");
      POST_BCAST(buf_A, b*b, MPI_DOUBLE, i, cdt_row, req1);
      RANK_PRINTF((my_row*np_col+my_col), (my_row*np_col+my_col),
                  "POSTED BCAST\n");
    }
    if (my_row == i){
      RANK_PRINTF((my_row*np_col+my_col), (my_row*np_col+my_col),
                  "POSTING BCAST\n");
      POST_BCAST(loc_B, b*b, MPI_DOUBLE, i, cdt_col, req2);
      RANK_PRINTF((my_row*np_col+my_col), (my_row*np_col+my_col),
                  "POSTED BCAST\n");
      buf_B = loc_B;
    } else {
      RANK_PRINTF((my_row*np_col+my_col), (my_row*np_col+my_col),
                  "POSTING BCAST\n");
      POST_BCAST(buf_B, b*b, MPI_DOUBLE, i, cdt_col, req2);
      RANK_PRINTF((my_row*np_col+my_col), (my_row*np_col+my_col),
                  "POSTED BCAST\n");
    }
      RANK_PRINTF((my_row*np_col+my_col), (my_row*np_col+my_col),
                  "WAIT ATTEMPT\n");
      WAIT_BCAST(cdt_row, req1);
      RANK_PRINTF((my_row*np_col+my_col), (my_row*np_col+my_col),
                  "WAIT SUCCESS\n");
      WAIT_BCAST(cdt_col, req2);
      RANK_PRINTF((my_row*np_col+my_col), (my_row*np_col+my_col),
                  "WAIT SUCCESS\n");
    cdgemm(args->trans_A, args->trans_B, 
           b, b, b, 1.0, buf_A, b, buf_B, b, (i>0)*1.0, mat_C, args->lda_C); 

  }
  TAU_FSTOP(d2_topo_bcast_gemm);
}






