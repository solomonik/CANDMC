#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include "topo_pdgemm_algs.h"
#include "../../shared/util.h"

#ifdef	USE_MIC
#include <mkl.h>
#endif

#ifndef ASSERT
#define ASSERT(...)             \
  do{                           \
    assert(__VA_ARGS__);        \
  } while (0)
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

template <int64_t overlap>
inline static
void d25_summa_tmp(ctb_args_t const        * args,
                        double                  * mat_A,
                        double                  * mat_B,
                        double                  * mat_C,
                        double                  * buffer,
#ifdef USE_MIC
                        int                        mic_portion,
                        int                        mic_id,
#endif
                        CommData_t              cdt_row,
                        CommData_t              cdt_col,
                        CommData_t              cdt_kdir){
  int64_t i;

  const int np_row = cdt_col.np;
  const int np_col = cdt_row.np;
  const int c_rep  = cdt_kdir.np;
  const int my_row = cdt_col.rank;
  const int my_col = cdt_row.rank;
  const int my_lr  = cdt_kdir.rank;

  const int64_t n      = args->n;
  const int64_t b      = n / np_col;

  /* make sure we have enough buffer space */
  ASSERT(args->buffer_size >= buffer_space_req(b,overlap));
  ASSERT(np_row == np_col);
  ASSERT(n % np_row == 0);
  ASSERT(np_row % c_rep == 0);

  MPI_Request req1, req2;

  double * loc_A, * loc_B, * buf_A, * buf_B, * buf_C;
  double * ovp_A, * ovp_B, * swp_X;
  loc_A   = buffer;
  loc_B   = loc_A+b*b;
  buf_C   = loc_B+b*b;

  if (overlap){
    ovp_A = buf_C+b*b;
    ovp_B = ovp_A+b*b;
  }

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

  int64_t * bid = (int64_t*)malloc(sizeof(int64_t));
  bid[0] = 0;

#ifdef USE_MIC
  int sig_wait, auxi;
  double *buf_A_mic, *buf_B_mic, *buf_C_mic;
  int lines_on_mic = mic_portion;
  double *mic_aux_chunk_A;
  double *mic_aux_chunk_B;
  double *mic_aux_chunk_C;
  assert(posix_memalign((void**)&mic_aux_chunk_A,
                  ALIGN_BYTES,
                 lines_on_mic*b*sizeof(double)) == 0);
  assert(posix_memalign((void**)&mic_aux_chunk_B,
              ALIGN_BYTES,
              lines_on_mic*b*sizeof(double)) == 0);
  assert(posix_memalign((void**)&mic_aux_chunk_C,
              ALIGN_BYTES,
              b*b*sizeof(double)) == 0);
  buf_A_mic = mic_aux_chunk_A;
  buf_B_mic = mic_aux_chunk_B;
  buf_C_mic = mic_aux_chunk_C;
  // Allocate the buffers on MIC
  #pragma offload_transfer target(mic:mic_id) \
    nocopy(buf_A_mic:length(lines_on_mic*b) alloc_if(1) free_if(0))  \
    nocopy(buf_B_mic:length(lines_on_mic*b) alloc_if(1) free_if(0))  \
    nocopy(buf_C_mic:length(b*b) alloc_if(1) free_if(0))
#endif

  TAU_FSTART(d25_summa_gemm);
  if (overlap){
    for (i=my_lr*(np_col/c_rep); i<(my_lr+1)*(np_col/c_rep); i++){
      if (my_col == i){
        POST_BCAST(loc_A, b*b, MPI_DOUBLE, i, cdt_row, req1);
        buf_A = loc_A;
      } else {
        POST_BCAST(buf_A, b*b, MPI_DOUBLE, i, cdt_row, req1);
      }
      if (my_row == i){
        POST_BCAST(loc_B, b*b, MPI_DOUBLE, i, cdt_col, req2);
        buf_B = loc_B;
      } else {
        POST_BCAST(buf_B, b*b, MPI_DOUBLE, i, cdt_col, req2);
      }
      if (i>0){
        cdgemm(args->trans_A, args->trans_B,
               b, b, b, 1.0, ovp_A, b, ovp_B, b, (i>0)*1.0, buf_C, args->lda_C);
      }
      swp_X = ovp_A, ovp_A = buf_A, buf_A = swp_X;
      swp_X = ovp_B, ovp_B = buf_B, buf_B = swp_X;
      WAIT_BCAST(cdt_row, req1);
      WAIT_BCAST(cdt_col, req2);

    }
    cdgemm(args->trans_A, args->trans_B,
           b, b, b, 1.0, ovp_A, b, ovp_B, b, (i>0)*1.0, buf_C, args->lda_C);
    MPI_Allreduce(buf_C, mat_C, b*b, MPI_DOUBLE, MPI_SUM, cdt_kdir.cm);
  } else {
    for (i=my_lr*(np_col/c_rep); i<(my_lr+1)*(np_col/c_rep); i++){
      if (my_col == i){
        POST_BCAST(loc_A, b*b, MPI_DOUBLE, i, cdt_row, req1);
        buf_A = loc_A;
      } else {
        POST_BCAST(buf_A, b*b, MPI_DOUBLE, i, cdt_row, req1);
      }
      if (my_row == i){
        POST_BCAST(loc_B, b*b, MPI_DOUBLE, i, cdt_col, req2);
        buf_B = loc_B;
      } else {
        POST_BCAST(buf_B, b*b, MPI_DOUBLE, i, cdt_col, req2);
      }
      WAIT_BCAST(cdt_row, req1);
      WAIT_BCAST(cdt_col, req2);
#ifdef USE_MIC
    // Do the lda copy to MIC aux buffer for B
    for (auxi = 0; auxi < b; auxi++) {
      memcpy(mic_aux_chunk_B+lines_on_mic*auxi, buf_B+b*auxi, lines_on_mic*sizeof(double));
    }
    memcpy(mic_aux_chunk_A, buf_A, lines_on_mic*b*sizeof(double));
    buf_B_mic = mic_aux_chunk_B;
    buf_A_mic = mic_aux_chunk_A;
    buf_C_mic = mic_aux_chunk_C;

      // Wait for the previous MIC computation to complete
      if (i > my_lr*(np_col/c_rep)) {
        #pragma offload_wait target(mic:mic_id) wait(&sig_wait)
      }

      // Post asynchronous MIC computation
    char trans_A = args->trans_A;
    char trans_B = args->trans_B;

    TAU_FSTART(mcdgemm_compute);
    #pragma offload target(mic:mic_id) signal(&sig_wait) \
    in(buf_A_mic:length(lines_on_mic*b) alloc_if(0) free_if(0))  \
    in(buf_B_mic:length(lines_on_mic*b) alloc_if(0) free_if(0))  \
    nocopy(buf_C_mic:length(b*b) alloc_if(0) free_if(0))
    {
      mcdgemm(trans_A, trans_B,
           b, b, lines_on_mic, 1.0, buf_A_mic, b, buf_B_mic, lines_on_mic, (i>0)*1.0, buf_C_mic, b);
    }
     TAU_FSTOP(mcdgemm_compute);
    // Do the other computation on HOST - the lower part of A and C
    if(lines_on_mic < b)
      cdgemm(args->trans_A, args->trans_B,
         b, b, b-lines_on_mic, 1.0, buf_A+lines_on_mic*b, b, buf_B+lines_on_mic, b, (i>0)*1.0, buf_C, args->lda_C);
#else
    cdgemm(args->trans_A, args->trans_B,
         b, b, b, 1.0, buf_A, b, buf_B, b, (i>0)*1.0, buf_C, args->lda_C);
#endif

    }

#ifdef USE_MIC
  // Wait for the last MIC computation to complete and transfer C from MIC to HOST
  #pragma offload_wait target(mic:mic_id) wait(&sig_wait)
  buf_C_mic = mic_aux_chunk_C;
  #pragma offload_transfer target(mic:mic_id) out(buf_C_mic:length(b*b) alloc_if(0) free_if(1))
  /* Sum MIC's C and HOST's C */
  //for (auxi = 0; auxi < b*b; auxi++) {
  //  buf_C[auxi] += buf_C_mic[auxi];
  //}
  int b_times_b = b*b;
  int one_int = 1;
  double one = 1.0;
  DAXPY(&b_times_b,&one,buf_C_mic,&one_int,buf_C,&one_int);
#endif

    MPI_Allreduce(buf_C, mat_C, b*b, MPI_DOUBLE, MPI_SUM, cdt_kdir.cm);
  }
  TAU_FSTOP(d25_summa_gemm);
}


#ifdef USE_MIC
template void d25_summa_tmp<0>(ctb_args_t const*, double *, double *,
                                    double *, double *, int, int, CommData_t,
                                    CommData_t, CommData_t);

template void d25_summa_tmp<1>(ctb_args_t const*, double *, double *,
                                    double *, double *, int, int, CommData_t,
                                    CommData_t, CommData_t);
#else
template void d25_summa_tmp<0>(ctb_args_t const*, double *, double *,
                                    double *, double *, CommData_t,
                                    CommData_t, CommData_t);

template void d25_summa_tmp<1>(ctb_args_t const*, double *, double *,
                                    double *, double *, CommData_t,
                                    CommData_t, CommData_t);
#endif

void d25_summa(ctb_args_t const    * args,
                    double              * mat_A,
                    double              * mat_B,
                    double              * mat_C,
                    double              * buffer,
#ifdef USE_MIC
                       int                    mic_portion,
                       int                    mic_id,
#endif
                    CommData_t          cdt_row,
                    CommData_t          cdt_col,
                    CommData_t          cdt_kdir){
  d25_summa_tmp<0>(args,mat_A,mat_B,mat_C,buffer,
#ifdef USE_MIC
                   mic_portion, mic_id,
#endif
                 cdt_row,cdt_col,cdt_kdir);
}

void d25_summa_ovp(ctb_args_t const        * args,
                        double                  * mat_A,
                        double                  * mat_B,
                        double                  * mat_C,
                        double                  * buffer,
#ifdef USE_MIC
                        int                        mic_portion,
                        int                     mic_id,
#endif
                        CommData_t              cdt_row,
                        CommData_t              cdt_col,
                        CommData_t              cdt_kdir){
  d25_summa_tmp<1>(args,mat_A,mat_B,mat_C,buffer,
#ifdef USE_MIC
                   mic_portion, mic_id,
#endif
                 cdt_row,cdt_col,cdt_kdir);
}






