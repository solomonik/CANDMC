#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "../../shared/util.h"
#include "bitree_tsqr.h"


/**
 * \brief Apply Q^T to B, where Q^T is represented implicitly by Y and
 * tree_data, obtained by running binary tree TSQR
 *
 * \param[in] Y m-by-b matrix of Householder vectors from TSQR
 * \param[in] lda_Y lda of Y
 * \param[in] tau TAU values associated with the first level of TSQR HH vecs
 * \param[in,out] B m-by-k matrix to apply QT to
 * \param[in] lda_B length of leading dimension of B
 * \param[in] m number of rows in Y
 * \param[in] b number of columns in Y
 * \param[in] k number of columns of B to apply Y to
 * \param[in] myRank rank in cmunicator column
 * \param[in] numPes number of processes in column
 * \param[in] root the root of the tree (who will own R at the end)
 * \param[in] cdt MPI cmunicator for column
 * \param[in] tree_data TAU and Y data for the TSQR tree, 
 *                must be of size ((log2(p)+1)*b+b)/2-by-b
 **/
void apply_butterfly_tsqr_QT( double const *  Y,
                              int64_t const   lda_Y,
                              double const *  tau,
                              double *        B,
                              int64_t const   lda_B,
                              int64_t const   m,
                              int64_t const   b,
                              int64_t const   k,
                              int64_t const   myRank,
                              int64_t const   numPes,
                              int64_t const   root,
                              CommData_t      cdt,
                              double *        tree_data){
  int req_id = 0;
  int info;
  int64_t comm_pe, offset, mb, bmb, i, buf_sz, j, coff, np_work;
  double * T, * Y2, * buffer, * B_buf;
  
  mb = (m+root*b)/numPes;
  bmb = (m+root*b)/numPes;
  if (myRank < root){
    offset = (numPes-root)*mb+myRank*(mb-b);
    mb-=b;
  } else
    offset = (myRank-root)*mb;
  
  /* This is not an invarient butterfly, there is an up */
  int myr = myRank-root;
  if (myr < 0)
    myr += numPes;
  
  /* determine the number of processors that have a block involved */
  if (numPes * b > m){
    np_work = m/b;
  } else 
    np_work = numPes;
    
  /* find the smallest power of two less than or equal to number of working processors */
  int inp = np_work;
  int pow2_np_work = 1;
  while (inp > 1){
    pow2_np_work = pow2_np_work*2;
    inp = inp/2;
  }
  assert(pow2_np_work <= np_work);

  buf_sz = mb*MAX(k,b);
  
  buffer = (double*)malloc(sizeof(double)*buf_sz);
  // exit early if only one process involved
  if (np_work==1){
    if (myr == 0){
      TAU_FSTART(apply_tsqr_QT_local);
      cdormqr('L', 'T', m, k, b, Y, lda_Y, tau, B, lda_B, buffer, buf_sz, &info);
      TAU_FSTOP(apply_tsqr_QT_local);
    }
    free(buffer);
    return;
  }
  
  Y2 = (double*)malloc(sizeof(double)*b*b*2);
  B_buf = (double*)malloc(sizeof(double)*b*k*2);
  T = (double*)malloc(sizeof(double)*b*b);
  std::fill(Y2, Y2+b*b*2, 0.0);
  
  if (myr < np_work){
    TAU_FSTART(apply_tsqr_QT_local);
      //cdormqr('L', 'T', mb, k, b, Y, lda_Y, tau, B, lda_B, buffer, buf_sz, &info);
    cdlarft('F', 'C', mb, b, Y, lda_Y, tau, T, b);
    cdlarfb('L', 'T', 'F', 'C', mb, k, b, Y, lda_Y, T, b, B, lda_B, buffer, k);
    TAU_FSTOP(apply_tsqr_QT_local);

    lda_cpy(b,k,lda_B,2*b,B,B_buf);
  }
    
  // if processor count now power of two do a special butterfly level
  if (np_work > pow2_np_work){
    TAU_FSTART(apply_TSQR_QT_clipped_wing);
    if ((myr >= pow2_np_work && myr < np_work) || (myr < np_work - pow2_np_work)){
      int parity = (myr>=pow2_np_work);
      int vcomm_pe = myr - parity*pow2_np_work + (1-parity)*pow2_np_work;
      int comm_pe = (vcomm_pe + root)%numPes;
      int kw = k/2 + (parity)*(k%2);
      int kp = k-kw;

      lda_cpy(b, kp, 2*b, b, B_buf+kw*(1-parity)*2*b, buffer);
      if (parity){
        lda_cpy(b, kw, 2*b, 2*b, B_buf+kp*2*b, B_buf+b);
      }
      MPI_Status stat;
      MPI_Sendrecv(buffer,      kp*b, MPI_DOUBLE, comm_pe, req_id + parity,
                   buffer+kp*b, kw*b, MPI_DOUBLE, comm_pe, req_id + 1-parity, 
                   cdt.cm, &stat);
      lda_cpy(b, kw, b, 2*b, buffer+kp*b, B_buf+b*(1-parity));

      memcpy(T, tree_data, b*MIN(b,TAU_BLK)*sizeof(double));
      tree_data += b*MIN(b,TAU_BLK);
      unpack_upper(tree_data, Y2+b, b, 2*b);
      tree_data += psz_upr(b);
      TAU_FSTART(cdtpmqrt);
      cdtpmqrt('L', 'T', b, kw, b, b, MIN(b,TAU_BLK), Y2+b, 2*b, T, MIN(b,TAU_BLK), B_buf, 2*b,
                   B_buf+b, 2*b, buffer, &info);
      TAU_FSTOP(cdtpmqrt);
      // immediately return rows to owner
      lda_cpy(b, kw, 2*b, b, B_buf+(1-parity)*b, buffer);
      MPI_Sendrecv(buffer,      kw*b, MPI_DOUBLE, comm_pe, req_id + parity,
                   buffer+kw*b, kp*b, MPI_DOUBLE, comm_pe, req_id + 1-parity, 
                   cdt.cm, &stat);
      if (parity){
        lda_cpy(b, kw, 2*b, b, B_buf+b, buffer);
        memcpy(B_buf+kp*b, buffer, sizeof(double)*kw*b);
        lda_cpy(b, kp, b, b, buffer+kw*b,  B_buf+kw*(1-parity)*b);
      } else
        lda_cpy(b, kp, b, 2*b, buffer+kw*b,  B_buf+kw*(1-parity)*2*b);
    }
#ifdef PROFILE
    MPI_Barrier(cdt.cm);
#endif
    TAU_FSTOP(apply_TSQR_QT_clipped_wing);
  } 

  if (pow2_np_work > 1){
    /* Ensure that the we have a power-of-2 number of processors involved, otherwise butterfly
        will be unbalanced when we apply Q^T */
    assert(m/b >= pow2_np_work && 1<<(int)log2(pow2_np_work) == pow2_np_work);
    double * Bk_sv = (double*)malloc(sizeof(double)*b*(k+(int)log2(pow2_np_work)));
    int pk_sv = 0;

    int kws[(int)log2(pow2_np_work)+1];
    int kps[(int)log2(pow2_np_work)+1];
    int kw = k;
    int ikw = 0;
    TAU_FSTART(apply_tsqr_QT_buttefly);
    if (myr < pow2_np_work){
      /* Tournament tree is a butterfly */
      for (int level=pow2_np_work; level>1; level=level/2){
        /* parity determines which buttefly wing this proc is on */
        int parity = (myr%level)>=level/2;
        /* comm_pe finds the other wing */
        int vcomm_pe = level*(myr/level) + (((myr%level)+(level/2))%level);
        int comm_pe = (vcomm_pe + root)%numPes;

        assert(kw>=2);
        int nkw = kw/2 + (parity)*(kw%2);
        int kp = kw-nkw;
        kw = nkw;
        
        kws[ikw] = kw;
        kps[ikw] = kp;
        ikw++;

        MPI_Status stat;
        lda_cpy(b, kp, 2*b, b, B_buf+kw*(1-parity)*2*b, buffer);
        if (parity){
          lda_cpy(b, kw, 2*b, 2*b, B_buf+kp*2*b, B_buf+b);
        }
        MPI_Sendrecv(buffer,      kp*b, MPI_DOUBLE, comm_pe, req_id + parity,
                     buffer+kp*b, kw*b, MPI_DOUBLE, comm_pe, req_id + 1-parity, 
                     cdt.cm, &stat);
        lda_cpy(b, kw, b, 2*b, buffer+kp*b, B_buf+b*(1-parity));

        memcpy(T, tree_data, b*MIN(b,TAU_BLK)*sizeof(double));
        tree_data += b*MIN(b,TAU_BLK);
        unpack_upper(tree_data, Y2+b, b, 2*b);
        tree_data += psz_upr(b);
        TAU_FSTART(cdtpmqrt);
        cdtpmqrt('L', 'T', b, kw, b, b, MIN(b,TAU_BLK), Y2+b, 2*b, T, MIN(b,TAU_BLK), B_buf, 2*b,
                     B_buf+b, 2*b, buffer, &info);
        lda_cpy(b, kw, 2*b, b, B_buf+b, Bk_sv+pk_sv);
        pk_sv += kw*b;
        TAU_FSTOP(cdtpmqrt);
      }
    }
#ifdef PROFILE
    MPI_Barrier(cdt.cm);
#endif
    TAU_FSTOP(apply_tsqr_QT_buttefly);
    TAU_FSTART(reverse_QT_buttefly);
    if (myr < pow2_np_work){
      lda_cpy(b, kws[ikw-1], 2*b, b, B_buf, buffer);
      memcpy(B_buf, buffer, b*kws[ikw-1]*sizeof(double));


      for (int level=2; level<=pow2_np_work; level*=2) {
        /* parity determines which buttefly wing this proc is on */
        int parity = (myr%level)>=level/2;
        /* comm_pe finds the other wing */
        int vcomm_pe = level*(myr/level) + (((myr%level)+(level/2))%level);
        int comm_pe = (vcomm_pe + root)%numPes;
       
        ikw--;
        kw=kws[ikw];
        int kp=kps[ikw];
        pk_sv -= kw*b;
        
        MPI_Status stat;
        if (parity){
          MPI_Sendrecv(B_buf,       kw*b, MPI_DOUBLE, comm_pe, req_id + parity,
                       buffer,      kp*b, MPI_DOUBLE, comm_pe, req_id + 1-parity, 
                       cdt.cm, &stat);
          memcpy(B_buf,buffer,kp*b*sizeof(double));
          memcpy(B_buf+kp*b,Bk_sv+pk_sv,kw*b*sizeof(double));
        } else {
          MPI_Sendrecv(Bk_sv+pk_sv,     kw*b, MPI_DOUBLE, comm_pe, req_id + parity,
                       B_buf+kw*b,      kp*b, MPI_DOUBLE, comm_pe, req_id + 1-parity, 
                       cdt.cm, &stat);
        }
      }
    }
    free(Bk_sv);
#ifdef PROFILE
    MPI_Barrier(cdt.cm);
#endif
    TAU_FSTOP(reverse_QT_buttefly);
  }
  if (myr < np_work)
    lda_cpy(b,k,b,lda_B,B_buf,B);

  free(T);
  free(B_buf);
  free(buffer);
  free(Y2);
}


