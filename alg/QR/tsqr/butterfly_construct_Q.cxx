#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "../../shared/util.h"
#include "butterfly_tsqr.h"
#include "bitree_tsqr.h"


/**
 * \brief Construct Q from butterfly TSQR over a (sub)-column of processors 
 *
 * \param[in] Y m-by-b matrix of Householder vectors from TSQR
 * \param[in] lda_Y lda of Y
 * \param[in] tau TAU values associated with the first level of TSQR HH vecs
 * \param[in,out] Q1 first k columns of Q (buffer should be prealloced and 
 *                                         preset if is_form_q set to 0)
 * \param[in] lda_Q length of leading dimension of Q1
 * \param[in] m number of rows in Y
 * \param[in] b number of columns in Y, number of columns of Q1 to compute
 * \param[in] myRank rank in communicator column
 * \param[in] numPes number of processes in column
 * \param[in] root the root of the tree (who will own R at the end)
 * \param[in] cdt MPI communicator for column
 * \param[in] tree_data TAU and Y data for the TSQR tree, 
 *                must be of size ((log2(p)+1)*b+b)/2-by-b
 **/
void butterfly_construct_Q1(double const *  Y,
                            int64_t         lda_Y,
                            double const *  tau,
                            double *        Q1,
                            int64_t         lda_Q,
                            int64_t         m,
                            int64_t         b,
                            int64_t         myRank,
                            int64_t         numPes,
                            int64_t         root,
                            CommData_t      cdt,
                            double *        tree_data){
  int req_id = 0;
  int info;
  int64_t  myr, mb, bmb, i, buf_sz, j, np_work;
  int64_t * nps;
  double * T, * Y2, * buffer, * Q1_buf;

  mb = ((m/b)/numPes)*b;
  if ((m/b) % numPes > (myRank + numPes - root) % numPes) mb+=b;
  if ((m/b) % numPes > 0)
    bmb = mb+b;
  else
    bmb = mb;
 
  /* This is not an invarient butterfly, there is an up */
  myr = myRank-root;
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
  int nlvl = 0;
  while (inp > 1){
    pow2_np_work = pow2_np_work*2;
    inp = inp/2;
    nlvl++;
  }
  assert(pow2_np_work <= np_work);


  if (mb <= 0){
    return;
  } else {
    myr = myRank-root;
    if (myr < 0)
      myr += numPes;
  }
  if (numPes * b > m)
    np_work = m/b;
  else 
    np_work = numPes;
  buf_sz = MAX(psz_upr(b), mb*b);
  
  buffer = (double*)malloc(sizeof(double)*buf_sz);

  assert(mb == lda_Q);
  std::fill(Q1, Q1+mb*b, 0.0);


  // exit early if only one process involved
  if (np_work == 1){
    if (myr >= 0){
      for (i=0; i<mb; i++){
        if (i < b)
          Q1[(i)*mb+i] = 1.0;
      }
      // Q1 = Q[I,0]^T
      // copy Y to Q1 (use dlacopy instead?)
     //memcpy(Q1, Y, m*k*sizeof(double));
     // generate Q1 in place
     lda_cpy(m,b,lda_Y,lda_Q,Y,Q1);
     cdorgqr(m, b, b, Q1, lda_Q, tau, buffer, buf_sz, &info);
    }
    free(buffer);
    return;
  }
  
  Y2 = (double*)malloc(sizeof(double)*b*b*2);
  Q1_buf = (double*)malloc(sizeof(double)*b*b*2);
  T = (double*)malloc(sizeof(double)*b*b);
  nps = (int64_t*)malloc(sizeof(int64_t)*(numPes+5));
  std::fill(Y2, Y2+b*b*2, 0.0);
  int id_cols = b/pow2_np_work;
  if (b%pow2_np_work > myr) id_cols++;

  int st_id_cols = (b/pow2_np_work)*myr+MIN(myr,b%pow2_np_work);

  std::fill(Q1_buf, Q1_buf+b*b*2, 0.0);
  //move down columns
  Q1_buf+=st_id_cols*2*b;
  for (i=0; i<id_cols; i++){
    Q1_buf[i*2*b+st_id_cols+i] = 1.0;
  }

  tree_data += (b*MIN(b,TAU_BLK)+psz_upr(b))*nlvl;

  int k = b/pow2_np_work;
  if (b%pow2_np_work > 0) k++;
   
  TAU_FSTART(constructQ_butterfly);
  if (pow2_np_work > 1 && myr < pow2_np_work){
    /* Tournament tree is a butterfly */
    for (int level=2; level<=pow2_np_work; level=level*2){
      /* parity determines which buttefly wing this proc is on */
      int parity = (myr%level)>=level/2;
      /* comm_pe finds the other wing */
      int vcomm_pe = level*(myr/level) + (((myr%level)+(level/2))%level);
      int comm_pe = (vcomm_pe + root)%numPes;
        
      tree_data -= psz_upr(b);
      unpack_upper(tree_data, Y2+b, b, 2*b);
      tree_data -= b*MIN(b,TAU_BLK);
      memcpy(T, tree_data, b*MIN(b,TAU_BLK)*sizeof(double));
  
      cdormqr('L', 'N', 2*b, k, b, Y2, 2*b, T, Q1_buf, 2*b, 
              buffer, buf_sz, &info);
      
      lda_cpy(b,k,2*b,b,Q1_buf+b,buffer);

      MPI_Status stat;
      MPI_Sendrecv(buffer,     k*b, MPI_DOUBLE, comm_pe, req_id+parity,
                   buffer+k*b, k*b, MPI_DOUBLE, comm_pe, req_id+1-parity, 
                   cdt.cm, &stat);

      if (parity == 1){
        Q1_buf -= k*2*b;
        lda_cpy(b,k,b,2*b,buffer+k*b,Q1_buf);
      } else {
        lda_cpy(b,k,b,2*b,buffer+k*b,Q1_buf+k*2*b);
      }
      k*=2;
    }
  }
#ifdef PROFILE
  MPI_Barrier(cdt.cm);
#endif
  TAU_FSTOP(contructQ_butterfly);



  // if processor count now power of two do one clipped butterfly level
  TAU_FSTART(constructQ_clipped_wing);
  if (np_work > pow2_np_work){
    /* Tournament tree is a butterfly */
    if ((myr >= pow2_np_work && myr < np_work) || (myr < np_work - pow2_np_work)){
      int parity = (myr>=pow2_np_work);
      int vcomm_pe = myr - parity*pow2_np_work + (1-parity)*pow2_np_work;
      int comm_pe = (vcomm_pe + root)%numPes;
              
      tree_data -= psz_upr(b);
      unpack_upper(tree_data, Y2+b, b, 2*b);
      tree_data -= b*MIN(b,TAU_BLK);
      memcpy(T, tree_data, b*MIN(b,TAU_BLK)*sizeof(double));
     
      cdormqr('L', 'N', 2*b, k, b, Y2, 2*b, T, Q1_buf, 2*b, 
              buffer, buf_sz, &info);
      
      lda_cpy(b,k,2*b,b,Q1_buf+b,buffer);

      MPI_Status stat;
      MPI_Sendrecv(buffer,     k*b, MPI_DOUBLE, comm_pe, req_id+parity,
                   buffer+k*b, k*b, MPI_DOUBLE, comm_pe, req_id+1-parity, 
                   cdt.cm, &stat);

      if (parity == 1){
        Q1_buf -= k*2*b;
        lda_cpy(b,k,b,2*b,buffer+k*b,Q1_buf);
      } else {
        lda_cpy(b,k,b,2*b,buffer+k*b,Q1_buf+k*2*b);
      }
      k*=2;
    }
  }
#ifdef PROFILE
  MPI_Barrier(cdt.cm);
#endif
  TAU_FSTOP(constructQ_clipped_wing);


  lda_cpy(b,b,2*b,lda_Q,Q1_buf,Q1);

  TAU_FSTART(Form_Q1);
  if (myr >= 0){
    // apply local Householder vectors to upper triangular matrix
    // old way (ignores triangular structure of C)

    // set inner blocking factor
    // new way
    int vecs, nb = TAU_BLK;//16;
    for (j=(b/nb)*nb; j>=0; j=j-nb){
      vecs = MIN(b-j,nb);
      cdormqr('L', 'N', mb-j, b-j, vecs, Y+lda_Y*j+j, lda_Y, tau+j, Q1+lda_Q*j+j, lda_Q, buffer, buf_sz, &info);
    }
   
  }
  TAU_FSTOP(Form_Q1);

  free(T);
  free(Q1_buf);
  free(buffer);
  free(nps);
  free(Y2);
}


