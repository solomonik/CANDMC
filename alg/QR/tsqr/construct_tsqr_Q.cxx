/* A part of this routines was written by Grey Ballard */

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "../../shared/util.h"
#include "bitree_tsqr.h"


/**
 * \brief Perform TSQR over a (sub)-column of processors 
 *
 * \param[in] Y m-by-b matrix of Householder vectors from TSQR
 * \param[in] lda_Y lda of Y
 * \param[in] tau TAU values associated with the first level of TSQR HH vecs
 * \param[in,out] Q1 first k columns of Q (buffer should be prealloced and 
 *                                         preset if is_form_q set to 0)
 * \param[in] lda_Q length of leading dimension of Q1
 * \param[in] m number of rows in Y
 * \param[in] b number of columns in Y
 * \param[in] k number of columns of Q1 to compute
 * \param[in] myRank rank in communicator column
 * \param[in] numPes number of processes in column
 * \param[in] root the root of the tree (who will own R at the end)
 * \param[in] cdt MPI communicator for column
 * \param[in] tree_data TAU and Y data for the TSQR tree, 
 *                must be of size ((log2(p)+1)*b+b)/2-by-b
 * \param[in] ID b-by-k matrix which replaces the identity on the tree root if not NULL
 **/
void construct_Q1(double const *  Y,
                  int64_t const   lda_Y,
                  double const *  tau,
                  double *        Q1,
                  int64_t const   lda_Q,
                  int64_t const   m,
                  int64_t const   b,
                  int64_t const   k,
                  int64_t const   myRank,
                  int64_t const   numPes,
                  int64_t const   root,
                  CommData_t      cdt,
                  double *        tree_data,
                  double *        ID){
  int req_id = 0;
  int info, mylvl;
  int64_t comm_pe, np, myr, offset, mb, bmb, i,nlvl, buf_sz, j, np_work, coff;
  int64_t * nps;
  double * T, * Y2, * buffer, * Q1_buf;
  
  mb = ((m/b)/numPes)*b;
  if ((m/b) % numPes > 0)
    bmb = mb+b;
  else
    bmb = mb;
  if ((m/b) % numPes > (myRank + numPes - root) % numPes) mb+=b;
  if (myRank < root){
    offset = (numPes-root)*mb+myRank*(mb-b);
  } else
    offset = (myRank-root)*mb;

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
  buf_sz = MAX(psz_upr(b)+(k-b)*b, mb*MAX(k,b));
  
  buffer = (double*)malloc(sizeof(double)*buf_sz);

  //assert(mb == lda_Q);
  //std::fill(Q1, Q1+mb*k, 0.0);
  lda_zero(mb, k, lda_Q, Q1);
  


  // exit early if only one process involved
  if (np_work == 1){

    if (myr >= 0){
      if (ID == NULL){
        for (i=0; i<mb; i++){
          if (offset+i < k)
            Q1[(offset+i)*lda_Q+i] = 1.0;
        }
        lda_cpy(m,k,lda_Y,lda_Q,Y,Q1);
        cdorgqr(m, k, b, Q1, lda_Q, tau, buffer, buf_sz, &info);
      } else  {
        lda_cpy(b,k,b,lda_Q,ID,Q1);
        cdormqr('L', 'N', m, b, k, Y, lda_Y, tau, Q1, lda_Q, buffer, buf_sz, &info);
      }
      // Q1 = Q[I,0]^T
      // copy Y to Q1 (use dlacopy instead?)
     //memcpy(Q1, Y, m*k*sizeof(double));
     // generate Q1 in place
    }
    free(buffer);
    return;
  }
  
  Y2 = (double*)malloc(sizeof(double)*b*b*2);
  Q1_buf = (double*)malloc(sizeof(double)*b*k*2);
  T = (double*)malloc(sizeof(double)*b*b);
  nps = (int64_t*)malloc(sizeof(int64_t)*(numPes+5));
  std::fill(Y2, Y2+b*b*2, 0.0);
  

  std::fill(Q1_buf, Q1_buf+k*b*2, 0.0);
  if (ID == NULL){
    for (i=0; i<b; i++){
      if (offset+i < k)
        Q1_buf[(offset+i)*2*b+i] = 1.0;
    }
  } else {
    assert(b==k);
    if (myRank == root){
      lda_cpy(b,k,b,2*b,ID,Q1_buf);
    }
  }


  /* Tournament tree is a butterfly */
  mylvl=0;
  nlvl=0;
  for (np = np_work; np > 1; np = np/2+(np%2)){
    nps[nlvl] = np;
    nlvl++;
    if ((myr > np/2 || myr*2 == np)  && myr < np ){
    } else if (myr < np/2 && myr >= 0) {
      mylvl++;
    }
  }
  if (tree_data != NULL){
    tree_data += mylvl*(b*MIN(b,TAU_BLK)+psz_upr(b));
  }

  TAU_FSTART(ctQ_tree);
  for (i=0; i<nlvl; i++){
    np = nps[nlvl-i-1];
    /* If I am in second half of processor list send my data to lower half */
    if ((myr > np/2 || myr*2 == np)  && myr < np ){
      comm_pe = myr-(np+1)/2;
      comm_pe = comm_pe + root;
      if (comm_pe >= numPes)
        comm_pe = comm_pe - numPes;

      if ((np%2 == 0 || myr != np/2)){
        if (tree_data == NULL){
          pack_upper(Y, buffer, b, lda_Y);
          MPI_Send(buffer, psz_upr(b), MPI_DOUBLE, comm_pe, req_id, cdt.cm);
        }
        MPI_Status stat;
        MPI_Recv(buffer, psz_upr(b)+(k-b)*b, MPI_DOUBLE, comm_pe, req_id, cdt.cm, &stat);
        unpack_upper(buffer,Q1_buf,b,2*b);
        lda_cpy(b, k-b, b, 2*b, buffer+psz_upr(b), Q1_buf+2*b*b);
      }
    } else if (myr < np/2 && myr >= 0) {
      TAU_FSTART(ctQ_tree_worker);
      comm_pe = myr+(np+1)/2;
      comm_pe = comm_pe + root;
      if (comm_pe >= numPes){
        comm_pe = comm_pe - numPes;
        coff = comm_pe*(bmb-b)+(numPes-root)*bmb;
      } else
        coff = (comm_pe-root)*bmb;
      for (j=0; j<b; j++){
        if (coff+j < k)
          Q1_buf[(coff+j)*2*b+b+j] = 1.0;
      }
      if (tree_data == NULL){
        MPI_Status stat;
        MPI_Recv(buffer, psz_upr(b), MPI_DOUBLE, comm_pe, req_id, cdt.cm, &stat);
        unpack_upper(buffer, Y2+b, b, 2*b);
        tau_recon('U', b, b, 2*b, Y2+b, T);
        cdormqr('L', 'N', 2*b, k, b, Y2, 2*b, T, Q1_buf, 2*b, 
                buffer, buf_sz, &info);
      } else {
        tree_data -= psz_upr(b);
        unpack_upper(tree_data, Y2+b, b, 2*b);
        //tree_data -= psz_upr(b);
        //unpack_upper(tree_data, T, b, b);
        tree_data -= b*MIN(b,TAU_BLK);
        memcpy(T, tree_data, b*MIN(b,TAU_BLK)*sizeof(double));
        TAU_FSTART(cdtpmqrt);
        // Grey Ballard wrote initial implementation of this loop
        int kf = ((b-1)/TAU_BLK)*TAU_BLK;
        for (j=kf; j>=0; j-=TAU_BLK){
          int ib = MIN( TAU_BLK, b-j );  
          int kb = MIN( j+ib, b );
          int lb;
          if( j>=b ) lb = 0;
          else lb = kb-j;
          cdtprfb('L', 'N', 'F', 'C', kb, k-j, ib, lb,
                  Y2+2*j*b+b, 2*b, 
                  T+j*MIN(b,TAU_BLK), MIN(b,TAU_BLK), 
                  Q1_buf+j+j*2*b, 2*b, 
                  Q1_buf+b+j*2*b, 2*b, 
                  buffer, ib );
        }
        TAU_FSTOP(cdtpmqrt);
      }
      pack_upper(Q1_buf+b,buffer,b,2*b);
      lda_cpy(b,k-b,2*b,b,Q1_buf+(2*b+1)*b,buffer+psz_upr(b));
      MPI_Send(buffer, psz_upr(b)+(k-b)*b, MPI_DOUBLE, comm_pe, req_id, cdt.cm);
      lda_zero(b,k,2*b,Q1_buf+b);
      TAU_FSTOP(ctQ_tree_worker);
    }
  }
  TAU_FSTOP(ctQ_tree);
  lda_cpy(b,k,2*b,lda_Q,Q1_buf,Q1);

  TAU_FSTART(Form_Q1);
  if (myr >= 0){
    // apply local Householder vectors to upper triangular matrix
    // old way (ignores triangular structure of C)

    // set inner blocking factor
    // new way
    int vecs, nb = TAU_BLK;//16;
    for (j=(b/nb)*nb; j>=0; j=j-nb){
      vecs = MIN(b-j,nb);
      cdormqr('L', 'N', mb-j, k-j, vecs, Y+lda_Y*j+j, lda_Y, tau+j, Q1+lda_Q*j+j, lda_Q, buffer, buf_sz, &info);
    }
   
  }
  TAU_FSTOP(Form_Q1);

  free(T);
  free(Q1_buf);
  free(buffer);
  free(nps);
  free(Y2);
}


