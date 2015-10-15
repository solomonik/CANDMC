/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "../../shared/util.h"
#include "qr_2d.h"

/**
 * \brief computes triangular T from Y by dsyrk
 *
 * \param[in] Y Househodler vectors stored along root processor column, local size mb-by-b
 * \param[in] lda_Y lda of Y matrix
 * \param[in,out] T computed triangular T corresponding to Y
 * \param[in] lda_T lda of T matrix
 * \param[in] mb length of local piece of Householder vectors
 * \param[in] b number of Householder vectors
 * \param[in] pv current parallel view
 */
void compute_invT_from_Y(double const * Y,
                      int64_t        lda_Y,
                      double       * T,
                      int64_t        lda_T,
                      int64_t        mb,
                      int64_t        b,
                      pview *        pv){
  int64_t i;
  
  double * T_buf = (double*)malloc(sizeof(double)*psz_upr(b));
  /* TAU^-1 = Y^T*Y */
  TAU_FSTART(Form_Tinv);
  if (pv->crow.rank == pv->rcol){
    TAU_FSTART(Compute_T_from_Y);
    /* S_i = Y_i^T*Y_i */
    TAU_FSTART(DSYRK);
    if (mb > 0)
      cdsyrk('L','T',b,mb,1.0,Y,lda_Y,0.0,T,lda_T);
    TAU_FSTOP(DSYRK);
    /* T^-T = stril(sum_i S_i) */
    pack_lower(T, T_buf, b, lda_T, 1);
    TAU_FSTART(ALLRED_TAU);
    MPI_Allreduce(MPI_IN_PLACE, T_buf, psz_upr(b), MPI_DOUBLE, MPI_SUM, pv->ccol.cm);
    TAU_FSTOP(ALLRED_TAU);
    int off = 0;
    for (i=0; i<b; i++){
      /* tau = 2/yy^T */
      T_buf[off] = T_buf[off]/2.;
      off+=b-i;
    }
    TAU_FSTOP(Compute_T_from_Y);
  }
  TAU_FSTART(Bcast_TAU);
  MPI_Bcast(T_buf, psz_upr(b), MPI_DOUBLE, pv->rcol, pv->crow.cm);
  TAU_FSTOP(Bcast_TAU);

  unpack_lower(T_buf, T, b, lda_T, 1);
  TAU_FSTOP(Form_Tinv);
}

/**
 * \brief computes rectangular T from Y1 and Y2 by gemm
 *
 * \param[in] Y1 Househodler vectors stored along root processor column, local size mb-by-b1
 * \param[in] lda_Y1 lda of Y1 matrix
 * \param[in] Y2 Househodler vectors stored along root processor column, localsize mb-by-b2
 * \param[in] lda_Y2 lda of Y2 matrix
 * \param[in,out] T computed b1-by-b2 T corresponding to Y1^TY2
 * \param[in] lda_T lda of T matrix
 * \param[in] mb length of local piece of Householder vectors
 * \param[in] b1 number of Householder vectors in Y1
 * \param[in] b2 number of Householder vectors in Y2
 * \param[in] pv current parallel view
 */
void compute_invT_from_Y(double const * Y1,
                      int64_t        lda_Y1,
                      double const * Y2,
                      int64_t        lda_Y2,
                      double       * T,
                      int64_t        lda_T,
                      int64_t        mb,
                      int64_t        b1,
                      int64_t        b2,
                      pview *        pv){
  double * T_buf = (double*)malloc(sizeof(double)*b1*b2);
  /* TAU^-1 = Y^T*Y */
  if (pv->crow.rank == pv->rcol){
    TAU_FSTART(Compute_T_from_2Ys);
    /* S_i = Y_i^T*Y_i */
    TAU_FSTART(DGEMM_T_FROM_Y);
    if (mb > 0)
      cdgemm('T','N',b1,b2,mb,1.0,Y1,lda_Y1,Y2,lda_Y2,0.0,T_buf,b1);
    TAU_FSTOP(DGEMM_T_FROM_Y);
    /* T^-T = stril(sum_i S_i) */
    TAU_FSTART(ALLRED_TAU);
    MPI_Allreduce(MPI_IN_PLACE, T_buf, b1*b2, MPI_DOUBLE, MPI_SUM, pv->ccol.cm);
    TAU_FSTOP(ALLRED_TAU);
    TAU_FSTOP(Compute_T_from_2Ys);
  }
  TAU_FSTART(Bcast_TAU);
  MPI_Bcast(T_buf, b1*b2, MPI_DOUBLE, pv->rcol, pv->crow.cm);
  TAU_FSTOP(Bcast_TAU);

  lda_cpy(b1,b2,b1,lda_T,T_buf,T);
}

/**
 * \brief Perform (I-YTY^T)A 
 *
 * \param[in,out] Y m-by-k dense lower-triangular matrix of HH vecs
 * \param[in] lda_Y lda of Y
 * \param[in,out] A m-by-k dense matrix of HH vecs
 * \param[in] lda_A lda of A
 * \param[in] m number of rows in A
 * \param[in] k number of columns in A
 * \param[in] b number of Householder vectors
 * \param[in] W is either null or Y1^TT
 * \param[in] pv current processor grid view
 * \param[in] aggreg_Y is either null or a buffer into which we should save Ybuf 
 * \param[in] lda_aY lda of aggreg_Y if not null
 * \param[in] W_is_T whether W contains T
 **/
void update_A(double const *  Y,
              int64_t         lda_Y,
              double *        A,
              int64_t         lda_A,
              int64_t         m,
              int64_t         k,
              int64_t         b,
              double const *  W,
              pview *         pv,
              double *        aggreg_Y,
              int64_t         lda_aY,
              bool            W_is_T){
  int64_t i, j, mb, kb, info;
  double * Ybuf;
  int64_t * signs;

  mb = (m/b)/pv->ccol.np;
  if ((pv->ccol.rank+pv->ccol.np-pv->rrow)%pv->ccol.np < (m/b)%pv->ccol.np)
    mb++;
  mb *= b;
  kb = (k/b)/pv->crow.np;
  if ((pv->crow.rank+pv->crow.np-pv->rcol-1)%pv->crow.np < (k/b)%pv->crow.np)
    kb++;
  kb *= b;
/*  printf("rank = %d, pv->rrow = %d pv->rcol = %d m = %d k = %d mb = %d kb =%d\n",
         pv->cworld->rank,pv->rrow,pv->rcol,m,k,mb,kb);*/
  /*mb = (m+pv->rrow*b)/pv->ccol.np;
  if (pv->ccol.rank < pv->rrow) mb-=b;
  kb = (k+b+pv->rcol*b)/pv->crow.np;
  if (pv->crow.rank <= pv->rcol) kb-=b;
*/
  Ybuf = (double*)malloc(sizeof(double)*mb*b);

  if (pv->crow.rank == pv->rcol){
    if (pv->ccol.rank == pv->rrow){
      copy_lower(Y, Ybuf, b, mb, lda_Y, mb, 1);
      for (i=0; i<b; i++){
        Ybuf[i*mb+i] = 1.0;
      }
    }
    else lda_cpy(mb, b, lda_Y, mb, Y, Ybuf);
  }

  TAU_FSTART(Bcast_update);
  MPI_Bcast(Ybuf, mb*b, MPI_DOUBLE, pv->rcol, pv->crow.cm);
  TAU_FSTOP(Bcast_update);
  upd_A(Ybuf, mb, A, lda_A, mb, kb, b, W, pv, W_is_T);

  if (aggreg_Y != NULL && mb > 0){
    lda_cpy(mb, b, mb, lda_aY, Ybuf, aggreg_Y);
  }

  free(Ybuf);
}
   
void comp_bcast_T_from_W(int64_t        b,
                         double const * W,
                         double const * A,
                         int            lda_A,
                         double **      pT,
                         int            root,
                         bool           is_root,
                         MPI_Comm       cm){
  double * T = (double*)malloc(sizeof(double)*b*b);
  std::fill(T,T+b*b,0.0);
  double * T_buf = (double*)malloc(sizeof(double)*psz_upr(b));
  if (is_root){
    TAU_FSTART(Compute_T);
    /*T contains Y1*/
    lda_cpy( b, b, lda_A, b, A,T);
  
    compute_invT_from_W(W,b,T);

    /* T^-T = stril(sum_i S_i) */
    pack_lower(T,T_buf, b, b, 1);
    TAU_FSTOP(Compute_T);
  } 
  TAU_FSTART(Bcast_TAU);
  MPI_Bcast(T_buf, psz_upr(b), MPI_DOUBLE, root, cm);
  TAU_FSTOP(Bcast_TAU);

  unpack_lower(T_buf,T, b, b, 1);
  free(T_buf);
  *pT = T;
}


/**
 * \brief Perform (I-YTY^T)A  after bcasting Y
 *
 * \param[in,out] Ybuf m-by-k dense lower-triangular matrix of HH vecs owned on
 *                all columns of the 2D grid
 * \param[in,out] A m-by-k dense matrix of HH vecs
 * \param[in] lda_A lda of A
 * \param[in] mb number of rows I own in A
 * \param[in] kb number of columns I own in A
 * \param[in] b number of Householder vectors
 * \param[in] pv current processor grid view
 * \param[in] W_is_T whether W contains T
 **/
void upd_A( double const *  Ybuf,
            int64_t         lda_Y,
            double *        A,
            int64_t         lda_A,
            int64_t         mb,
            int64_t         kb,
            int64_t         b,
            double const *  W,
            pview *         pv,
            bool            W_is_T){
  int64_t i, j, info;
  double * YTbuf, *T, *T_buf;
  double const * cT;
  int64_t * signs;

  if (kb > 0)
    YTbuf = (double*)malloc(sizeof(double)*kb*b);
  
  if (W == NULL){
    T = (double*)malloc(sizeof(double)*b*b);
    std::fill(T,T+b*b,0.0);
    compute_invT_from_Y(Ybuf, lda_Y, T, b, mb, b, pv);
    cT = T;
  } else {
    if (W_is_T) cT=W;
    else { 
      comp_bcast_T_from_W(b, W, Ybuf, lda_Y, &T, pv->rcol+pv->rrow*pv->crow.np, (pv->rrow==pv->ccol.rank) & (pv->rcol==pv->crow.rank), pv->cworld.cm);
      cT = T;
    }
  }


  if (mb > 0 && kb > 0){ //m >= pv->ccol.np*b || pv->crow.rank >= pv->rcol) && (k >= pv->crow.np*b || pv->ccol.rank >= pv->rrow)){
    /* Y^T * A */
    TAU_FSTART(YT_A);
    cdgemm('T','N',b,kb,mb,1.0,Ybuf,lda_Y,A,lda_A,0.0,YTbuf,b);
    TAU_FSTOP(YT_A);
  } else if (kb > 0)
    std::fill(YTbuf, YTbuf+kb*b, 0.0);
  if (kb > 0){
    TAU_FSTART(Allreduce_YTA);
    MPI_Allreduce(MPI_IN_PLACE, YTbuf, kb*b, MPI_DOUBLE, MPI_SUM, pv->ccol.cm);
    TAU_FSTOP(Allreduce_YTA);
  }
  if (mb > 0 && kb > 0){ //m >= pv->ccol.np*b || pv->crow.rank >= pv->rcol) && (k >= pv->crow.np*b || pv->ccol.rank >= pv->rrow)){
    /* T^T * (Y^T * A) */
    TAU_FSTART(Tinv_YTA);
    cdtrsm('L','L','N','N',b,kb,1.0,cT,b,YTbuf,b);
    TAU_FSTOP(Tinv_YTA);
    /* A = A - Y * (T^T * (Y^T * A)) */
    TAU_FSTART(Y_TYTA);
    cdgemm('N','N',mb,kb,b,-1.0,Ybuf,lda_Y,YTbuf,b,1.0,A,lda_A);
    TAU_FSTOP(Y_TYTA);
  }

  if (kb > 0)
    free(YTbuf);
  if (!W_is_T) free(T);
}


/**
 * \brief Perform 2D QR with one level of blocking
 *
 * \param[in,out] A m-by-k dense matrix
 * \param[in] lda_A lda of A
 * \param[in] m number of rows in A
 * \param[in] k number of columns in A
 * \param[in] b number of Householder vectors
 * \param[in] W is either null or Y1^TT
 * \param[in] pv current processor grid view
 * \param[in] aggreg_Y is either null or a buffer into which we should aggregate
 *                      broadcasted panels
 * \param[in] lda_aY lda of aggreg_Y if not null
 **/
void QR_2D( double * A,
            int64_t  lda_A,
            int64_t  m,
            int64_t  k,
            int64_t  b,
            pview *  pv,
            double * aggreg_Y,
            int64_t  lda_aY){
  int64_t i, j, pe_st_new, move_ptr;
  TAU_FSTART(QR_2D);
  double * W = (double*)malloc(b*b*sizeof(double));
  /* TSQR + Householder reconstruction on column */
  TAU_FSTART(Panel_QR);
  if (pv->crow.rank == pv->rcol){
    hh_recon_qr(A,lda_A,m,MIN(k,b),W,pv->ccol.rank,pv->ccol.np,pv->rrow,42,pv->ccol);
  }
  MPI_Barrier(pv->cworld.cm);
  TAU_FSTOP(Panel_QR);
  /* Iterate over panels */
  if (k-b>0 && m-b>0){
    move_ptr = 0;
    if (pv->crow.rank == pv->rcol)
      move_ptr = b*lda_A;
    /* Update 2D distributed matrix */
    TAU_FSTART(trailing_matrix_QR_update);

    update_A(A, lda_A, A+move_ptr, lda_A, m, k-b, b, W, pv,
             aggreg_Y, lda_aY);
    MPI_Barrier(pv->cworld.cm);
    TAU_FSTOP(trailing_matrix_QR_update);
    if (pv->ccol.rank == pv->rrow)
      move_ptr += b;

    if (aggreg_Y != NULL){ 
      aggreg_Y += b*lda_aY;
      if (pv->ccol.rank == pv->rrow) aggreg_Y += b;
    }
    pv->rrow = (pv->rrow+1) % pv->ccol.np;
    pv->rcol = (pv->rcol+1) % pv->crow.np;
    /* Recurse into the next step */
    QR_2D(A+move_ptr, lda_A, 
          m-b, k-b, b, pv, aggreg_Y, lda_aY);

  } else {
    //if (aggreg_Y != NULL && m-b>0){
    if (aggreg_Y != NULL && m-b>=0){
      int64_t mb = (m+pv->rrow*b)/pv->ccol.np;
      if (pv->ccol.rank < pv->rrow) mb-=b;
      double * Ybuf = (double*)malloc(sizeof(double)*mb*b);
      if (pv->crow.rank == pv->rcol){
        if (pv->ccol.rank == pv->rrow){
          copy_lower(A, Ybuf, b, mb, lda_A, mb, 1);
          for (i=0; i<b; i++){
            Ybuf[i*mb+i] = 1.0;
          }
        } else lda_cpy(mb, b, lda_A, mb, A, Ybuf);
      }
      MPI_Bcast(Ybuf, mb*b, MPI_DOUBLE, pv->rcol, pv->crow.cm);
      lda_cpy(mb,b,mb,lda_aY,Ybuf,aggreg_Y);
      free(Ybuf);
    }
  }
  free(W);
  TAU_FSTOP(QR_2D);
} 

/**
 * \brief Perform 2D QR with one level of blocking and with pipelining
 *
 * \param[in,out] A m-by-k dense matrix
 * \param[in] lda_A lda of A
 * \param[in] m number of rows in A
 * \param[in] k number of columns in A
 * \param[in] b number of Householder vectors
 * \param[in] W is either null or Y1^TT
 * \param[in] pv current processor grid view
 * \param[in] last_Y is the Y broadcast at the last step or is null
 * \param[in] lda_lY lda of last_Y if not null
 **/
void QR_2D_pipe(double * A,
                int64_t  lda_A,
                int64_t  m,
                int64_t  k,
                int64_t  b,
                pview *  pv,
                double * last_Y,
                int64_t  lda_lY,
                double * last_W,
                double * my_last_W){
  int64_t i, j, pe_st_new;
  // pipeline looks like the following (ti = ith TSQR, ui ith update, ui0 small lookahead update, ui1 delayed part of lookahead update
  //                  iter
  //        0     1     2     end of loop updates
  //     0  t0    u0    u1    u2   
  // col 1        u00t1 u01   u1   u2
  //     2        u0    u10t2 u11  u2
  // zeroth column therefore needs to be treated specially here, since its not as delayed 
  //    (which is good since then it can start the next TSQR for the following pipeline early)


  // TAU_FSTART(QR_2D);
  TAU_FSTART(1L_panel);   
  double * W = NULL;
  //if not start of pipeline (FIXME: assumes first root col is zero)
  if (pv->rcol != 0){
    //rotate back proc grid
    pview pvb = *pv;
    pvb.rcol = (pvb.rcol-1+pvb.crow.np) % pvb.crow.np;
    pvb.rrow = (pvb.rrow-1+pvb.ccol.np) % pvb.ccol.np;
    int64_t mb, kb;
    mb = ((m+b)/b)/pvb.ccol.np;
    if ((pvb.ccol.rank+pvb.ccol.np-pvb.rrow)%pvb.ccol.np < ((m+b)/b)%pvb.ccol.np)
      mb++;
    mb *= b;
    kb = (k/b)/pvb.crow.np;
    if ((pvb.crow.rank+pvb.crow.np-pvb.rcol-1)%pvb.crow.np < (k/b)%pvb.crow.np)
      kb++;
    kb *= b;

    //receive the Y sent in update_A, but update using previous Y
    double * new_Y = (double*)malloc(mb*b*sizeof(double));
    //if was root column last iteration, broadcast Y and W
    if (pv->crow.rank == pv->rcol-1){

      if (pvb.ccol.rank == pvb.rrow){
        copy_lower(A, new_Y, b, mb, lda_A, mb, 1);
        for (i=0; i<b; i++){
          new_Y[i*mb+i] = 1.0;
        }
      } else lda_cpy(mb, b, lda_A, mb, A, new_Y);
      //keep pointer to latest A to be used up to date
      A = A + b*lda_A;

      MPI_Bcast(new_Y, mb*b, MPI_DOUBLE, pvb.rcol, pvb.crow.cm);
      if (pvb.ccol.rank == pvb.rrow){
        comp_bcast_T_from_W(b, my_last_W, new_Y, mb, &W, pvb.rrow+pvb.rcol*pvb.ccol.np, 1, pvb.cworld.cm);
        free(my_last_W);
      } else
        comp_bcast_T_from_W(b, NULL, NULL, 0, &W, pvb.rrow+pvb.rcol*pvb.ccol.np, 0, pvb.cworld.cm);
    }
    // if first proc column or a proc column that has not been root yet, perform the full update we just broadcast (if it is root perform part of the update)
    if (pv->crow.rank == 0 || pv->crow.rank >= pv->rcol){
      if (pv->crow.rank != pvb.rcol){
        MPI_Bcast(new_Y, mb*b, MPI_DOUBLE, pvb.rcol, pvb.crow.cm);
        comp_bcast_T_from_W(b, NULL, NULL, 0, &W, pvb.rrow+pvb.rcol*pvb.ccol.np, 0, pvb.cworld.cm);
      }
      //if in current root column update only the column about to be factorized
      if (pv->crow.rank == pv->rcol){
        upd_A(new_Y, mb, A, lda_A, mb,  b, b, W, &pvb, true);
        //keep pointer to latest A correct for TSQR, but will need to raise it to finish update later
        if (pv->ccol.rank == pvb.rrow) A+=b;
        //printf("After update:\n");
        //print_dist_mat(m+b, b, b, pvb.ccol.rank, pvb.ccol.np, pvb.rrow, 0, 1, 0, pvb.ccol.cm, A, lda_A);
        //save these Y and W to perform the rest of the update later
        last_Y = new_Y;
        last_W = W;
      } else {
        upd_A(new_Y, mb, A, lda_A, mb, kb, b, W, &pvb, true);
        //keep pointer to latest A to be used up to date
        if (pv->ccol.rank == pvb.rrow) A+=b;
        free(W);
        free(new_Y);
      }
    } else if (pv->crow.rank < pv->rcol){
      if (pv->crow.rank != pvb.rcol){
        MPI_Bcast(new_Y, mb*b, MPI_DOUBLE, pvb.rcol, pvb.crow.cm);
        comp_bcast_T_from_W(b, NULL, NULL, 0, &W, pvb.rrow+pvb.rcol*pvb.ccol.np, 0, pvb.cworld.cm);
      }
      
      pview pvbb = pvb;
      pvbb.rcol = (pvbb.rcol-1+pvbb.crow.np) % pvbb.crow.np;
      pvbb.rrow = (pvbb.rrow-1+pvbb.ccol.np) % pvbb.ccol.np;
      int64_t mbb, kbb;
      mbb = ((m+2*b)/b)/pvbb.ccol.np;
      if ((pvbb.ccol.rank+pvbb.ccol.np-pvbb.rrow)%pvbb.ccol.np < ((m+2*b)/b)%pvbb.ccol.np)
        mbb++;
      mbb *= b;
      kbb = ((k+b)/b)/pvbb.crow.np;
      if ((pvbb.crow.rank+pvbb.crow.np-pvbb.rcol-1)%pvbb.crow.np < ((k+b)/b)%pvbb.crow.np)
        kbb++;
      kbb *= b;

      if (pv->crow.rank == pv->rcol-1){
        int move_ptr = 0;
        if (pvbb.ccol.rank == pvbb.rrow) move_ptr=b;
        upd_A(last_Y, mbb, A-move_ptr, lda_A, mbb, kbb-b, b, last_W, &pvbb, true);
      } else {
        upd_A(last_Y, mbb, A  , lda_A, mbb, kbb  , b, last_W, &pvbb, true);
        //keep pointer to latest A to be used up to date
        if (pv->ccol.rank == pvbb.rrow) A+=b;
      }
    
      free(last_W);
      free(last_Y);

      last_W = W;  
      last_Y = new_Y;
    } else assert(0);
  } 

  // ========= perform 1D panel factorization START ==========
  W = (double*)malloc(sizeof(double)*b*b);
  if (pv->crow.rank == pv->rcol){
    hh_recon_qr(A,lda_A,m,MIN(k,b),W,pv->ccol.rank,pv->ccol.np,pv->rrow,42,pv->ccol);
  }
  TAU_FSTOP(1L_panel);
  
  // ========= perform 1D panel factorization END ==========

  TAU_FSTART(1L_update);      
  //save W if (diagonal) root processor
  if (pv->rrow == pv->ccol.rank && pv->rcol == pv->crow.rank)
    my_last_W = W;
  else {
    if (W!=NULL) free(W);
  }

  if (k-b>0 && m-b>0){
    TAU_FSTOP(1L_update);     

    //if we are at the end of the pipeline (after npcol recursive steps)
    if ((pv->rcol+1)%pv->crow.np == 0){
      /* Perform last pipeline update before recursing */

      int64_t mb, kb;
      mb = (m/b)/pv->ccol.np;
      if ((pv->ccol.rank+pv->ccol.np-pv->rrow)%pv->ccol.np < (m/b)%pv->ccol.np)
        mb++;
      mb *= b;
      kb = ((k-b)/b)/pv->crow.np;
      if ((pv->crow.rank+pv->crow.np-pv->rcol-1)%pv->crow.np < ((k-b)/b)%pv->crow.np)
        kb++;
      kb *= b;

      //receive the Y sent in update_A, but update using previous Y
      double * new_Y = (double*)malloc(mb*b*sizeof(double));

      if (pv->crow.rank == pv->rcol){
        if (pv->ccol.rank == pv->rrow){
          copy_lower(A, new_Y, b, mb, lda_A, mb, 1);
          for (i=0; i<b; i++){
            new_Y[i*mb+i] = 1.0;
          }
        } else lda_cpy(mb, b, lda_A, mb, A, new_Y);
        //keep pointer to latest A to be used up to date
        A += b*lda_A;
      }

      MPI_Bcast(new_Y, mb*b, MPI_DOUBLE, pv->rcol, pv->crow.cm);

      if (pv->crow.rank == pv->rcol && pv->ccol.rank == pv->rrow)
        comp_bcast_T_from_W(b, my_last_W, new_Y, mb, &W, pv->rrow+pv->rcol*pv->ccol.np, 1, pv->cworld.cm);
      else
        comp_bcast_T_from_W(b, NULL, NULL, 0, &W, pv->rrow+pv->rcol*pv->ccol.np, 0, pv->cworld.cm);
      //All except the first processor column need to do one more update before the last one
      if (pv->crow.rank > 0){
        pview pvb = *pv;
        //pvb.rcol--;
        pvb.rcol = (pvb.rcol-1+pvb.crow.np) % pvb.crow.np;
        pvb.rrow = (pvb.rrow-1+pvb.ccol.np) % pvb.ccol.np;
        int64_t mbb, kbb;
        mbb = ((m+b)/b)/pvb.ccol.np;
        if ((pvb.ccol.rank+pvb.ccol.np-pvb.rrow)%pvb.ccol.np < ((m+b)/b)%pvb.ccol.np)
          mbb++;
        mbb *= b;
        kbb = (k/b)/pvb.crow.np;
        if ((pvb.crow.rank+pvb.crow.np-pvb.rcol-1)%pvb.crow.np < (k/b)%pvb.crow.np)
          kbb++;
        kbb *= b;

        if (pv->crow.rank == pv->rcol){
          int move_ptr = 0;
          if (pvb.ccol.rank == pvb.rrow) move_ptr=b;
          upd_A(last_Y, mbb, A-move_ptr, lda_A, mbb, kbb-b, b, last_W, &pvb, true);
        } else {
          upd_A(last_Y, mbb, A  , lda_A, mbb, kbb  , b, last_W, &pvb, true);
          //keep pointer to latest A to be used up to date
          if (pv->ccol.rank == pvb.rrow) A+=b;
        }

        free(last_W);  
        free(last_Y);
      }

      //all processor columns need to do this last update
      upd_A(new_Y, mb, A, lda_A, mb, kb, b, W, pv, true);
      //keep pointer to latest A to be used up to date
      if (pv->ccol.rank == pv->rrow) A+=b;

      pv->rrow = (pv->rrow+1) % pv->ccol.np;
      pv->rcol = (pv->rcol+1) % pv->crow.np;
      QR_2D_pipe(A, lda_A, 
                 m-b, k-b, b, pv, NULL, 0, NULL, NULL);
    } else {
      pv->rrow = (pv->rrow+1) % pv->ccol.np;
      pv->rcol = (pv->rcol+1) % pv->crow.np;
      /* Recurse into the next step continuing pipline */
      QR_2D_pipe(A, lda_A, 
            m-b, k-b, b, pv, last_Y, lda_lY, last_W, my_last_W);
    }
  } else {
    //All except the first processor column need to do one more update
    if (pv->crow.rank > 0){
      pview pvb = *pv;
      pvb.rcol = (pvb.rcol-1+pvb.crow.np) % pvb.crow.np;
      pvb.rrow = (pvb.rrow-1+pvb.ccol.np) % pvb.ccol.np;
      int64_t mbb, kbb;
      mbb = ((m+b)/b)/pvb.ccol.np;
      if ((pvb.ccol.rank+pvb.ccol.np-pvb.rrow)%pvb.ccol.np < ((m+b)/b)%pvb.ccol.np)
        mbb++;
      mbb *= b;
      kbb = (k/b)/pvb.crow.np;
      if ((pvb.crow.rank+pvb.crow.np-pvb.rcol-1)%pvb.crow.np < (k/b)%pvb.crow.np)
        kbb++;
      kbb *= b;

      if (pv->crow.rank == pv->rcol){
        int move_ptr = 0;
        if (pvb.ccol.rank == pvb.rrow) move_ptr=b;
        upd_A(last_Y, mbb, A-move_ptr, lda_A, mbb, kbb-b, b, last_W, &pvb, true);
      } else {
        upd_A(last_Y, mbb, A  , lda_A, mbb, kbb  , b, last_W, &pvb, true);
        //keep pointer to latest A to be used up to date
        if (pv->ccol.rank == pvb.rrow) A+=b;
      }

      //keep pointer to latest A to be used up to date
      if (pv->ccol.rank == pvb.rrow) A+=b;

      free(last_W);  
      free(last_Y);
    }

  }
  // free(W);
  // TAU_FSTOP(QR_2D);
}  


/**
 * \brief Perform 2D QR with ScaLAPACK panel factorization
 *
 * \param[in,out] A m-by-k dense matrix (pointer to working matrix block)
 * \param[in] lda_A lda of A
 * \param[in] m number of rows in A
 * \param[in] k number of columns in A
 * \param[in] b number of Householder vectors
 * \param[in] W is either null or Y1^TT
 * \param[in] pv current processor grid view
 * \param[in] aggreg_Y is either null or a buffer into which we should aggregate
 *                      broadcasted panels
 * \param[in] lda_aY lda of aggreg_Y if not null
 * \param[in] desc_A descriptor for whole A matrix
 * \param[in] org_A pointer to top left corner of A matrix
 * \param[in] IA row index offset
 * \param[in] JA column index offset
 **/
void QR_scala_2D( double * A,
                  int64_t  lda_A,
                  int64_t  m,
                  int64_t  k,
                  int64_t  b,
                  pview *  pv,
                  double * aggreg_Y,
                  int64_t  lda_aY,
                  int const * desc_A,
                  double * org_A,
                  int64_t IA,
                  int64_t JA){
#ifndef USE_SCALAPACK
  assert(0);
}
#else
  TAU_FSTART(QR_scala_2D);
  int64_t i, j, pe_st_new, move_ptr;
//  double * W = (double*)malloc(b*b*sizeof(double));
  double * W = NULL;
  /* TSQR + Householder reconstruction on column */
  double * T = (double*)malloc((desc_A[2]+desc_A[3]+b*b)*sizeof(double));
  TAU_FSTART(Panel_QR);
//  if (pv->crow.rank == pv->rcol){
  int64_t lwork = m*k+2*b*b;
  double * work = (double*)malloc(lwork*sizeof(double));
  int info;
  TAU_FSTART(pdgeqrf);
  /*printf("org_A = %p calling pdgerqf with IA=%d IJ = %d, decs_A[2] = %d m = %d desc_A[3] = %d k = %d\n", 
         org_A,1+desc_A[2]-m,1+desc_A[3]-k, desc_A[2], m, desc_A[3], k);*/
  cpdgeqrf(m,MIN(k,b),org_A,IA,JA,desc_A,T,work,lwork,&info);
  if (m==MIN(k,b) && pv->rrow == pv->ccol.rank && pv->rcol == pv->crow.rank){
    A[(MIN(k,b)-1)*lda_A+MIN(k,b)-1] *= -1.0;
  }
//  cpdgeqrf(m,MIN(k,b),org_A,1,1,desc_A,T,work,lwork,&info);
  TAU_FSTOP(pdgeqrf);
    //cdgeqrf(m,MIN(k,b),A,lda_A,T,work,lwork,&info);
/*  if (pv->crow.rank == pv->rcol && pv->ccol.rank == pv->rrow){
    std::fill(W,W+b*b,0.0);
    for (i=0; i<b; i++){
      W[j*b+j] = 1.0;
      for (j=i+1; j<b; j++){
        W[j*b+i] = A[i*lda_A+j];
      }
    }
  }
  if (pv->rrow == pv->ccol.rank)
    MPI_Bcast(W, b*b, MPI_DOUBLE, pv->rcol, pv->crow.cm);
  MPI_Bcast(W, b*b, MPI_DOUBLE, pv->rrow, pv->ccol.cm);*/
  free(work);
    //hh_recon_qr(A,lda_A,m,MIN(k,b),W,pv->ccol.rank,pv->ccol.np,pv->rrow,42,pv->ccol);
//  }
  MPI_Barrier(pv->cworld.cm);
  TAU_FSTOP(Panel_QR);
  /* Iterate over panels */
  if (k-b>0 && m-b>0){
    move_ptr = 0;
    if (pv->crow.rank == pv->rcol)
      move_ptr = b*lda_A;
    /* Update 2D distributed matrix */
    TAU_FSTART(trailing_matrix_QR_update);

    update_A(A, lda_A, A+move_ptr, lda_A, m, k-b, b, W, pv,
             aggreg_Y, lda_aY);
    MPI_Barrier(pv->cworld.cm);
    TAU_FSTOP(trailing_matrix_QR_update);
    if (pv->ccol.rank == pv->rrow)
      move_ptr += b;

    if (aggreg_Y != NULL){ 
      aggreg_Y += b*lda_aY;
      if (pv->ccol.rank == pv->rrow) aggreg_Y += b;
    }
    pv->rrow = (pv->rrow+1) % pv->ccol.np;
    pv->rcol = (pv->rcol+1) % pv->crow.np;
    /* Recurse into the next step */
    QR_scala_2D(A+move_ptr, lda_A, 
          m-b, k-b, b, pv, aggreg_Y, lda_aY, desc_A, org_A, IA+b, JA+b);

  } else {
    if (aggreg_Y != NULL && m-b>=0){
      int64_t mb = (m+pv->rrow*b)/pv->ccol.np;
      if (pv->ccol.rank < pv->rrow) mb-=b;
      double * Ybuf = (double*)malloc(sizeof(double)*mb*b);
      if (pv->crow.rank == pv->rcol){
        if (pv->ccol.rank == pv->rrow){
          copy_lower(A, Ybuf, b, mb, lda_A, mb, 1);
          for (i=0; i<b; i++){
            Ybuf[i*mb+i] = 1.0;
          }
        } else lda_cpy(mb, b, lda_A, mb, A, Ybuf);
      }
      MPI_Bcast(Ybuf, mb*b, MPI_DOUBLE, pv->rcol, pv->crow.cm);
      lda_cpy(mb,b,mb,lda_aY,Ybuf,aggreg_Y);
      
    }
  }
  TAU_FSTOP(QR_scala_2D);
}  
#endif

/**
 * \brief Perform 2D QR with two levels of blocking
 *
 * \param[in,out] A m-by-k dense matrix
 * \param[in] lda_A lda of A
 * \param[in] m number of rows in A
 * \param[in] k number of columns in A
 * \param[in] b large block-size to which to aggregate Y, is a multiple of b_sub
 * \param[in] b_sub small block size with which matrix is distributed
 * \param[in] W is either null or Y1^TT
 * \param[in] pv current processor grid view
 * \param[in] aggreg_Y is either null or a buffer into which we should aggregate
 *                      broadcasted panels
 * \param[in] lda_aY lda of aggreg_Y if not null
 **/
void QR_2D_2D(double  * A,
              int64_t   lda_A,
              int64_t   m,
              int64_t   k,
              int64_t   b,
              int64_t   b_sub,
              pview   * pv0,
              double  * aggreg_Y,
              int64_t   lda_aY,
              int *     desc_A,
              double *  org_A,
              int64_t   IA,
              int64_t   JA){
  int64_t i, j, pe_st_new, move_ptr, mb, kb, bmb, bkb;
  double * iaggreg_Y;
  int64_t lda_iaY;

  pview pvo = *pv0;
  pview * pv = &pvo;

  /* Iterate over panels */
  if (k-b>0 && m-b>0){
  /*  mb = (m+pv->rrow*b_sub)/pv->ccol.np;
    if (pv->ccol.rank < pv->rrow) mb-=b_sub;
    kb = (k+pv->rcol*b_sub)/pv->crow.np;
    if (pv->crow.rank < pv->rcol) kb-=b_sub;*/

    mb = (m/b_sub)/pv->ccol.np;
    if ((pv->ccol.rank+pv->ccol.np-pv->rrow)%pv->ccol.np < (m/b_sub)%pv->ccol.np)
      mb++;
    mb *= b_sub;
    kb = (k/b_sub)/pv->crow.np;
    if ((pv->crow.rank+pv->crow.np-pv->rcol)%pv->crow.np < (k/b_sub)%pv->crow.np)
      kb++;
    kb *= b_sub;
    
    // FIXME: code is incorrect for 144 processes on 12 by 12 grid, 384 by 384
    // matrix with block sizes 8 and 32. Becomes correct if second block size is
    // multiple of 3
    int fm = m-b;
    int fk = k-b;
    int frrow = (pv->rrow+b/b_sub) % pv->ccol.np;
    int frcol = (pv->rcol+b/b_sub) % pv->crow.np;
    bmb = (fm/b_sub)/pv->ccol.np;
    if ((pv->ccol.rank+pv->ccol.np-frrow)%pv->ccol.np < (fm/b_sub)%pv->ccol.np)
      bmb++;
    bmb *= b_sub;
    bkb = (fk/b_sub)/pv->crow.np;
    if ((pv->crow.rank+pv->crow.np-frcol)%pv->crow.np < (fk/b_sub)%pv->crow.np)
      bkb++;
    bkb *= b_sub;

    int str = ((pv->rrow+b/b_sub)%pv->ccol.np);
    int stc = ((pv->rcol+b/b_sub)%pv->crow.np);

    int bmb2 = (m-b+str*b_sub)/pv->ccol.np;
    if (pv->ccol.rank < str) bmb2-=b_sub;
    int bkb2 = (k-b+stc*b_sub)/pv->crow.np;
    if (pv->crow.rank < stc) bkb2-=b_sub;
    assert(bmb==bmb2);
    assert(bkb==bkb2);

    if (aggreg_Y == NULL){
      iaggreg_Y = (double*)malloc(mb*b*sizeof(double));
      std::fill(iaggreg_Y,iaggreg_Y+mb*b,0.0);
      lda_iaY = mb;
    } else {
      iaggreg_Y = aggreg_Y;
      lda_iaY = lda_aY;
    }
    /* TSQR + Householder reconstruction on column */

    TAU_FSTART(2D_Panel_QR);
    if (desc_A == NULL)
      QR_2D(A, lda_A, m, MIN(k, b), b_sub, pv, iaggreg_Y, lda_iaY);
    else
      QR_scala_2D(A, lda_A, m, MIN(k, b), b_sub, pv, iaggreg_Y, lda_iaY, desc_A, org_A, IA, JA);
    MPI_Barrier(pv->cworld.cm);
    TAU_FSTOP(2D_Panel_QR);
    /*printf("iaggreg_Y:\n");
    double * daY = iaggreg_Y;
    for (int cc=0; cc<pv->crow.np; cc++){
      if (pv->crow.rank == cc){
        for (int rr=0; rr<m/b_sub; rr++){
          if (pv->ccol.rank == (rr%numPes)){
            print_matrix(daY,b_sub,b,mb);
            daY+=b_sub;
          }
          MPI_Barrier(pv->ccol.cm);
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }*/

    move_ptr = (kb-bkb)*lda_A;

    //printf("rank = %d kb-bkb = %d mb - bmb = %d\n", pv->cworld->rank, kb-bkb, mb-bmb);

    /* Update 2D distributed matrix */
    TAU_FSTART(trailing_matrix_QR_update_upper);

    upd_A(iaggreg_Y, lda_iaY, A+move_ptr, lda_A, mb, bkb, b, NULL, pv);
    MPI_Barrier(pv->cworld.cm);
    TAU_FSTOP(trailing_matrix_QR_update_upper);
    move_ptr+=mb-bmb;


    //pv root row and col are moved inside QR_2D
    /*pv->rrow = (pv->rrow+b/b_sub) % pv->ccol.np;
    pv->rcol = (pv->rcol+b/b_sub) % pv->crow.np;*/
    pv->rrow = (pv->rrow+1) % pv->ccol.np;
    pv->rcol = (pv->rcol+1) % pv->crow.np;
    /* Recurse into the next step */
    if (aggreg_Y != NULL) aggreg_Y+=(lda_aY*b + mb-bmb);
    else free(iaggreg_Y);
    QR_2D_2D(A+move_ptr, lda_A, m-b, k-b, b, b_sub, pv, aggreg_Y, lda_aY, desc_A, org_A, IA+b, JA+b);
  } else {
    TAU_FSTART(2D_Panel_QR);
    if (desc_A == NULL)
      QR_2D(A,lda_A,m,k,b_sub,pv,aggreg_Y,lda_aY);
    else
      QR_scala_2D(A,lda_A,m,k,b_sub,pv,aggreg_Y,lda_aY,desc_A,org_A,IA,JA);
    MPI_Barrier(pv->cworld.cm);
    TAU_FSTOP(2D_Panel_QR);
  }
}  


/*
void QR_25D(double *        A,
            int64_t         lda_A,
            int64_t         m,
            int64_t         k,
            int64_t         b_25d,
            int64_t         b_2d_2d,
            int64_t         b_2d,
            pview *         pv){
  int64_t mb, kb;

  if (m-b_25d>0 && k-b_25d>0){
    mb = (m/b_2d)/pv->ccol.np;
    if ((pv->ccol.rank+pv->ccol.np-pv->rrow)%pv->ccol.np < (m/b_2d)%pv->ccol.np)
      mb++;
    mb *= b_2d;
    kb = (k/b_sub)/pv->crow.np;
    if ((pv->crow.rank+pv->crow.np-pv->rcol)%pv->crow.np < (k/b_sub)%pv->crow.np)
      kb++;
    kb *= b_sub;
    
    bmb = (m-b_25d+((pv->rrow+b_25d/b_sub)%pv->ccol.np)*b_sub)/pv->ccol.np;
    if (pv->ccol.rank < ((pv->rrow+b_25d/b_sub)%pv->ccol.np)) bmb-=b_sub;
    bkb = (k-b_25d+((pv->rcol+b_25d/b_sub)%pv->crow.np)*b_sub)/pv->crow.np;
    if (pv->crow.rank < ((pv->rcol+b_25d/b_sub)%pv->crow.np)) bkb-=b_sub;

    double * aggreg_Y = (double*)malloc(sizeof(double)*mb*b_25d);
    std::fill(aggreg_Y,aggreg_Y+mb*b_25d,0.0);

    QR_2D_2D(A,lda_A,m,b_25d,b_2d_2d,b_2d,pv,aggreg_Y,mb);

    move_ptr = (kb-bkb)*lda_A;
    
    TAU_FSTART(25d_trailing_matrix_QR_update);
    upd_A(aggreg_Y, mb, A+move_ptr, lda_A, mb, bkb, b_25d, NULL, pv);
    MPI_Barrier(pv->cworld.cm);
    TAU_FSTOP(25d_trailing_matrix_QR_update);
    move_ptr+=mb-bmb;

    QR

    TAU_FSTART(25d_2D_Panel_QR);
  } else {
    mb = (m/b_2d)/pv->ccol.np;
    if ((pv->ccol.rank+pv->ccol.np-pv->rrow)%pv->ccol.np < (m/b_2d)%pv->ccol.np)
      mb++;
    mb *= b_2d;
    double * aggreg_Y = (double*)malloc(sizeof(double)*mb*b_2d_2d);
    std::fill(aggreg_Y,aggreg_Y+mb*b_2d_2d,0.0);
    QR_2D_2D(A,lda_A,m,k,b_2d_2d,b_2d,pv,aggreg_Y,mb);
  }
}*/
