/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "../../shared/util.h"
#include "qr_y2d.h"
#include "../hh_recon/yamamoto.h"

 
aggregator::aggregator(int64_t lda_aQm_, int64_t lda_aT_){
  lda_aQm = lda_aQm_;
  lda_aT  = lda_aT_;

  aQm = (double*)malloc(sizeof(double)*lda_aT*lda_aQm);
  std::fill(aQm, aQm+lda_aT*lda_aQm, 0.0); 
  aT  = (double*)malloc(sizeof(double)*lda_aT*lda_aT);
  std::fill(aT, aT+lda_aT*lda_aT, 0.0);
  n = 0;
  shift = 0;
}

void aggregator::reset(){
  n=0;
  shift = 0;
  //FIXME: should not be necessary
  std::fill(aQm, aQm+lda_aT*lda_aQm, 0.0); 
  std::fill(aT, aT+lda_aT*lda_aT, 0.0);
}


void aggregator::shift_down(int64_t b){
  shift += b;
}
 
void aggregator::append(int64_t        mb,
                        int64_t        b,
                        double const * Qm,
                        int64_t        lda_Qm,
                        double const * T,
                        pview *        pv){
  if (mb > 0){
    lda_cpy(mb, b, mb, lda_aQm, Qm, aQm+n*lda_aQm+shift);
  }
  if (n==0) lda_cpy(b,b,b,lda_aT,T,aT);
  else {
    //compute T_{12} = -T_{11}(Y^TY)T_{22}
    double tmp_T[n*b];
    double tmp_T2[n*b];
    /*cdgemm('T','N',n,b,mb,1.0,aQm+shift,lda_aQm,Qm,lda_Qm,0.0,tmp_T2,n);
    MPI_Allreduce(tmp_T2, tmp_T, n*b, MPI_DOUBLE, MPI_SUM, pv->ccol.cm);
    cdgemm('N','N',n,b,n,-1.0,aT,lda_aT,tmp_T2,n,0.0,tmp_T,n);
    cdgemm('N','N',n,b,b, 1.0,tmp_T,n,T,b,0.0,aT+n*lda_aT,lda_aT);*/
    if (mb>0)
      cdgemm('T','N',b,n,mb,1.0,Qm,lda_Qm,aQm+shift,lda_aQm,0.0,tmp_T2,b);
    else
      std::fill(tmp_T2, tmp_T2+b*b, 0.0);
    MPI_Allreduce(tmp_T2, tmp_T, n*b, MPI_DOUBLE, MPI_SUM, pv->ccol.cm);
    cdgemm('N','N',b,n,n, 1.0,tmp_T,b,aT,lda_aT,0.0,tmp_T2,b);
    cdgemm('N','N',b,n,b, 1.0,T,b,tmp_T2,b,0.0,aT+n,lda_aT);
    lda_cpy(b,b,b,lda_aT,T,aT+n*lda_aT+n);
  }
  n+=b;
}

void update_Yamamoto_A(double *     Qm,
                       int64_t      lda_Qm,
                       double *     A,
                       int64_t      lda_A,
                       int64_t      m,
                       int64_t      k,
                       int64_t      b,
                       double *     T,
                       pview *      pv,
                       aggregator * agg){
  int64_t mb, kb;
  double * Qm_buf;

  mb = (m/b)/pv->ccol.np;
  if ((pv->ccol.rank+pv->ccol.np-pv->rrow)%pv->ccol.np < (m/b)%pv->ccol.np)
    mb++;
  mb *= b;
  kb = (k/b)/pv->crow.np;
  if ((pv->crow.rank+pv->crow.np-pv->rcol-1)%pv->crow.np < (k/b)%pv->crow.np)
    kb++;
  kb *= b;

/*
  if (pv->crow.rank == pv->rcol){
    if (pv->ccol.rank == pv->rrow){
      copy_lower(Qm, Qm_buf, b, mb, lda_Qm, mb, 1);
      for (i=0; i<b; i++){
        Qm_buf[i*mb+i] = 1.0;
      }
    }
    else 
  }*/
  if (mb != lda_Qm){
    Qm_buf = (double*)malloc(sizeof(double)*mb*b);
    lda_cpy(mb, b, lda_Qm, mb, Qm, Qm_buf);
  } else {
    Qm_buf = Qm;
  }

  TAU_FSTART(Bcast_update);
  MPI_Bcast(Qm_buf, mb*b, MPI_DOUBLE, pv->rcol, pv->crow.cm);
  TAU_FSTOP(Bcast_update);
  TAU_FSTART(Bcast_T);
  MPI_Bcast(T, b*b, MPI_DOUBLE, pv->rcol, pv->crow.cm);
  TAU_FSTOP(Bcast_T);
  upd_Yamamoto_A(Qm_buf, mb, A, lda_A, mb, kb, b, T, pv);

  if (agg != NULL)
    agg->append(mb, b, Qm_buf, lda_Qm, T, pv);

  if (mb != lda_Qm)
    free(Qm_buf);
}
  

void upd_Yamamoto_A(double const * Qm,
                    int64_t        lda_Qm,
                    double *       A,
                    int64_t        lda_A,
                    int64_t        mb,
                    int64_t        kb,
                    int64_t        b,
                    double const * T,
                    pview *        pv){
  double * QmTA, * QmTA2;

  if (kb > 0){
    QmTA = (double*)malloc(sizeof(double)*kb*b);
    QmTA2 = (double*)malloc(sizeof(double)*kb*b);
  }

  if (mb > 0 && kb > 0){ //m >= pv->ccol.np*b || pv->crow.rank >= pv->rcol) && (k >= pv->crow.np*b || pv->ccol.rank >= pv->rrow)){
    /* Qm^T * A */
    TAU_FSTART(QmT_A);
    cdgemm('T','N',b,kb,mb,1.0,Qm,lda_Qm,A,lda_A,0.0,QmTA,b);
    TAU_FSTOP(QmT_A);
  } else if (kb > 0)
    std::fill(QmTA, QmTA+kb*b, 0.0);
  if (kb > 0){
    TAU_FSTART(Allreduce_QmTA);
    MPI_Allreduce(MPI_IN_PLACE, QmTA, kb*b, MPI_DOUBLE, MPI_SUM, pv->ccol.cm);
    TAU_FSTOP(Allreduce_QmTA);
  }
  if (mb > 0 && kb > 0){ 
    /* (S-Q)^-T * (Qm^T * A) */
    TAU_FSTART(QmSinv_QmTA);
    //sign since we compute LU of Q-S and not S-Q
    //cdtrsm('L','L','T','U',b,kb,1.0,T,b,QmTA,b);
    //cdtrsm('L','U','T','N',b,kb,1.0,T,b,QmTA,b);
    cdgemm('N','N',b,kb,b,-1.0,T,b,QmTA,b,0.0,QmTA2,b);
    TAU_FSTOP(QmSinv_QmTA);
    /* A = A - Qm * ((Q-S)^-1 * (Qm^T * A)) */
    TAU_FSTART(Y_TQmTA);
    cdgemm('N','N',mb,kb,b,-1.0,Qm,lda_Qm,QmTA2,b,1.0,A,lda_A);
    TAU_FSTOP(Y_TQmTA);
  }

  if (kb > 0){
    free(QmTA);
    free(QmTA2);
  }
}


void QR_Yamamoto_2D(double *     A,
                    int64_t      lda_A,
                    int64_t      m,
                    int64_t      k,
                    int64_t      b,
                    pview *      pv,
                    aggregator * agg,
                    bool         compute_Y){
  int64_t move_ptr;
  TAU_FSTART(QR_2D);
  double * T = (double*)malloc(b*b*sizeof(double));
  /* TSQR + Householder reconstruction on column */
  TAU_FSTART(Panel_QR);
  int64_t mb = (m/b)/pv->ccol.np;
  if ((pv->ccol.rank+pv->ccol.np-pv->rrow)%pv->ccol.np < (m/b)%pv->ccol.np)
    mb++;
  mb *= b;

  double * Qm;
  int64_t lda_Qm;
  /*if (aggreg_Qm != NULL){
    Qm = aggreg_Qm;
    lda_Qm = lda_aQm;
  } else {
  }*/
  Qm = (double*)malloc(mb*b*sizeof(double));
  lda_Qm = mb;
  if (pv->crow.rank == pv->rcol){
    Yamamoto(A,lda_A,Qm,lda_Qm,m,MIN(k,b),T,pv->ccol.rank,pv->ccol.np,pv->rrow,42,pv->ccol);
  }
#ifdef TAU
  MPI_Barrier(pv->cworld.cm);
#endif
  TAU_FSTOP(Panel_QR);
  move_ptr = 0;
  /* Iterate over panels */
  if (k-b>0 && m-b>0){
    if (pv->crow.rank == pv->rcol)
      move_ptr = b*lda_A;
    /* Update 2D distributed matrix */
    TAU_FSTART(trailing_matrix_QR_update);

    update_Yamamoto_A(Qm, lda_Qm, A+move_ptr, lda_A, m, k-b, b, T, pv, agg);
#ifdef TAU
    MPI_Barrier(pv->cworld.cm);
#endif
    TAU_FSTOP(trailing_matrix_QR_update);
  }
  if (compute_Y){
    double * LU = (double*)malloc(sizeof(double)*b*b);
    if (pv->crow.rank == pv->rcol){
      if (pv->ccol.rank == pv->rrow){
        int * ipiv = (int*)malloc(sizeof(int)*b);
        int info;
        lda_cpy(b,b,lda_Qm,b,Qm,LU);
        cdgetrf(b,b,LU,b,ipiv,&info);
      }
      MPI_Bcast(LU, b*b, MPI_DOUBLE, pv->rrow, pv->ccol.cm);
      double Qm2[mb*b];
      memcpy(Qm2,Qm,mb*b*sizeof(double));
      if (mb > 0){
        cdtrsm('R', 'U', 'N', 'N', mb, b, 1.0, LU, b, Qm2, lda_Qm);
        if (pv->ccol.rank == pv->rrow){
          copy_lower(Qm2,A,b,mb,lda_Qm,lda_A,0);
        } else {
          lda_cpy(mb,b,lda_Qm,lda_A,Qm2,A);
        }
      }
    }
    free(LU);
  }
  
  if (k-b>0 && m-b>0){
    free(Qm);
    free(T);
    if (pv->ccol.rank == pv->rrow)
      move_ptr += b;

    /*if (aggreg_Qm != NULL){ 
      assert(0); 
      aggreg_Qm += b*lda_aQm;
      if (pv->ccol.rank == pv->rrow) aggreg_Qm += b;
    }*/
    if (agg != NULL && pv->ccol.rank == pv->rrow) agg->shift_down(b);
    pv->rrow = (pv->rrow+1) % pv->ccol.np;
    pv->rcol = (pv->rcol+1) % pv->crow.np;
    /* Recurse into the next step */
    QR_Yamamoto_2D(A+move_ptr, lda_A, m-b, k-b, b, pv, agg, compute_Y);

  } else {
    if (agg != NULL && m-b>=0){
      //int64_t mb = (m+pv->rrow*b)/pv->ccol.np;
      //if (pv->ccol.rank < pv->rrow) mb-=b;
      MPI_Bcast(Qm, mb*b, MPI_DOUBLE, pv->rcol, pv->crow.cm);
      MPI_Bcast(T, b*b, MPI_DOUBLE, pv->rcol, pv->crow.cm);
      agg->append(mb, b, Qm, mb, T, pv);
    }
    free(Qm);
    free(T);
  }
  TAU_FSTOP(QR_2D);
}

void QR_Yamamoto_2D_2D(double  *    A,
                       int64_t      lda_A,
                       int64_t      m,
                       int64_t      k,
                       int64_t      b,
                       int64_t      b_sub,
                       pview   *    pv0,
                       aggregator * agg,
                       bool         compute_Y){
  int64_t move_ptr, mb, kb, bmb, bkb;
  aggregator * iagg;

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

    if (agg == NULL){
      iagg = new aggregator(mb,b);
    } else {
      iagg = agg;
    }
    /* TSQR + Householder reconstruction on column */

    TAU_FSTART(2D_Panel_QR);
    QR_Yamamoto_2D(A, lda_A, m, MIN(k, b), b_sub, pv, iagg, compute_Y);
#ifdef TAU
    MPI_Barrier(pv->cworld.cm);
#endif
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

    upd_Yamamoto_A(iagg->aQm, iagg->lda_aQm, A+move_ptr, lda_A, mb, bkb, b, iagg->aT, pv);
#ifdef TAU
    MPI_Barrier(pv->cworld.cm);
#endif
    TAU_FSTOP(trailing_matrix_QR_update_upper);
    move_ptr+=mb-bmb;


    //pv root row and col are moved inside QR_2D
    /*pv->rrow = (pv->rrow+b/b_sub) % pv->ccol.np;
    pv->rcol = (pv->rcol+b/b_sub) % pv->crow.np;*/
    pv->rrow = (pv->rrow+1) % pv->ccol.np;
    pv->rcol = (pv->rcol+1) % pv->crow.np;
    /* Recurse into the next step */
    iagg->reset();
    QR_Yamamoto_2D_2D(A+move_ptr, lda_A, m-b, k-b, b, b_sub, pv, iagg, compute_Y);
    if (agg == NULL)
      delete iagg;
  } else {
    TAU_FSTART(2D_Panel_QR);
    QR_Yamamoto_2D(A,lda_A,m,k,b_sub,pv,agg,compute_Y);
#ifdef TAU
    MPI_Barrier(pv->cworld.cm);
#endif
    TAU_FSTOP(2D_Panel_QR);
  }
}  
 
