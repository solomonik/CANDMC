/* Author: Edgar Solomonik, April 9, 2014 */

/* File contains routines for reduction from fully-dense to banded */

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define PRINTALL
#include "../QR/qr_2d/qr_2d.h"
#include "dmatrix.h"
#include "../shared/util.h"
/**
 * \brief Perform reduction to banded using a 3D algorithm
 *
 * \param[in,out] A n-by-n dense symmetric matrix, stored unpacked
 *                  pointer should refer to current working corner of A
 *                 blocked accrows crow and ccol pv and replicated over each layer of c_lyr
 * \param[in] lda_A lda of A
 * \param[in] n number of rows and columns in A
 * \param[in] b_agg is the number of U vectors to aggregate before applying to 
 *                  full trailing matrix (must be multiple of b_sub)
 * \param[in] bw is the bandwidth to reduce to (2D QR size)
 * \param[in] b_sub small block size with which matrix is distributed
 * \param[in] pv current processor grid view oriented at corner of A 
 **/
void sym_full2band_3d(double *    A,
                      int64_t     lda_A,
                      int64_t     n,
                      int64_t     b_agg,
                      int64_t     bw,
                      int64_t     b_sub,
                      pview_3d *  pv){
  assert((b_agg/pv->plyr.crow.np)%b_sub == 0);
  assert((b_agg/pv->plyr.ccol.np)%b_sub == 0);
  assert(b_agg%bw==0);
  assert(bw%b_sub==0);
  //assert((bw/b_sub)%(pv->plyr.ccol.np*pv->clyr.np)==0);
  assert((bw/b_sub)%pv->clyr.np==0);

  int lyrcols = bw/pv->clyr.np;

  //initialize whole buffer of U and V to zero
  DMatrix dU_lyr= DMatrix(n, b_agg, b_sub, pv->plyr, "U");
  dU_lyr.set_to_zero();

  DMatrix dVT_lyr= DMatrix(n, b_agg, b_sub, pv->plyr, "V^T");
  dVT_lyr.set_to_zero();

  //set their rows/cols to zero, will grow as we go along
  dU_lyr.ncol = 0;
  dVT_lyr.ncol = 0;

  DMatrix dA = DMatrix(n, n, b_sub, pv->plyr, lda_A, NULL, A);
  dA.set_name("A");
  
  

  for (int istep=0; istep<std::min(n/bw,b_agg/bw); istep++){
    //A has to be replicated to do W=A*Y
    DMatrix dA_slice = dA.slice(0, dA.nrow, pv->clyr.rank*lyrcols, lyrcols);

    if (istep>0){
      DMatrix dU_slice  =  dU_lyr.slice(lyrcols*pv->clyr.rank, lyrcols, 0, dU_lyr.ncol);
      DMatrix dVT_slice = dVT_lyr.slice(lyrcols*pv->clyr.rank, lyrcols, 0, dVT_lyr.ncol);
    
      cpdgemm(dU_lyr,  dVT_slice.transp(), -1.0, dA_slice, 1.0);
      cpdgemm(dVT_lyr, dU_slice.transp(),  -1.0, dA_slice, 1.0);

      DMatrix dA_cblk = dA_slice.slice(0,bw,0,lyrcols).get_contig();
      DMatrix dA_allcblk = DMatrix(bw, bw, b_sub, pv->plyr);
    
      MPI_Allgather(dA_cblk,    dA_cblk.get_mysize(), MPI_DOUBLE,
                    dA_allcblk, dA_cblk.get_mysize(), MPI_DOUBLE,
                    pv->clyr.cm);

      DMatrix dA_ccblk = dA.slice(0,bw,0,bw);
      dA_ccblk.set_to_zero();
      dA_ccblk.daxpy(dA_allcblk, 1.0);

      dA_cblk.destroy();
      dA_allcblk.destroy();

      if (bw*istep == n-bw) break;
    }
   
    dA_slice.move_ptr(bw, 0);
    
    DMatrix dA_fold = dA_slice.foldcols(pv->clyr.np);
    dA_fold.set_name("A_panel");
    DMatrix dY_rect = DMatrix(dA_slice.nrow, bw, b_sub, pv->prect);
    dY_rect.set_name("Y_rect");
    MPI_Alltoall(dA_fold.data, dA_fold.get_mysize()/pv->clyr.np, MPI_DOUBLE,
                 dY_rect.data, dA_fold.get_mysize()/pv->clyr.np, MPI_DOUBLE,
                 pv->clyr.cm);
    dY_rect.print();
    DQRMatrix dY_rectQR = DQRMatrix(dY_rect);

    DMatrix dR_fsq = DMatrix(bw/pv->clyr.np, bw*pv->clyr.np, b_sub, pv->plyr, "R");

    //Write R back and set lower panel to zero
    MPI_Allgather(dY_rectQR.R, dY_rectQR.R.get_mysize(), MPI_DOUBLE,
                  dR_fsq, dY_rectQR.R.get_mysize(), MPI_DOUBLE,
                  pv->clyr.cm);

    DMatrix dR_sq = dR_fsq.foldrows(pv->clyr.np);
    dR_fsq.destroy();
    dR_sq.print();

    DMatrix dB = dA.slice(bw,bw,0,bw);
    dB.set_to_zero();
    dB.daxpy(dR_sq, 1.0);   
    dB = dA.slice(0,bw,bw,bw);
    dB.set_to_zero();
    dB.daxpy(dR_sq.transp(), 1.0);   
    dR_sq.destroy();
    if (dA.nrow > 2*bw){
      dB = dA.slice(2*bw,dA.nrow-2*bw,0,bw);
      dB.set_to_zero();
      dB = dA.slice(0,bw,2*bw,dA.nrow-2*bw);
      dB.set_to_zero();
    }

    dVT_lyr.move_ptr(bw, dVT_lyr.nrow-bw, 0, dVT_lyr.ncol);
    dU_lyr. move_ptr(bw,  dU_lyr.nrow-bw, 0, dU_lyr.ncol);
    dA.move_ptr(bw, dA.nrow-bw, bw, dA.ncol-bw);

    dY_rect.destroy();
    dY_rect = dY_rectQR.Y;
    dY_rect.set_name("Y_rect");

    DMatrix dfY_lyr = dA_fold.clone();
    dA_fold.destroy();

    MPI_Alltoall(dY_rect, dY_rect.get_mysize()/pv->clyr.np, MPI_DOUBLE,
                 dfY_lyr,  dY_rect.get_mysize()/pv->clyr.np, MPI_DOUBLE,
                 pv->clyr.cm);

    DMatrix dY_lyr = dfY_lyr.foldrows(pv->clyr.np);
    dfY_lyr.destroy();
    dY_lyr.set_name("Y_lyr");

    DMatrix dW_lyr = DMatrix(dY_lyr.nrow, dY_lyr.ncol, b_sub, pv->plyr, "W_lyr"); 
    if (istep>0){
      //compute W=(-UV'-VU')*Y
      DMatrix dZ_lyr = DMatrix(dVT_lyr.ncol, dY_lyr.ncol, b_sub, pv->plyr, "Z_lyr");
      cpdgemm(dVT_lyr.transp(), dY_lyr, 1.0, dZ_lyr, 0.0);
      cpdgemm(dU_lyr, dZ_lyr, -1.0, dW_lyr, 0.0);
      cpdgemm(dU_lyr.transp(), dY_lyr, 1.0, dZ_lyr, 0.0);
      cpdgemm(dVT_lyr, dZ_lyr, -1.0, dW_lyr, 1.0);
      dZ_lyr.destroy();
    } else {
      //set W = 0
      dW_lyr.set_to_zero();
    }
    //compute W=W+A*Y
    //cpdgemm(dA, dY_lyr, 1.0, dW_lyr, 1.0);
    CTF_Timer tmr1("manual W=A*Y");
    tmr1.start();
    DMatrix dY_lyr_trsp = dY_lyr.transpose_data();
    double * dY_lyr_rep = dY_lyr_trsp.replicate_vertical();
    double * dW_lyr_cntrb = (double*)malloc(sizeof(double)*dY_lyr.ncol*dY_lyr.get_mynrow());
    cdgemm('N', 'N', dA.get_mynrow(), dY_lyr.ncol, dA.get_myncol(), 1.0,
           dA, dA.lda, 
           dY_lyr_rep, dY_lyr.get_mynrow(), 0.0,
           dW_lyr_cntrb, dA.get_mynrow());
    dW_lyr.reduce_scatter_horizontal(dW_lyr_cntrb);
    free(dY_lyr_rep);
    free(dW_lyr_cntrb);
    dY_lyr_trsp.destroy();

    tmr1.stop();
    
    dY_lyr.destroy();
    
    DMatrix dW_fold = dW_lyr.foldcols(pv->clyr.np);
    dW_lyr.destroy();

    assert(dW_fold.get_mysize()%pv->clyr.np == 0);

    DMatrix dW_rect = DMatrix(dY_rect.nrow, dY_rect.ncol, b_sub, pv->prect, "W_rect"); 
    dW_rect.set_to_zero();

    MPI_Alltoall(dW_fold, dW_fold.get_mysize()/pv->clyr.np, MPI_DOUBLE,
                 dW_rect, dW_fold.get_mysize()/pv->clyr.np, MPI_DOUBLE,
                 pv->clyr.cm);
    dW_fold.destroy();
    dW_rect.print();

    DMatrix dZ_rect = DMatrix(dY_rect.ncol, dW_rect.ncol, b_sub, pv->prect, "Z_rect");

    //Z = Y'W = Y'AY or Y(A-UV'-VU')Y 
    cpdgemm(dY_rect.transp(), dW_rect, 1.0, dZ_rect, 0.0);

    dZ_rect.print();

    //compute T^-T from Y
    DMatrix invT = dY_rectQR.compute_invT();
    invT.set_name("invT");

    invT.print();

    dY_rect.print();   
    //compute U = YT'
    cpdtrsm(invT, 1.0, dY_rect);
    invT.destroy();

    //rename Y->U
    DMatrix dU_rect = dY_rect;
    dU_rect.set_name("U_rect");
    
    dU_rect.print();   
 
    DMatrix dfnU_lyr = DMatrix(dA_slice.nrow/pv->clyr.np, bw*pv->clyr.np, b_sub, pv->plyr, "U_lyr");

    MPI_Allgather(dU_rect, dU_rect.get_mysize(), MPI_DOUBLE,
                  dfnU_lyr,  dU_rect.get_mysize(), MPI_DOUBLE,
                  pv->clyr.cm);
    
    DMatrix dnU_lyr = dfnU_lyr.foldrows(pv->clyr.np);
    dfnU_lyr.destroy();
    if (pv->clyr.rank == 0)
      dnU_lyr.print();
    
    int64_t cur_dU_ncol = dU_lyr.ncol;
    int64_t cur_dU_nrow = dU_lyr.nrow;
    
    dU_lyr.move_ptr(0, cur_dU_nrow, cur_dU_ncol, bw);

    dU_lyr.daxpy(dnU_lyr, 1.0);
    dnU_lyr.destroy();
    
    dU_lyr.move_ptr(0, cur_dU_nrow, -cur_dU_ncol, cur_dU_ncol+bw);
       
    //rename W->V
    DMatrix dVT_rect = dW_rect;
    dVT_rect.set_name("V_rect");
  
    // Transform W into V' = W - .5ZU' = Y'A - .5Y'AYTY' 
    cpdgemm(dU_rect, dZ_rect, -.5, dVT_rect, 1.0);
    dZ_rect.destroy();

    dVT_rect.print();

    DMatrix dfnVT_lyr = DMatrix(dA_slice.nrow/pv->clyr.np, bw*pv->clyr.np, b_sub, pv->plyr, "V^T_lyr");
   
    MPI_Allgather(dVT_rect, dVT_rect.get_mysize(), MPI_DOUBLE,
                  dfnVT_lyr,   dVT_rect.get_mysize(), MPI_DOUBLE,
                  pv->clyr.cm);

    dVT_rect.destroy();
    
    DMatrix dnVT_lyr = dfnVT_lyr.foldrows(pv->clyr.np);
    dfnVT_lyr.destroy();

    int64_t cur_dV_ncol = dVT_lyr.ncol;
    int64_t cur_dV_nrow = dVT_lyr.nrow;
    
    dVT_lyr.move_ptr(0, cur_dV_nrow, cur_dV_ncol, bw);

    dVT_lyr.daxpy(dnVT_lyr, 1.0);
    dnVT_lyr.destroy();
    
    dVT_lyr.move_ptr(0, cur_dV_nrow, -cur_dV_ncol, cur_dV_ncol+bw);

    pv->plyr.rrow  = (pv->plyr.rrow + bw/b_sub)%pv->plyr.ccol.np;
    pv->plyr.rcol  = (pv->plyr.rcol + bw/b_sub)%pv->plyr.crow.np;
    pv->prect.rrow = pv->plyr.rrow;
    pv->prect.rcol = pv->plyr.rcol;
  }
  if (n>b_agg){
    //update trailing matrix with aggregated U and V on each layer
    int trail_lyrcols = (n-b_agg)/pv->clyr.np;
    DMatrix dA_slice = dA.slice(0, dA.nrow, trail_lyrcols*pv->clyr.rank, trail_lyrcols);
    dA_slice = dA_slice.get_contig();
    if (pv->clyr.rank == 0){
      dA_slice.print();
    }
    
    DMatrix dU_slice  =  dU_lyr.slice(trail_lyrcols*pv->clyr.rank, trail_lyrcols, 0, dU_lyr.ncol);
    DMatrix dVT_slice = dVT_lyr.slice(trail_lyrcols*pv->clyr.rank, trail_lyrcols, 0, dVT_lyr.ncol);
    
    //cpdgemm(dU_lyr, dVT_slice.transp(), -1.0, dA_slice, 1.0);
    //cpdgemm(dVT_lyr, dU_slice.transp(), -1.0, dA_slice, 1.0);

    CTF_Timer tmr("manual A-UV'-VU' update");
    tmr.start();
    DMatrix dUT_slice = dU_slice.transpose_data();
    DMatrix dV_slice  = dVT_slice.transpose_data();

    //dU_slice.destroy();
    //dVT_slice.destroy();

    double * dUT_slice_rep = dUT_slice.replicate_vertical(); 
    double * dV_slice_rep = dV_slice.replicate_vertical(); 
    double * dU_rep = dU_lyr.replicate_horizontal(); 
    double * dVT_rep = dVT_lyr.replicate_horizontal();

    cdgemm('N', 'T', dA_slice.get_mynrow(), dA_slice.get_myncol(), dU_lyr.ncol, -1.0,
           dU_rep, dA_slice.get_mynrow(), 
           dV_slice_rep, dA_slice.get_myncol(), 1.0,
           dA_slice, dA_slice.lda);
    
    cdgemm('N', 'T', dA_slice.get_mynrow(), dA_slice.get_myncol(), dU_lyr.ncol, -1.0,
           dVT_rep, dA_slice.get_mynrow(), 
           dUT_slice_rep, dA_slice.get_myncol(), 1.0,
           dA_slice, dA_slice.lda);
    tmr.stop();

    free(dUT_slice_rep);
    free(dV_slice_rep);
    free(dU_rep);
    free(dVT_rep);

    dUT_slice.destroy();
    dV_slice.destroy();
     
   
    DMatrix dA_trail = DMatrix(n-b_agg, n-b_agg, b_sub, pv->plyr, "At_lyr");
    
    MPI_Allgather(dA_slice, dA_slice.get_mysize(), MPI_DOUBLE,
                  dA_trail, dA_slice.get_mysize(), MPI_DOUBLE,
                  pv->clyr.cm);

    dA_trail.print();
    dA.set_to_zero();
    dA.daxpy(dA_trail, 1.0);

    dA_slice.destroy();
    dA_trail.destroy();
  }
  dU_lyr.destroy();
  dVT_lyr.destroy();
  
 
  if (n>b_agg+bw)  
    sym_full2band_3d(dA, lda_A, n-b_agg, b_agg, bw, b_sub, pv);

/*  DMatrix dA2 = DMatrix(n, n, b_sub, pv->plyr, lda_A, A);
  dA2.name = "A-UVT-VUT\n";
  if (pv->clyr.rank == 0)
    dA2.print();*/
}

