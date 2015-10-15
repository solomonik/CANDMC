#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include "lu_25d_pvt.h"
#include "tnmt_pvt.h"
#include "partial_pvt.h"
#include "../shared/util.h"

//#include "../shared/seq_lu.h"
//#include "../shared/seq_matmul.h"
//#include "../shared/threxecutioner.h"


#ifndef AGG_PVT
#define AGG_PVT 0
#endif

#ifdef OFFLOAD
#include "lu_offload.h"
#endif

int start_signal;


typedef struct lu_25d_state {
  int num_big_blocks_dim;
  int r_blk;
  int i_big;
  int my_num_blocks_dim;
  int i_sm;
  int num_big_trsm_blocks;
  int my_num_big_trsm_blocks;
  int start_L_big_trsm_block;
  int start_U_big_trsm_block;
  int num_L_big_trsm_blocks;
  int num_U_big_trsm_blocks;
} lu_25d_state_t;


static  
void assign_trsm_blocks(const lu_25d_pvt_params_t       *p,
                        lu_25d_state_t                  *s){

  /* Schedule big blocks contiguously among layers (so if c = 4)
   * layers 0-3 take charge of blocks as follows
   *  0 2 2 3 3 3 
   *  0 x x x x x
   *  1 x x x x x 
   *  1 x x x x x
   *  1 x x x x x
   *  2 x x x x x
   *
   */
  const int num_big_blocks_dim  = s->num_big_blocks_dim;
  const int i_big               = s->i_big;
  const int layerRank           = p->layerRank;
  const int c_rep               = p->c_rep;  
  const int act_big_dim         =num_big_blocks_dim-i_big;

  /* Calculate how many TRSM b-gblocks there are in these panels */
  s->num_big_trsm_blocks = 2*act_big_dim-1;

  s->num_L_big_trsm_blocks = act_big_dim/c_rep;
  if (layerRank == 0){
    if (s->num_L_big_trsm_blocks == 0)
      s->num_L_big_trsm_blocks = 1;
    s->start_L_big_trsm_block = 0;
  } else {
    if (s->num_L_big_trsm_blocks == 0){
      if (act_big_dim%c_rep > layerRank){
        s->num_L_big_trsm_blocks++;
        s->start_L_big_trsm_block = s->num_L_big_trsm_blocks*layerRank;
      } else {
        s->start_L_big_trsm_block = (act_big_dim%c_rep)+s->num_L_big_trsm_blocks*layerRank;
      }
    } else  {
      if (act_big_dim%c_rep > layerRank-1){
        s->num_L_big_trsm_blocks++;
        s->start_L_big_trsm_block = act_big_dim/c_rep 
                                    + s->num_L_big_trsm_blocks*(layerRank-1);
      } else {
        s->start_L_big_trsm_block = (act_big_dim%c_rep)+s->num_L_big_trsm_blocks*layerRank;
      }
    }
  } 
  s->num_U_big_trsm_blocks = (act_big_dim-1)/c_rep;
  if ((act_big_dim-1)%c_rep > c_rep-layerRank-1){
    s->num_U_big_trsm_blocks++;
    s->start_U_big_trsm_block = (act_big_dim-1) - s->num_U_big_trsm_blocks
                                                  *(c_rep-layerRank);
  } else {
    s->start_U_big_trsm_block = act_big_dim-1 - s->num_U_big_trsm_blocks
                                                *(c_rep-layerRank)
                                              - ((act_big_dim-1)%c_rep);
  }

  s->my_num_big_trsm_blocks = s->num_L_big_trsm_blocks+
                              s->num_U_big_trsm_blocks;
}


/* Performs CA-pivoting on the top corner block */
static
void pivot_step(lu_25d_pvt_params_t             *p,
                const lu_25d_state_t            *s,
                int                             *mat_I,
                int                             *mat_pvt,
                int                             *pvt_buffer,
                int                             *pvt_top,
                double                          *mat_A,
                double                          *buffer){ 

  int ncol;

  int * P_app, *P_new;
  double *start_A;

  const int r_blk                       = s->r_blk;
  const int my_num_blocks_dim           = s->my_num_blocks_dim;
  const int i_sm                        = s->i_sm;
  const int i_big                       = s->i_big;
  const int num_big_blocks_dim          = s->num_big_blocks_dim;
  const int start_L_big_trsm_block      = s->start_L_big_trsm_block;
  const int num_L_big_trsm_blocks       = s->num_L_big_trsm_blocks;
  const int64_t blockDim            = p->blockDim; 
  const int64_t big_blockDim        = p->big_blockDim; 
  const int num_pes_dim         = p->num_pes_dim;
  const int layerRank           = p->layerRank;
  const int c_rep               = p->c_rep;
  const int myRow               = p->myRow;
  const int myCol               = p->myCol;
#ifdef DEBUG
  const int myRank              = p->myRank;
#endif
  CommData cdt_row     = p->cdt_row;
  CommData cdt_col     = p->cdt_col;
  CommData cdt_kdir    = p->cdt_kdir;
  CommData cdt_kcol    = p->cdt_kcol;
 
  const int i_bb = i_sm/num_pes_dim;
  const int i_sb = i_sm%num_pes_dim;
  const int first_active_col = (i_bb + (i_sb > myCol));
  int num_col_blk, idx_off, top_off, num_act_pes, num_big_act, num_act_layers;
 
  /* Figure out who is involved in pivoting */ 
  num_big_act = (num_big_blocks_dim-i_big);
  num_act_layers = MIN(num_big_act,c_rep);
  if (layerRank == 0){
    num_col_blk = r_blk - (i_bb + (i_sb > myRow)) 
                         + (num_L_big_trsm_blocks-1)*r_blk;
    idx_off = (i_bb+(i_sb>myRow)+i_big*r_blk)*blockDim;
    top_off = (i_bb+(i_sb>myRow)+i_big*r_blk)*blockDim;
    if (num_L_big_trsm_blocks == 1 && i_bb == r_blk-1)
      num_act_pes = num_pes_dim-i_sb;
    else 
      num_act_pes = num_pes_dim;
  } else {
    num_col_blk = num_L_big_trsm_blocks*r_blk;
    top_off = (i_bb+(i_sb>myRow)+i_big*r_blk)*blockDim;
    idx_off = (i_big+start_L_big_trsm_block)*r_blk*blockDim;
    num_act_pes = num_pes_dim;
  }
  RANK_PRINTF(myRank,myRank,"pivoting among %d blocks col=%d, i_sb=%d\n",num_col_blk,myCol,i_sb);
  if (!p->is_tnmt_pvt){
    TAU_FSTART(partial_pvt);
    if (myCol == i_sb/* && num_L_big_trsm_blocks > 0*/){
      DEBUG_PRINTF("[%d] partial pivoting, idx_off=%d, num_col_blk = %d first_active_col = %d\n",
                    myRank,idx_off,num_col_blk,first_active_col);
      partial_pvt(mat_A+idx_off+first_active_col*blockDim
                                *blockDim*my_num_blocks_dim,
                  my_num_blocks_dim*blockDim,
                  mat_I+idx_off,
                  num_col_blk*blockDim,
                  blockDim,
                  myRow+layerRank*num_pes_dim,
                  num_pes_dim*num_act_layers,
                  i_sb,
                  cdt_kcol);
      P_app = pvt_buffer;
      memcpy(P_app, mat_I+idx_off, blockDim*sizeof(int));
#ifdef TAU
      MPI_Barrier(cdt_row.cm);
#endif
    //  printf("%d %d pivoting\n",cdt_row.rank,cdt_col.rank );
    } else {
      P_app = pvt_buffer;
    //  printf("%d %d waiting\n",cdt_row.rank,cdt_col.rank );
#ifdef TAU
      TAU_FSTART(partial_pvt_wait);
      MPI_Barrier(cdt_row.cm);
      TAU_FSTOP(partial_pvt_wait);
#endif
    }
    TAU_FSTOP(partial_pvt);
  } else {
    TAU_FSTART(tnmt_pvt);
    if (myCol == i_sb && num_L_big_trsm_blocks > 0){
      DEBUG_PRINTF("[%d] performing local tourmaent, idx_off=%d, num_col_blk = %d first_active_col = %d\n",
             myRank,idx_off,num_col_blk,first_active_col);
      /* Get my best rows via a single tall skinny LU */
      if (num_col_blk > 0){
        TAU_FSTART(local_tnmt);
        local_tournament_col_maj(mat_A+idx_off+first_active_col*blockDim
                                               *blockDim*my_num_blocks_dim,
                                 buffer,
                                 pvt_buffer,
                                 num_col_blk*blockDim,
                                 blockDim,
                                 my_num_blocks_dim*blockDim);
        TAU_FSTOP(local_tnmt);
        /* Apply the IPIV swap matrix to an identity, to get the absolute pivot matrix */
        pivot_conv(blockDim, pvt_buffer, mat_I+idx_off);
      }
      

      P_app = pvt_buffer;
      P_new = pvt_buffer + 3*blockDim;

      DEBUG_PRINTF("[%d] performing column tourmaent %d %d\n",myRank, num_col_blk,num_act_pes);
      /* Perform a CA-pivoting binary tournament among all pes in column */
      TAU_FSTART(col_tnmt);
      tnmt_pvt_1d(buffer, 
                  buffer + 2*blockDim*blockDim,
                  mat_I+idx_off,
                  buffer + 4*blockDim*blockDim,
                  P_app,
                  blockDim,
                  myRow,
                  num_pes_dim-num_act_pes,
                  num_act_pes,
                  i_sb,
                  cdt_col);
      TAU_FSTOP(col_tnmt);
      RANK_PRINTF(myRow,i_sb,"[%d,%d%d] P_app el = %d, %d // %d\n",
                  myRow,myCol,layerRank,*P_app,P_app[1],idx_off);
    }
    else {
      P_app = pvt_buffer;
      P_new = pvt_buffer + 3*blockDim;
    }
    if (myRow == i_sb && myCol == i_sb && c_rep > 1){
      DEBUG_PRINTF("[%d] performing layer tourmaent\n",myRank);
      P_new = pvt_buffer;
      P_app = pvt_buffer + 3*blockDim;
      /* Perform a CA-pivoting binary tournament among all layers */
      TAU_FSTART(kdir_tnmt);
      tnmt_pvt_1d(buffer + 2*blockDim*blockDim, 
                  buffer,
                  P_new,
                  buffer + 4*blockDim*blockDim,
                  P_app,
                  blockDim,
                  layerRank,
                  0,
                  num_act_layers,
                  0,
                  cdt_kdir);
      TAU_FSTOP(kdir_tnmt);
      RANK_PRINTF(myRow,i_sb,"[%d,%d%d] P_app el = %d, %d // %d\n",
                  myRow,myCol,layerRank,*P_app,P_app[1],idx_off);
    }
    else {
      P_app = pvt_buffer;
      P_new = pvt_buffer + 3*blockDim;
    }
#ifdef TAU
    TAU_FSTART(tnmt_pvt_wait);
    MPI_Barrier(cdt_row.cm);
    TAU_FSTOP(tnmt_pvt_wait);
#endif
    TAU_FSTOP(tnmt_pvt);
  }

  ncol = blockDim*r_blk;
  start_A = mat_A + top_off;

  /*printf("myRank = %d\n",myRank);

  if (myRow == i_sb && myCol == i_sb){
    for (int i=0; i<blockDim; i++){
      printf("P_app[%d] = %d\n", i, P_app[i]);
    }
  }*/
 
  if (c_rep > 1 && myRow == i_sb){
    if (myCol == i_sb)
      MPI_Bcast(P_app, blockDim, MPI_INT, 0, cdt_kdir.cm); 
    if (layerRank == 0 && myRow == i_sb)
      lda_cpy(blockDim, ncol, my_num_blocks_dim*blockDim, blockDim, start_A, buffer);
    MPI_Bcast(buffer, blockDim*ncol, MPI_DOUBLE, 0, cdt_kdir.cm); 
    MPI_Bcast(mat_pvt+top_off, blockDim, MPI_INT, 0, cdt_kdir.cm); 
    if (layerRank > 0)
      lda_cpy(blockDim, ncol, blockDim, my_num_blocks_dim*blockDim, buffer, start_A);
  }
  
#if (USE_GATHER_SB==0)
  if (num_L_big_trsm_blocks > 0 || layerRank == 0){
#endif
    RANK_PRINTF(myRank,myRank,"try applying the pivoting i_sb=%d\n",i_sb);
    if (myRow == i_sb)
      MPI_Bcast(P_app, blockDim, MPI_INT, i_sb, cdt_row.cm); 
    
/*    RANK_PRINTF(myRank,myRank,"applying the pivoting %d %d %lf %lf\n",P_app[0],
                idx_off, mat_A[0], mat_A[1]);*/
    /*for (int j=0; j<num_pes_dim*num_pes_dim; j++){
      if( myRank == j){
        printf("before par_pivot rank %d matrix:\n", myRank);
        print_matrix(mat_A, p->matrixDim/cdt_col.np, p->matrixDim/cdt_row.np);
      }
    }*/
    TAU_FSTART(par_pivot);
    par_pivot(start_A,
              buffer, 
              blockDim,
              ncol,
              blockDim,
              my_num_blocks_dim*blockDim,
              top_off,
              i_big*big_blockDim+i_bb*blockDim*num_pes_dim+i_sb*blockDim,
              mat_pvt+top_off,
              P_app,
              myRow,
              layerRank,
              num_pes_dim,
              c_rep,
              i_sb,
              (i_big+start_L_big_trsm_block)*r_blk+i_bb,
              num_L_big_trsm_blocks*r_blk,
              cdt_col,
              cdt_kdir,
              pvt_top,
              p->is_tnmt_pvt,
              (i_bb+(myCol <= i_sb))*blockDim);
/*    for (int j=0; j<num_pes_dim*num_pes_dim; j++){
      if( myRank == j){
        printf("after par_pivot rank %d matrix:\n", myRank);
        print_matrix(mat_A, p->matrixDim/cdt_col.np, p->matrixDim/cdt_row.np);
      }
    }*/
    TAU_FSTOP(par_pivot);
/*    printf("mat_pvt[0] = %d [1] = %d\n",mat_pvt[0],mat_pvt[1]);
    printf("P_app[0] = %d [1] = %d\n",P_app[0],P_app[1]);*/
#if (USE_GATHER_SB==0)
  } else {
    std::fill(buffer, buffer+ncol*blockDim, 0.0);
    if (myRow == i_sb)
      std::fill(mat_pvt+(i_big*r_blk+i_bb)*blockDim,
                mat_pvt+(i_big*r_blk+i_bb+1)*blockDim, 0);
  }
  if (myRow == i_sb && c_rep > 1) {
    RANK_PRINTF(myRank,myRank,"collecting pivoting results\n");
    TAU_FSTART(coll_piv);
    MPI_Allreduce(MPI_IN_PLACE,buffer,
              blockDim*ncol,MPI_DOUBLE,
              MPI_SUM,cdt_kdir.cm);
    MPI_Allreduce(MPI_IN_PLACE, mat_pvt+(i_big*r_blk+i_bb)*blockDim,
              blockDim,MPI_INT,
              MPI_SUM,cdt_kdir.cm);
//    memcpy(mat_pvt+(i_big*r_blk+i_bb)*blockDim,P_new,blockDim*sizeof(int));
    lda_cpy(blockDim, ncol, blockDim, 
            blockDim*my_num_blocks_dim, buffer, 
            start_A - top_off + (i_big*r_blk+i_bb)*blockDim);
    TAU_FSTOP(coll_piv);
  }
#endif
}

/* Factorizes small block panels */
static
void panel_trsm(const lu_25d_pvt_params_t       *p,
                const lu_25d_state_t            *s,
                const int                       LU_mask, /* 1 -> L, 2-> U */
                int                             *mat_I,
                int                             *mat_pvt,
                int                             *pvt_buffer,
                double                          *&buf_update_L,
                double                          *&buf_update_U,
                double                          *advance_A,
                double                          *mat_A,
                double                          *buffer){ 

  int info, lda_cuL;

  double * curr_A, * curr_upd_L, * curr_upd_U; 

  const int r_blk                       = s->r_blk;
  const int my_num_blocks_dim           = s->my_num_blocks_dim;
  const int64_t i_sm                        = s->i_sm;
  const int64_t i_big                       = s->i_big;
  const int my_num_big_trsm_blocks      = s->my_num_big_trsm_blocks;
  const int start_L_big_trsm_block      = s->start_L_big_trsm_block;
  const int start_U_big_trsm_block      = s->start_U_big_trsm_block;
  const int num_L_big_trsm_blocks       = s->num_L_big_trsm_blocks;
  const int num_U_big_trsm_blocks       = s->num_U_big_trsm_blocks;
  const int64_t blockDim            = p->blockDim; 
  const int num_pes_dim         = p->num_pes_dim;
  const int layerRank           = p->layerRank;
  const int myRow               = p->myRow;
  const int myCol               = p->myCol;
#ifdef DEBUG
  const int myRank              = p->myRank;
#endif
  const CommData cdt_row       = p->cdt_row;
  const CommData cdt_col       = p->cdt_col;
  const CommData cdt_kdir      = p->cdt_kdir;
 
  const int i_bb = i_sm/num_pes_dim;
  const int i_sb = i_sm%num_pes_dim;
  const int num_active_row = r_blk - (i_bb + (i_sb >= myRow));
  const int num_active_col = r_blk - (i_bb + (i_sb >= myCol));
  const int first_active_row = r_blk - num_active_row;
  const int first_active_col = r_blk - num_active_col;
  
  curr_upd_L = buffer;
  curr_upd_U = buffer;
  buf_update_L = buffer+blockDim*blockDim;
  buf_update_U = buffer+blockDim*blockDim*
                       (1+((num_L_big_trsm_blocks-(layerRank==0))*r_blk+num_active_row));

  /* Factorize top left corner small block 
   *LU x x | x x x 
   * x x x | x x x
   * x x x | x x x
   * -------------
   * x x x | x x x
   * x x x | x x x
   * x x x | x x x
   */

  if ((myCol == myRow) && (myCol == i_sb)){
    RANK_PRINTF(myRank,(i_sb*num_pes_dim+i_sb),"[%d] Factorizing corner small block  " PRId64 "\n", myRank, i_sm);
    curr_A = advance_A+(i_bb*blockDim*my_num_blocks_dim+i_bb)*blockDim;
    /* Only the first layer should factorize, since its the only one that has
       the correct matrix to factorize */
    if (LU_mask&0x1){
      if (layerRank == 0){
        if (p->is_tnmt_pvt || !p->pvt){
          TAU_FSTART(corner_LU);
          cdgetrf(blockDim,       blockDim,
                  curr_A,         blockDim*my_num_blocks_dim,
                  pvt_buffer,     &info);
          TAU_FSTOP(corner_LU);
        }
        lda_cpy(blockDim,                       blockDim,
                blockDim*my_num_blocks_dim,     blockDim,
                curr_A,                         buffer);
#if 0
        lda_cpy(blockDim,                       blockDim,
                blockDim*my_num_blocks_dim,     blockDim,
                curr_A,                         buffer+blockDim*blockDim);
        pivot_mat(blockDim, blockDim, pvt_buffer, 
                  buffer+blockDim*blockDim, buffer);
        lda_cpy(blockDim,    blockDim,
                blockDim,    blockDim*my_num_blocks_dim,
                buffer,      curr_A);
#endif
                
      }
      /* Broadcast factorized top corner to every layer */
      MPI_Bcast(buffer, blockDim*blockDim, MPI_DOUBLE, 0, cdt_kdir.cm); 
      if (layerRank > 0){
        lda_cpy(blockDim,       blockDim,
                blockDim,       blockDim*my_num_blocks_dim,
                buffer,         curr_A);
      } 
      if (p->pvt && p->is_tnmt_pvt) {
        MPI_Bcast(pvt_buffer, blockDim, MPI_INT, 0, cdt_kdir.cm);
        if (num_active_col > 0){
          cdlaswp(blockDim*num_active_col, 
                  curr_A + blockDim*blockDim*my_num_blocks_dim,
                  my_num_blocks_dim*blockDim,
                  1, blockDim, pvt_buffer, 1);
        } if (i_bb > 0){
          cdlaswp(blockDim*i_bb, 
                  advance_A + blockDim*i_bb,
                  my_num_blocks_dim*blockDim,
                  1, blockDim, pvt_buffer, 1);
        }
      }
    } else {
      lda_cpy(blockDim,                         blockDim,
              blockDim*my_num_blocks_dim,       blockDim,
              curr_A,                           buffer);
    }
    /* Now that each layer has the factorized top corner block, broadcast it
       down each row and column of the processor grid in order to do updates */
    if (my_num_big_trsm_blocks > 0) {
      MPI_Bcast(buffer, blockDim*blockDim, MPI_DOUBLE, i_sb, cdt_row.cm); 
      if (LU_mask&0x1){
        if (!p->pvt || p->is_tnmt_pvt){
          MPI_Bcast(buffer, blockDim*blockDim, MPI_DOUBLE, i_sb, cdt_col.cm); 
        }
        if (p->is_tnmt_pvt){
          if (p->pvt) MPI_Bcast(pvt_buffer, blockDim, MPI_INT, i_sb, cdt_row.cm);
        }
      }
    }
  } 
  /* Now lets perform the updates on the big-block row and columns 
   * \ U U | U U U 
   * L x x | x x x
   * L x x | x x x
   * -------------
   * L x x | x x x
   * L x x | x x x
   * L x x | x x x
   */
  /* At first we only compute one small-block row/column of our TRSM blocks */
  /* Compute small_block column TRSMs */
  if ((LU_mask&0x1) && myCol == i_sb){
    RANK_PRINTF(myRank,(i_sb*num_pes_dim+i_sb),"Doing small block column updates\n");
    if (my_num_big_trsm_blocks > 0) {
      /* Receive U from top block factorization */
      if (p->is_tnmt_pvt || !p->pvt){
        if (myRow != i_sb){
          MPI_Bcast(curr_upd_U, blockDim*blockDim, MPI_DOUBLE, i_sb, cdt_col.cm); 
        }
      }
      
      if (num_L_big_trsm_blocks>0){
        /* If I am responsible for the top corner big block */
        if (start_L_big_trsm_block == 0){
          if (num_active_row > 0 || num_L_big_trsm_blocks > 1){
            PRINT_INT(first_active_row);
            /* Perform TRSM on the subcolumn of small blocks I am responsible for */
            if (p->is_tnmt_pvt || !p->pvt){
              cdtrsm('R', 'U', 'N', 'N', 
                     blockDim*(r_blk*num_L_big_trsm_blocks
                                -first_active_row), 
                     blockDim,
                     1.0, curr_upd_U, blockDim,
                     advance_A + blockDim*(i_bb*blockDim*my_num_blocks_dim 
                                            + first_active_row),
                     my_num_blocks_dim*blockDim);
            }
            /* Copy computed subcolumn of small blocks to contiguous buffer */
            lda_cpy(blockDim*(r_blk*num_L_big_trsm_blocks
                                  -first_active_row), 
                    blockDim,
                    my_num_blocks_dim*blockDim,
                    blockDim*(r_blk*num_L_big_trsm_blocks
                                   -first_active_row),
                    advance_A + blockDim*(i_bb*blockDim*my_num_blocks_dim
                                   +first_active_row),
                    buf_update_L);
            /* Copy the top block part of the computed subcolumn into a separate contiguous buffer */
            lda_cpy(blockDim*(r_blk-first_active_row), 
                    blockDim,
                    blockDim*(r_blk*num_L_big_trsm_blocks
                                    -first_active_row),
                    blockDim*(r_blk-first_active_row),
                    buf_update_L,
                    buf_update_L+num_L_big_trsm_blocks*blockDim
                                  *blockDim*r_blk);
          }     
        } else {
          /* Perform TRSM on the subcolumn of small blocks I am responsible for */
          if (p->is_tnmt_pvt || !p->pvt){
            cdtrsm('R', 'U', 'N', 'N', 
                   blockDim*r_blk*num_L_big_trsm_blocks, blockDim,
                   1.0, curr_upd_U, blockDim,
                   advance_A+i_bb*blockDim*blockDim*my_num_blocks_dim
                            +start_L_big_trsm_block*blockDim*r_blk,
                   my_num_blocks_dim*blockDim);
          }
          /* Copy computed subcolumn of small blocks to contiguous buffer */
          lda_cpy(blockDim*r_blk*num_L_big_trsm_blocks, 
                  blockDim,
                  my_num_blocks_dim*blockDim, 
                  blockDim*r_blk*num_L_big_trsm_blocks,
                  advance_A+i_bb*blockDim*blockDim*my_num_blocks_dim
                           +start_L_big_trsm_block*blockDim*r_blk,
                  buf_update_L);
        }
      }
    }
    /* Every layer needs the top part of the L factorization for panel updates.
     * Therefore, distribute it across layers than broadcast. */
    if (num_active_row > 0){ 
      MPI_Bcast(buf_update_L+num_L_big_trsm_blocks*blockDim*blockDim*r_blk,
            num_active_row*blockDim*blockDim, MPI_DOUBLE, 0, cdt_kdir.cm); 
    }
  }
  /* Compute small block row TRSMs */
  if (myRow == i_sb){
    RANK_PRINTF(myRank,(i_sb*num_pes_dim+i_sb),"Doing small block row updates\n");
    if (my_num_big_trsm_blocks > 0) {
      /* Receive L from top block factorization */
      if (myCol != i_sb){
        MPI_Bcast(curr_upd_L, (blockDim*blockDim), MPI_DOUBLE, 
              i_sb, cdt_row.cm); 
        lda_cuL = blockDim;
        if (p->pvt && p->is_tnmt_pvt && LU_mask&0x1) 
          MPI_Bcast(pvt_buffer, blockDim, MPI_INT, i_sb, cdt_row.cm);
      } else {
        lda_cuL = blockDim;
//      curr_upd_L = advance_A+(i_bb*blockDim*my_num_blocks_dim+i_bb)*blockDim;
//      lda_cuL = blockDim*my_num_blocks_dim;
      }
      if (p->pvt && p->is_tnmt_pvt && LU_mask&0x1) 
//        pivot_conv_direct(blockDim, pvt_buffer, mat_pvt+(i_big*r_blk+i_bb)*blockDim);
        pivot_conv(blockDim, pvt_buffer, mat_pvt+(i_big*r_blk+i_bb)*blockDim);
      if (num_U_big_trsm_blocks>0 || start_L_big_trsm_block == 0){
        /* If I am responsible for the top corner big block */
        if (layerRank == 0){
          if (p->pvt && p->is_tnmt_pvt && (LU_mask&0x1) && myCol != i_sb){ 
#if AGG_PVT
            cdlaswp(blockDim*my_num_blocks_dim,
                    mat_A + blockDim*(i_bb + r_blk*i_big),
                    my_num_blocks_dim*blockDim,
                    1, blockDim, pvt_buffer, 1);
#else
            cdlaswp(blockDim*r_blk, 
                    advance_A + blockDim*i_bb,
                    my_num_blocks_dim*blockDim,
                    1, blockDim, pvt_buffer, 1);
          }
#endif
          /* Perform TRSM on the subrow of small blocks I am responsible for */
          cdtrsm('L', 'L', 'N', 'U', 
                 blockDim, 
                 blockDim*(r_blk*num_U_big_trsm_blocks*((LU_mask&0x2)>>1)
                            +num_active_col*((LU_mask&0x1))), 
                 1.0, curr_upd_L, lda_cuL,
                  advance_A + i_bb*blockDim 
                      + (first_active_col*(LU_mask&0x1) + r_blk*((!(LU_mask&0x1))&0x1))
                        *blockDim*blockDim*my_num_blocks_dim,
                 my_num_blocks_dim*blockDim);
          /* Copy computed subrow of small blocks to contiguous buffer */
          lda_cpy(blockDim,
                  blockDim*(r_blk*num_U_big_trsm_blocks*((LU_mask&0x2)>>1)
                            +num_active_col*((LU_mask&0x1))), 
                  my_num_blocks_dim*blockDim,
                  blockDim,
                  advance_A + i_bb*blockDim 
                      + (first_active_col*(LU_mask&0x1) + r_blk*((!(LU_mask&0x1))&0x1))
                        *blockDim*blockDim*my_num_blocks_dim,
                  buf_update_U);
          RANK_PRINTF(myRank,myRank,"buL %lf\n", *buf_update_L);
                  
        } else if ((LU_mask&0x2)>>1) {
          RANK_PRINTF(myRank,myRank,"doing trsm updates %d\n",
                      start_U_big_trsm_block);
          /* Perform TRSM on the subrow of small blocks I am responsible for */
          cdtrsm('L', 'L', 'N', 'U', 
                 blockDim, 
                 blockDim*r_blk*num_U_big_trsm_blocks, 
                 1.0, curr_upd_L, lda_cuL,
                 advance_A + i_bb*blockDim 
                           + (start_U_big_trsm_block+1)*blockDim*blockDim       
                                *my_num_blocks_dim*r_blk,
                 my_num_blocks_dim*blockDim);
          /* Copy computed subrow of small blocks to contiguous buffer */
          lda_cpy(blockDim,
                  blockDim*r_blk*num_U_big_trsm_blocks, 
                  my_num_blocks_dim*blockDim,
                  blockDim,
                  advance_A + i_bb*blockDim 
                            + (start_U_big_trsm_block+1)*blockDim*blockDim      
                              *my_num_blocks_dim*r_blk,
                  buf_update_U);
        }
      }
    }
    /* Every layer needs the leftmost part of the U factorization for panel updates.
     * Therefore, distribute it across layers than broadcast.
     * Remember that this update is always sent last! */
    if ((LU_mask&0x1) && num_active_col > 0){ 
      if (layerRank == 0) {
        MPI_Bcast(buf_update_U, num_active_col*blockDim*blockDim, 
                            MPI_DOUBLE, 0, cdt_kdir.cm); 
      } else {
        MPI_Bcast(buf_update_U+num_U_big_trsm_blocks*blockDim*blockDim*r_blk*((LU_mask&0x2)>>1),
              num_active_col*blockDim*blockDim, MPI_DOUBLE, 0, cdt_kdir.cm); 
      }
    } 
  }
  /* Now we broadcast the updates within layers */
  if (my_num_big_trsm_blocks > 0) {
    RANK_PRINTF(myRank,myRank,"Broadcasting small block L and U\n");
    if ((LU_mask&0x3)) {
      MPI_Bcast(buf_update_L, 
            ((num_L_big_trsm_blocks-(layerRank==0))*r_blk*(LU_mask&0x1)
              + num_active_row)*blockDim*blockDim, 
            MPI_DOUBLE, i_sb, cdt_row.cm); 
    } 
    MPI_Bcast(buf_update_U, 
          (num_U_big_trsm_blocks*r_blk*((LU_mask&0x2)>>1)
           +num_active_col*(LU_mask&0x1))*blockDim*blockDim, 
          MPI_DOUBLE, i_sb, cdt_col.cm); 
  }
}

/* Compute big block updates */
static
void panel_updt(const lu_25d_pvt_params_t       *p,
                const lu_25d_state_t            *s,
                const int                       LU_mask, /* 1 -> L, 2-> U */
                double                          *buf_update_L,
                double                          *buf_update_U,
                double                          *advance_A){

  const int r_blk                       = s->r_blk;
  const int my_num_blocks_dim           = s->my_num_blocks_dim;
  const int i_sm                        = s->i_sm;
  const int start_L_big_trsm_block      = s->start_L_big_trsm_block;
  const int start_U_big_trsm_block      = s->start_U_big_trsm_block;
  const int num_L_big_trsm_blocks       = s->num_L_big_trsm_blocks;
  const int num_U_big_trsm_blocks       = s->num_U_big_trsm_blocks;
  const int64_t blockDim            = p->blockDim; 
  const int num_pes_dim         = p->num_pes_dim;
  const int layerRank           = p->layerRank;
  const int myRow               = p->myRow;
  const int myCol               = p->myCol;
#ifdef DEBUG
  const int myRank              = p->myRank;
#endif
  const CommData cdt_row       = p->cdt_row;
 
  const int i_bb = i_sm/num_pes_dim;
  const int i_sb = i_sm%num_pes_dim;
  const int num_active_row = r_blk - (i_bb + (i_sb >= myRow));
  const int num_active_col = r_blk - (i_bb + (i_sb >= myCol));
  const int first_active_row = r_blk - num_active_row;
  const int first_active_col = r_blk - num_active_col;
#ifdef OFFLOAD_SKINNY_GEMM
  int64_t advance_offset = s->i_big*s->r_blk*s->my_num_blocks_dim*blockDim*blockDim
                          +s->i_big*s->r_blk*blockDim;
#endif

  int lda_buL; 

  /* Update top big block 
   *   | |
   * \ V V | x x x 
   * ->A'A'| x x x
   * ->A'A'| x x x
   * -------------
   * x x x | x x x
   * x x x | x x x
   * x x x | x x x
   */

  /* If I need to do the top big corner block update */
  if (layerRank == 0 && (num_active_row > 0 || num_active_col>0)){
    RANK_PRINTF(myRank,myRank,"Doing big block updates %d %d " PRId64 "\n",
             num_L_big_trsm_blocks,
            first_active_row,
             blockDim*(r_blk*num_L_big_trsm_blocks
                        -first_active_row));
    if (!(LU_mask&0x1)){
      lda_buL = blockDim*num_active_row;
      if (myCol == i_sb){
        lda_cpy(lda_buL,                        blockDim,
                blockDim*my_num_blocks_dim,     lda_buL,
                advance_A + first_active_row*blockDim 
                          + i_bb*blockDim*blockDim*my_num_blocks_dim,
                buf_update_L);
      }
      MPI_Bcast(buf_update_L, lda_buL*blockDim, MPI_DOUBLE, i_sb, cdt_row.cm); 
    } else {
      lda_buL = blockDim*(r_blk*num_L_big_trsm_blocks-first_active_row);
    }
 
    if ((LU_mask&0x1) && num_active_col > 0 && 
        (num_active_row > 0 ||  num_L_big_trsm_blocks > 1)){
      RANK_PRINTF(myRank,myRank,"Doing  dgemm\n");
#ifdef OFFLOAD_SKINNY_GEMM
        assert(lda_buL==((num_L_big_trsm_blocks-1)*r_blk 
                +num_active_row)*blockDim);
#ifdef USE_MIC
#ifdef ASYNC_GEMM
      if (start_signal){
        TAU_FSTART(wait_for_signal);
        wait_gemm();
        TAU_FSTOP(wait_for_signal);
      } else start_signal = 1;
#endif
#endif
      TAU_FSTART(upload_sk_transfer);
      upload_lda_cpy(num_active_row*blockDim,
                     blockDim,
                     lda_buL,
                     lda_buL,
                     buf_update_L,
                     0,
                     OFF_L);
      upload_lda_cpy(blockDim,
                     num_active_col*blockDim,
                     blockDim,
                     blockDim,
                     buf_update_U,
                     0,
                     OFF_U);
/*  offload_A = advance_A-advance_offset;
  #pragma offload target(mic:mic_rank) \
    out (offload_A:length(my_num_blocks_dim*blockDim* \
                         my_num_blocks_dim*blockDim)\
        alloc_if(0) free_if(0) )
  {
  }*/
      int sm_off_L = 0, sm_off_U = 0;
      TAU_FSTOP(upload_sk_transfer);
      TAU_FSTART(download_sk_transfer);
      
      if (myCol == (i_sb+1)%num_pes_dim){
        sm_off_U = 1;
        download_lda_cpy(num_active_row*blockDim,
                         blockDim,
                         my_num_blocks_dim*blockDim,
                         my_num_blocks_dim*blockDim,
                         advance_offset
                          + first_active_row*blockDim 
                          + first_active_col*blockDim*blockDim*my_num_blocks_dim,
                         advance_A
                          + first_active_row*blockDim 
                          + first_active_col*blockDim*blockDim*my_num_blocks_dim,
                         OFF_A);
        /*printf("rank %d downloaded " PRId64 " elements starting with %lf from " PRId64 "\n", 
                myRank, num_active_row*blockDim,
                         advance_A[first_active_row*blockDim 
                         + first_active_col*blockDim*blockDim*my_num_blocks_dim],
                          advance_offset + first_active_row*blockDim 
                          + first_active_col*blockDim*blockDim*my_num_blocks_dim);*/
      
      } 
      if (myRow == (i_sb+1)%num_pes_dim){
        sm_off_L = 1;
        download_lda_cpy(blockDim,
                         num_active_col*blockDim,
                         my_num_blocks_dim*blockDim,
                         my_num_blocks_dim*blockDim,
                         advance_offset
                          + first_active_row*blockDim 
                          + first_active_col*blockDim*blockDim*my_num_blocks_dim,
                         advance_A
                          + first_active_row*blockDim 
                          + first_active_col*blockDim*blockDim*my_num_blocks_dim,
                         OFF_A);
/*        printf("rank %d downloaded " PRId64 " elements starting with %lf from " PRId64 "\n", 
                myRank, num_active_col*blockDim,
                         advance_A[first_active_row*blockDim 
                         + first_active_col*blockDim*blockDim*my_num_blocks_dim],
                          advance_offset + first_active_row*blockDim 
                          + first_active_col*blockDim*blockDim*my_num_blocks_dim);*/
      
      }
      TAU_FSTOP(download_sk_transfer);
      TAU_FSTART(offload_sk_gemm);
      /*printf("[%d] doing " PRId64 " by " PRId64 " by " PRId64 " gemm on offset " PRId64 "\n",
             myRank,
             ((num_L_big_trsm_blocks-1)*r_blk 
              +num_active_row-sm_off_L)*blockDim,
             (num_active_col-sm_off_U)*blockDim,
             blockDim, 
             advance_offset + (first_active_row+sm_off_L)*blockDim
                       + (first_active_col+sm_off_U)*blockDim*blockDim*my_num_blocks_dim);*/

      offload_gemm_A('N','N',
             ((num_L_big_trsm_blocks-1)*r_blk 
              +num_active_row-sm_off_L)*blockDim,
             (num_active_col-sm_off_U)*blockDim,
             blockDim, 
             -1.0, 
             blockDim*sm_off_L,
             OFF_L,
             lda_buL,
             blockDim*blockDim*sm_off_U,
             OFF_U,
             blockDim,
             1.0,
             advance_offset + (first_active_row+sm_off_L)*blockDim 
                       + (first_active_col+sm_off_U)*blockDim*blockDim*my_num_blocks_dim,
             OFF_A,
             blockDim*my_num_blocks_dim);
      TAU_FSTOP(offload_sk_gemm);
      TAU_FSTART(cpu_sk_gemm);
      if (sm_off_U){
        cdgemm('N', 'N', 
               ((num_L_big_trsm_blocks-1)*r_blk 
                +num_active_row)*blockDim,
               blockDim,
               blockDim, 
               -1.0, 
               buf_update_L,
               lda_buL,
               buf_update_U,
               blockDim,
               1.0,
               advance_A + first_active_row*blockDim 
                         + first_active_col*blockDim*blockDim*my_num_blocks_dim,
               blockDim*my_num_blocks_dim);
      } 
      if (sm_off_L){
        cdgemm('N', 'N', 
               blockDim,
               (num_active_col-sm_off_U)*blockDim,
               blockDim, 
               -1.0, 
               buf_update_L,
               lda_buL,
               buf_update_U+sm_off_U*blockDim*blockDim,
               blockDim,
               1.0,
               advance_A + first_active_row*blockDim 
                         + (first_active_col+sm_off_U)*blockDim*blockDim*my_num_blocks_dim,
               blockDim*my_num_blocks_dim);
      }
      TAU_FSTOP(cpu_sk_gemm);
    /*} else {
      TAU_FSTART(offload_sk_gemm);
#ifdef USE_MIC
#ifdef ASYNC_GEMM
  #pragma offload target(mic:mic_rank) signal(gemm_signal)
#else
    #pragma offload target(mic:mic_rank) 
#endif
#endif
        offload_gemm_A('N','N',
               ((num_L_big_trsm_blocks-1)*r_blk 
                +num_active_row)*blockDim,
               num_active_col*blockDim,
               blockDim, 
               -1.0, 
               0,
               OFF_L,
               lda_buL,
               0,
               OFF_U,
               blockDim,
               1.0,
               advance_offset + first_active_row*blockDim 
                         + first_active_col*blockDim*blockDim*my_num_blocks_dim,
               OFF_A,
               blockDim*my_num_blocks_dim);
        TAU_FSTOP(offload_sk_gemm);
      }*/
#else 
      cdgemm('N', 'N', 
             ((num_L_big_trsm_blocks-1)*r_blk 
              +num_active_row)*blockDim,
             num_active_col*blockDim,
             blockDim, 
             -1.0, 
             buf_update_L,
             lda_buL,
             buf_update_U,
             blockDim,
             1.0,
             advance_A + first_active_row*blockDim 
                       + first_active_col*blockDim*blockDim*my_num_blocks_dim,
             blockDim*my_num_blocks_dim);
#endif
    }
    RANK_PRINTF(myRank,myRank,"Did first big block updates\n");
    if (((LU_mask&0x2)>>1) && num_U_big_trsm_blocks > 0 && num_active_row > 0){
      RANK_PRINTF(myRank,myRank,"Doing big block updates\n");
      cdgemm('N', 'N', 
             num_active_row*blockDim,
             num_U_big_trsm_blocks*r_blk*blockDim,
             blockDim, 
             -1.0, 
             buf_update_L,
             lda_buL,
             buf_update_U+blockDim*blockDim*num_active_col*(LU_mask&0x1),
             blockDim,
             1.0,
             advance_A + first_active_row*blockDim 
              + (start_U_big_trsm_block+1)*blockDim*blockDim*my_num_blocks_dim*r_blk,
             blockDim*my_num_blocks_dim);
      RANK_PRINTF(myRank,myRank,"dbb %lf %lf %lf\n",
                  *buf_update_L, *buf_update_U, 
                  *(advance_A + first_active_row*blockDim 
                    + (start_U_big_trsm_block+1)*blockDim*blockDim*my_num_blocks_dim*r_blk));
    }
    PRINT_INT(first_active_row);
  } else {
    if (!(LU_mask&0x1)){
      lda_buL = blockDim*num_active_row;
      if (myCol == i_sb){
        lda_cpy(lda_buL,                        blockDim,
                blockDim*my_num_blocks_dim,     lda_buL,
                advance_A + first_active_row*blockDim 
                          + i_bb*blockDim*blockDim*my_num_blocks_dim,
                buf_update_L);
      }
      MPI_Bcast(buf_update_L, lda_buL*blockDim, MPI_DOUBLE, i_sb, cdt_row.cm); 
    } else {
      lda_buL = blockDim*r_blk*num_L_big_trsm_blocks;
    }
 
    /* If there are L updates I need to do */
    if ((LU_mask&0x1) && num_L_big_trsm_blocks > 0 && num_active_col > 0 &&
        start_L_big_trsm_block > 0){
      RANK_PRINTF(myRank,myRank,"Doing big block updates\n");
      cdgemm('N', 'N', 
             num_L_big_trsm_blocks*r_blk*blockDim,
             num_active_col*blockDim,
             blockDim, 
             -1.0, 
             buf_update_L,
             lda_buL,
             buf_update_U+num_U_big_trsm_blocks*blockDim
                          *blockDim*r_blk*((LU_mask&0x2)>>1),
             blockDim,
             1.0,
             advance_A + first_active_col*blockDim*my_num_blocks_dim*blockDim
                       + start_L_big_trsm_block*blockDim*r_blk,
             blockDim*my_num_blocks_dim);
    }
    /* If there are U updates I need to do */
    if ((LU_mask&0x2)>>1 && num_U_big_trsm_blocks > 0 && num_active_row > 0){
      RANK_PRINTF(myRank,myRank,"Doing big block updates\n");
      cdgemm('N', 'N', 
             num_active_row*blockDim,
             num_U_big_trsm_blocks*r_blk*blockDim,
             blockDim, 
             -1.0, 
             buf_update_L+(LU_mask&0x1)*num_L_big_trsm_blocks*blockDim*blockDim*r_blk,
             blockDim*num_active_row,
             buf_update_U,
             blockDim,
             1.0,
             advance_A + first_active_row*blockDim
                       + (start_U_big_trsm_block+1)*blockDim*blockDim
                          *my_num_blocks_dim*r_blk,
             blockDim*my_num_blocks_dim);
    }
  }
}

static
void xchng_blks(const lu_25d_pvt_params_t       *p,
                const lu_25d_state_t            *s,
                const int                       LU_mask, /* 1 -> L, 2-> U */
                double                          *advance_A,
                double                          *buffer,
                int                             *pvt_acc){
  int i;

  const int r_blk                       = s->r_blk;
  const int i_big                       = s->i_big;
  const int my_num_blocks_dim           = s->my_num_blocks_dim;
  const int my_num_big_trsm_blocks      = s->my_num_big_trsm_blocks;
  const int start_L_big_trsm_block      = s->start_L_big_trsm_block;
  const int start_U_big_trsm_block      = s->start_U_big_trsm_block;
  const int num_L_big_trsm_blocks       = s->num_L_big_trsm_blocks;
  const int num_U_big_trsm_blocks       = s->num_U_big_trsm_blocks;
  const int c_rep               = p->c_rep;  
  const int64_t blockDim            = p->blockDim; 
  const int layerRank           = p->layerRank;
  const CommData cdt_kdir      = p->cdt_kdir;
#ifdef DEBUG
  const int myRank              = p->myRank;
#endif
  
  lu_25d_state_t recv_s = *s; 
  lu_25d_pvt_params_t recv_p    = *p; 
 
  /* Broadcast the panels of L and U we just computed so all layers have them */
  /* FIXME: We are doing redundant communication here */
  RANK_PRINTF(myRank,myRank,"doing trsm block exchanges\n");
  for (i=0; i<c_rep; i++){
    if (layerRank == i && my_num_big_trsm_blocks > 0){
      if ((LU_mask&0x1) && num_L_big_trsm_blocks > 0) {
        lda_cpy(num_L_big_trsm_blocks*blockDim*r_blk,
                blockDim*r_blk,
                blockDim*my_num_blocks_dim,
                num_L_big_trsm_blocks*blockDim*r_blk,
                advance_A+start_L_big_trsm_block*blockDim*r_blk,
                buffer);
      }
      if (((LU_mask&0x2)>>1) && num_U_big_trsm_blocks > 0) {
        lda_cpy(blockDim*r_blk,
                num_U_big_trsm_blocks*blockDim*r_blk,
                blockDim*my_num_blocks_dim,
                blockDim*r_blk,
                advance_A+(start_U_big_trsm_block+1)*blockDim*blockDim
                          *r_blk*my_num_blocks_dim,
                buffer+num_L_big_trsm_blocks*blockDim*r_blk
                            *blockDim*r_blk*(LU_mask&0x1));
      }
      RANK_PRINTF(myRank,myRank,"[i=%d] sending %d %d\n",
            i, num_L_big_trsm_blocks, num_U_big_trsm_blocks);
      MPI_Bcast(buffer, 
            ((LU_mask&0x1)*num_L_big_trsm_blocks+
             ((LU_mask&0x2)>>1)*num_U_big_trsm_blocks)
              *blockDim*r_blk*blockDim*r_blk,
            MPI_DOUBLE,
            i,
            cdt_kdir.cm);
      if (p->pvt && LU_mask&0x1 && num_L_big_trsm_blocks > 0){
        MPI_Bcast(pvt_acc + (i_big+start_L_big_trsm_block)*r_blk*blockDim, 
              num_L_big_trsm_blocks*r_blk*blockDim, 
              MPI_INT, i, cdt_kdir.cm); 
      }
    } else {
      recv_p.layerRank = i;
      assign_trsm_blocks(&recv_p, &recv_s);
      if (recv_s.num_big_trsm_blocks > 0){
        MPI_Bcast(buffer, 
              ((LU_mask&0x1)*recv_s.num_L_big_trsm_blocks+
               ((LU_mask&0x2)>>1)*recv_s.num_U_big_trsm_blocks)
              *blockDim*r_blk*blockDim*r_blk,
              MPI_DOUBLE,
              i,
              cdt_kdir.cm);

      }
      if ((LU_mask&0x1) && recv_s.num_L_big_trsm_blocks > 0) {
        lda_cpy(recv_s.num_L_big_trsm_blocks*blockDim*r_blk,
                blockDim*r_blk,
                recv_s.num_L_big_trsm_blocks*blockDim*r_blk,
                blockDim*my_num_blocks_dim,
                buffer,
                advance_A
                  +recv_s.start_L_big_trsm_block*blockDim*r_blk);
      }
      if (((LU_mask&0x2)>>1) && recv_s.num_U_big_trsm_blocks > 0) {
        lda_cpy(blockDim*r_blk,
                recv_s.num_U_big_trsm_blocks*blockDim*r_blk,
                blockDim*r_blk,
                blockDim*my_num_blocks_dim,
                buffer
                  +recv_s.num_L_big_trsm_blocks*blockDim*r_blk
                            *blockDim*r_blk*(LU_mask&0x1),
                advance_A+(recv_s.start_U_big_trsm_block+1)*blockDim*blockDim
                          *r_blk*my_num_blocks_dim);
      }
      if (p->pvt && LU_mask&0x1 && recv_s.num_L_big_trsm_blocks > 0){
        MPI_Bcast(pvt_acc + (i_big+recv_s.start_L_big_trsm_block)*r_blk*blockDim, 
               recv_s.num_L_big_trsm_blocks*r_blk*blockDim, 
               MPI_INT, i, cdt_kdir.cm); 
      }
    } 
  }
  RANK_PRINTF(myRank,myRank,"did trsm block exchanges\n");
}


/* Computes the large schur complement update in 2.5D LU */
/* Update Schur Compelemnt 
 * \ U U | x x x
 * L \ U | U U U\
 * L L \ | x x x |
 * ------------- V
 * x L x | S S S
 * x L x | S S S
 * x L x | S S S
 *   \--->
 */
static 
void schur_upd(const lu_25d_pvt_params_t        *p,
               const lu_25d_state_t             *s,
               double                           *advance_A,
               double                           *buffer){ 
  
  const int r_blk                       = s->r_blk;
  const int my_num_blocks_dim           = s->my_num_blocks_dim;
  const int num_big_blocks_dim          = s->num_big_blocks_dim;
  const int i_big                       = s->i_big; 
  const int c_rep               = p->c_rep; 
  const int64_t blockDim            = p->blockDim; 
  const int num_pes_dim         = p->num_pes_dim;
  const int layerRank           = p->layerRank;
  const int myRow               = p->myRow;
  const int myCol               = p->myCol;
  const CommData cdt_row       = p->cdt_row;
  const CommData cdt_col       = p->cdt_col;
  
  int panel_idx, i;

  const int num_active_blk_dim = r_blk*(num_big_blocks_dim-i_big-1);
  const int num_panels = (num_pes_dim*(layerRank+1))/c_rep 
                          - (num_pes_dim*layerRank)/c_rep;

  /* For each thin panel multiplication my layer needs to do... */
  for (i = 0, panel_idx = (num_pes_dim*layerRank)/c_rep; 
       i < num_panels; 
       i++, panel_idx++){
    /* If its my horizontal U panel, broadcast it */
    if (panel_idx == myRow){
      lda_cpy(r_blk*blockDim,
              num_active_blk_dim*blockDim, 
              my_num_blocks_dim*blockDim,
              r_blk*blockDim,
              advance_A+blockDim*blockDim*my_num_blocks_dim*r_blk,
              buffer);
              
    } 
    MPI_Bcast(buffer,
          blockDim*blockDim*num_active_blk_dim*r_blk,
          MPI_DOUBLE,
          panel_idx, cdt_col.cm);
    lda_cpy(r_blk*blockDim,
            num_active_blk_dim*blockDim, 
            r_blk*blockDim,
            num_panels*r_blk*blockDim,
            buffer,
            buffer+ i*r_blk*blockDim + num_panels*blockDim*blockDim
                   *num_active_blk_dim*r_blk);
  }
  for (i = 0, panel_idx = (num_pes_dim*layerRank)/c_rep; 
       i < num_panels; 
       i++, panel_idx++){
    /* If its my horizontal U panel, broadcast it */
    if (panel_idx == myCol){
      lda_cpy(num_active_blk_dim*blockDim, 
              r_blk*blockDim,
              my_num_blocks_dim*blockDim,
              num_active_blk_dim*blockDim,
              advance_A+blockDim*r_blk,
              buffer+i*blockDim*blockDim*num_active_blk_dim*r_blk);
              
    } 
    MPI_Bcast(buffer+i*blockDim*blockDim*num_active_blk_dim*r_blk,
          blockDim*blockDim*num_active_blk_dim*r_blk,
          MPI_DOUBLE,
          panel_idx, cdt_row.cm);
  }
              
  PRINT_INT(num_panels);
#if defined(OFFLOAD) & defined(OFFLOAD_FAT_GEMM)
  int64_t advance_offset = s->i_big*s->r_blk*s->my_num_blocks_dim*blockDim*blockDim
                          +s->i_big*s->r_blk*blockDim;
  if (0){//p->pvt){
    TAU_FSTART(upload_transfer);
    upload_lda_cpy(num_active_blk_dim*blockDim,
                   num_active_blk_dim*blockDim,
                   my_num_blocks_dim*blockDim,
                   my_num_blocks_dim*blockDim,
                   advance_A+(my_num_blocks_dim*blockDim+1)*r_blk*blockDim,
                   (my_num_blocks_dim*blockDim+1)*r_blk*blockDim+advance_offset,
                   OFF_A);
    upload_lda_cpy(blockDim*num_active_blk_dim,
                   num_panels*blockDim*r_blk,
                   blockDim*num_active_blk_dim,
                   blockDim*num_active_blk_dim,
                   buffer,
                   0,
                   OFF_L);
    upload_lda_cpy(num_panels*blockDim*r_blk,
                   blockDim*num_active_blk_dim,
                   num_panels*blockDim*r_blk,
                   num_panels*blockDim*r_blk,
                   buffer+num_panels*blockDim*blockDim*num_active_blk_dim*r_blk,
                   0,
                   OFF_U);

    TAU_FSTOP(upload_transfer);
    TAU_FSTART(offload_GEMM);
    offload_gemm_A('N','N',
                   num_active_blk_dim*blockDim,
                   num_active_blk_dim*blockDim,
                   num_panels*blockDim*r_blk,
                   -1.0,
                   0,
                   OFF_L,
                   num_active_blk_dim*blockDim,
                   0,
                   OFF_U,
                   num_panels*blockDim*r_blk,
                   1.0,
                   (my_num_blocks_dim*blockDim+1)*r_blk*blockDim+advance_offset,
                   OFF_A,
                   my_num_blocks_dim*blockDim);
    TAU_FSTOP(offload_GEMM);

    download_lda_cpy(num_active_blk_dim*blockDim,
                     num_active_blk_dim*blockDim,
                     my_num_blocks_dim*blockDim,
                     my_num_blocks_dim*blockDim,
                     (my_num_blocks_dim*blockDim+1)*r_blk*blockDim+advance_offset,
                     advance_A+(my_num_blocks_dim*blockDim+1)*r_blk*blockDim,
                     OFF_A);
  } else {

    if (!p->pvt && i_big>0){
      TAU_FSTART(wait_for_signal);
      wait_gemm();
      TAU_FSTOP(wait_for_signal);
    }
    if (p->pvt || i_big>0){
      TAU_FSTART(download_transfer);
      download_lda_cpy(num_active_blk_dim*blockDim,
                       r_blk*blockDim,
                       my_num_blocks_dim*blockDim,
                       my_num_blocks_dim*blockDim,
                       (my_num_blocks_dim*blockDim+1)*r_blk*blockDim+advance_offset,
                       advance_A+(my_num_blocks_dim*blockDim+1)*r_blk*blockDim,
                       OFF_A);
      DEBUG_PRINTF("Downloaded A[" PRId64 "] = %lf\n",
                       (my_num_blocks_dim*blockDim+1)*r_blk*blockDim+advance_offset,
                       advance_A[(my_num_blocks_dim*blockDim+1)*r_blk*blockDim]);
      TAU_FSTOP(download_transfer);
    }
    if (num_active_blk_dim > r_blk){
      if (!p->pvt && i_big>0){
        TAU_FSTART(download_transfer);
        download_lda_cpy(r_blk*blockDim,
                         (num_active_blk_dim-r_blk)*blockDim,
                         my_num_blocks_dim*blockDim,
                         my_num_blocks_dim*blockDim,
                         (2*my_num_blocks_dim*blockDim+1)*r_blk*blockDim+advance_offset,
                         advance_A+(2*my_num_blocks_dim*blockDim+1)*r_blk*blockDim,
                         OFF_A);
        TAU_FSTOP(download_transfer);
      }
      if (p->pvt){
        TAU_FSTART(upload_transfer);
        DEBUG_PRINTF("Uploading L[0] = %lf\n",
                     buffer[r_blk*blockDim]);
        upload_lda_cpy(blockDim*num_active_blk_dim,
                       num_panels*blockDim*r_blk,
                       blockDim*num_active_blk_dim,
                       blockDim*num_active_blk_dim,
                       buffer,
                       0,
                       OFF_L);
        DEBUG_PRINTF("Uploading U[0] = %lf\n",
                       buffer[num_panels*blockDim*blockDim*(num_active_blk_dim+r_blk)*r_blk]);
        upload_lda_cpy(num_panels*blockDim*r_blk,
                       blockDim*(num_active_blk_dim-r_blk),
                       num_panels*blockDim*r_blk,
                       num_panels*blockDim*r_blk,
                       buffer+num_panels*blockDim*blockDim*(num_active_blk_dim+r_blk)*r_blk,
                       0,
                       OFF_U);
        TAU_FSTOP(upload_transfer);

        TAU_FSTART(offload_gemm);
        offload_gemm_A('N','N',
                       num_active_blk_dim*blockDim,
                       (num_active_blk_dim-r_blk)*blockDim,
                       num_panels*blockDim*r_blk,
                       -1.0,
                       0,
                       OFF_L,
                       num_active_blk_dim*blockDim,
                       0,
                       OFF_U,
                       num_panels*blockDim*r_blk,
                       1.0,
                       (2*my_num_blocks_dim*blockDim+1)*r_blk*blockDim+advance_offset,
                       OFF_A,
                       my_num_blocks_dim*blockDim);
        TAU_FSTOP(offload_gemm);

      } else {
        TAU_FSTART(upload_transfer);
        DEBUG_PRINTF("Uploading L[0] = %lf\n",
                     buffer[r_blk*blockDim]);
        upload_lda_cpy(blockDim*(num_active_blk_dim-r_blk),
                       num_panels*blockDim*r_blk,
                       blockDim*num_active_blk_dim,
                       blockDim*(num_active_blk_dim-r_blk),
                       buffer+r_blk*blockDim,
                       0,
                       OFF_L);
        DEBUG_PRINTF("Uploading U[0] = %lf\n",
                       buffer[num_panels*blockDim*blockDim*(num_active_blk_dim+r_blk)*r_blk]);
        upload_lda_cpy(num_panels*blockDim*r_blk,
                       blockDim*(num_active_blk_dim-r_blk),
                       num_panels*blockDim*r_blk,
                       num_panels*blockDim*r_blk,
                       buffer+num_panels*blockDim*blockDim*(num_active_blk_dim+r_blk)*r_blk,
                       0,
                       OFF_U);
        TAU_FSTOP(upload_transfer);

        TAU_FSTART(offload_gemm);
        offload_gemm_A('N','N',
                       (num_active_blk_dim-r_blk)*blockDim,
                       (num_active_blk_dim-r_blk)*blockDim,
                       num_panels*blockDim*r_blk,
                       -1.0,
                       0,
                       OFF_L,
                       (num_active_blk_dim-r_blk)*blockDim,
                       0,
                       OFF_U,
                       num_panels*blockDim*r_blk,
                       1.0,
                       (2*my_num_blocks_dim*blockDim+2)*r_blk*blockDim+advance_offset,
                       OFF_A,
                       my_num_blocks_dim*blockDim);
        TAU_FSTOP(offload_gemm);

        
        TAU_FSTART(gemm1);
        cdgemm('N','N',
               r_blk*blockDim,
               (num_active_blk_dim-r_blk)*blockDim,
               num_panels*blockDim*r_blk,
               -1.0,
               buffer,
               num_active_blk_dim*blockDim,
               buffer+(num_panels*r_blk*(num_active_blk_dim+r_blk))*blockDim*blockDim,
               num_panels*blockDim*r_blk,
               1.0,
               advance_A + (2*my_num_blocks_dim*blockDim+1)*r_blk*blockDim,
               blockDim*my_num_blocks_dim);
        TAU_FSTOP(gemm1);
      }
    }
    TAU_FSTART(gemm2);
    cdgemm('N','N',
           num_active_blk_dim*blockDim,
           r_blk*blockDim,
           num_panels*blockDim*r_blk,
           -1.0,
           buffer,
           num_active_blk_dim*blockDim,
           buffer+num_panels*blockDim*blockDim
                            *num_active_blk_dim*r_blk,
           num_panels*blockDim*r_blk,
           1.0,
           advance_A + (my_num_blocks_dim*blockDim+1)*r_blk*blockDim,
           blockDim*my_num_blocks_dim);
    TAU_FSTOP(gemm2);
  }
#else
  cdgemm('N','N',
         num_active_blk_dim*blockDim,
         num_active_blk_dim*blockDim,
         num_panels*blockDim*r_blk,
         -1.0,
         buffer,
         num_active_blk_dim*blockDim,
         buffer+num_panels*blockDim*blockDim
                          *num_active_blk_dim*r_blk,
         num_panels*blockDim*r_blk,
         1.0,
         advance_A + (my_num_blocks_dim*blockDim+1)*r_blk*blockDim,
         blockDim*my_num_blocks_dim);
#endif
}

/* Reduces next schur complement panel */
/* Now reduce the next panels. The partition we are reducing looks like
 * -------------------------
 * | x x x   x x x   x x x | 
 * | x x x   x x x   x x x | 
 * | x x x   x x x   x x x | 
 * |                       |
 * | x x x                 |
 * | x x x                 |
 * | x x x                 |
 * |                       |
 * | x x x                 |
 * | x x x                 |
 * | x x x                 |
 * -------------------------
 */
static 
void red_panel(const lu_25d_pvt_params_t        *p,
               const lu_25d_state_t             *s,
               const int                        LU_mask, /* 1 -> L, 2-> U */
               double                           *advance_A,
               double                           *buffer){ 
  
  const int r_blk                       = s->r_blk;
  const int my_num_blocks_dim           = s->my_num_blocks_dim;
  const int num_big_blocks_dim          = s->num_big_blocks_dim;
  const int i_big                       = s->i_big; 
#ifdef DEBUG
  const int myRank              = p->myRank;
#endif
  const int64_t blockDim            = p->blockDim; 
  const CommData cdt_kdir      = p->cdt_kdir;
 
  const int num_active_blk_dim = r_blk*(num_big_blocks_dim-i_big-1);
  /* copies panel of L into contig buffer */
  if (LU_mask&0x1){
    lda_cpy(num_active_blk_dim*blockDim,
            blockDim*r_blk,
            blockDim*my_num_blocks_dim,
            num_active_blk_dim*blockDim,
            advance_A + (my_num_blocks_dim*blockDim+1)*r_blk*blockDim,
            buffer);
  }
  /* copies panel of U into contig buffer */
  if (((LU_mask&0x2)>>1) && num_big_blocks_dim > i_big + 2){
    lda_cpy(blockDim*r_blk,
            (num_active_blk_dim-r_blk)*blockDim,
            blockDim*my_num_blocks_dim,
            blockDim*r_blk,
            advance_A 
              + (2*my_num_blocks_dim*blockDim+1)*r_blk*blockDim,
            buffer+(LU_mask&0x1)*num_active_blk_dim*blockDim*blockDim*r_blk);
  } 
  /* sums all contributions to full panel */
  DEBUG_PRINTF("buffer[0] = %lf\n",buffer[0]);
  MPI_Allreduce(buffer, 
            buffer+((LU_mask&0x1)*num_active_blk_dim + 
                    ((LU_mask&0x2)>>1)*(num_active_blk_dim-r_blk))
                    *blockDim*blockDim*r_blk,
            ((LU_mask&0x1)*num_active_blk_dim + 
             ((LU_mask&0x2)>>1)*(num_active_blk_dim-r_blk))
                    *blockDim*blockDim*r_blk,
            MPI_DOUBLE, MPI_SUM, cdt_kdir.cm);
  /* redistributes the full L and U panels to appropriate locations in matrix */
  if (LU_mask&0x1){
    lda_cpy(num_active_blk_dim*blockDim,
            blockDim*r_blk,
            num_active_blk_dim*blockDim,
            blockDim*my_num_blocks_dim,
            buffer+(num_active_blk_dim + 
                    ((LU_mask&0x2)>>1)*(num_active_blk_dim-r_blk))
                    *blockDim*blockDim*r_blk,
            advance_A + (my_num_blocks_dim*blockDim+1)*r_blk*blockDim);
  }
  if (((LU_mask&0x2)>>1) && num_big_blocks_dim > i_big + 2){
    lda_cpy(blockDim*r_blk,
            (num_active_blk_dim-r_blk)*blockDim,
            blockDim*r_blk,
            blockDim*my_num_blocks_dim,
            buffer+(2*(LU_mask&0x1)*num_active_blk_dim + 
                    (num_active_blk_dim-r_blk))
                    *blockDim*blockDim*r_blk,
            advance_A 
              + (2*my_num_blocks_dim*blockDim+1)*r_blk*blockDim);
  } 
}       

void lu_25d_pvt(lu_25d_pvt_params_t             *p, 
                double                          *mat_A,
                int                             *mat_pvt,
                int                             *pvt_buffer,
                double                          *buffer,
                int                             is_alloced){
  
  const int c_rep               = p->c_rep;
  const int64_t matrixDim           = p->matrixDim;
  const int64_t blockDim            = p->blockDim;
  const int64_t big_blockDim        = p->big_blockDim;
  const int num_pes_dim         = p->num_pes_dim;
  const int layerRank           = p->layerRank;
  const CommData cdt_kdir      = p->cdt_kdir;
#ifdef DEBUG
  const int myRank              = p->myRank;
#endif

  lu_25d_state_t s; 
  
  double * buf_update_L, * buf_update_U;

  double * advance_A;

  int i,j;
  int * pvt_buffer_adj, * pvt_acc;
  
  const int num_blocks_dim              = matrixDim/blockDim;
  s.my_num_blocks_dim                   = num_blocks_dim/num_pes_dim;
  s.num_big_blocks_dim                  = matrixDim/big_blockDim;
  const int num_small_in_big_blk        = big_blockDim/blockDim;
  s.r_blk                               = num_small_in_big_blk/num_pes_dim;

  /* FIXME */
  const int pvt = p->pvt;

#ifdef OFFLOAD
#ifdef USE_MIC
  start_signal = 0;
  set_mic_rank(p->myRank%(MIC_PER_NODE));
#endif
  const int num_panels = (num_pes_dim*(layerRank+1))/c_rep 
                          - (num_pes_dim*layerRank)/c_rep;
//  if (!is_alloced){
    TAU_FSTART(alloc_offload);
    TAU_FSTART(alloc_A);
    alloc_A(((int64_t)s.my_num_blocks_dim*blockDim)*s.my_num_blocks_dim*blockDim, mat_A);
    TAU_FSTOP(alloc_A);
#ifdef OFFLOAD_FAT_GEMM
    TAU_FSTART(alloc_transfer);
    alloc_transfer(num_panels*blockDim*blockDim*s.my_num_blocks_dim*s.r_blk);
    TAU_FSTOP(alloc_transfer);
    alloc_L(num_panels*blockDim*blockDim*s.my_num_blocks_dim*s.r_blk);
    alloc_U(num_panels*blockDim*blockDim*s.my_num_blocks_dim*s.r_blk);
#endif
#ifdef OFFLOAD_SKINNY_GEMM
    TAU_FSTART(alloc_transfer);
    alloc_transfer(blockDim*blockDim*s.my_num_blocks_dim);
    TAU_FSTOP(alloc_transfer);
    alloc_L(blockDim*blockDim*s.my_num_blocks_dim);
    alloc_U(blockDim*blockDim*s.my_num_blocks_dim);
#endif
    TAU_FSTOP(alloc_offload);
//  }
  DEBUG_PRINTF("Uploading offload matrix\n");
#ifdef DEBUG
  //print_matrix(mat_A, blockDim*s.my_num_blocks_dim,blockDim*s.my_num_blocks_dim);
#endif
  TAU_FSTART(copy_A_to_MIC);
#ifndef MIC
  double * offload_A = get_mat_handle(OFF_A);
  memcpy(offload_A, mat_A, 
         s.my_num_blocks_dim*blockDim*s.my_num_blocks_dim*blockDim*sizeof(double));
#endif
  TAU_FSTOP(copy_A_to_MIC);

#endif
  DEBUG_PRINTF("Done with allocations\n");


  if (pvt){
    for (i=0; i<s.my_num_blocks_dim; i++){
      for (j=0; j<blockDim; j++){
        mat_pvt[i*blockDim+j] = i*blockDim*num_pes_dim + p->myRow*blockDim + j;
      }
    }
    if (s.num_big_blocks_dim > 1){
      pvt_acc = pvt_buffer+blockDim*s.my_num_blocks_dim;
      pvt_buffer_adj = pvt_acc + blockDim*s.my_num_blocks_dim;
    }
    else {
      pvt_acc = mat_pvt;
      pvt_buffer_adj = pvt_buffer + blockDim*s.my_num_blocks_dim;
    }
  }

  TAU_FSTART(LU);
  /* For each big block we do the following
   * 1-overlap with-2. Factorize top left corner big-block
   * 2-overlap with-1. Update big-block panels
   * 3. Update the Schur complement
   * 4. Reduce whatever we need for the next step
   */ 
  for (s.i_big=0; s.i_big < s.num_big_blocks_dim; s.i_big++){
    RANK_PRINTF(myRank,0, "Working on big block %d\n", s.i_big);
    /* Offset A by the number of big blocks we've already factorized */
    advance_A = mat_A+s.i_big*s.r_blk*s.my_num_blocks_dim*blockDim*blockDim
                     +s.i_big*s.r_blk*blockDim;
    
    /* dont need to do this for s.i_big > 0 since we are using all reduce */
    if (c_rep > 1 && s.i_big == 0)
    {
      if (layerRank == 0){
        lda_cpy(blockDim*s.r_blk*(s.num_big_blocks_dim-s.i_big),
                blockDim*s.r_blk,
                s.my_num_blocks_dim*blockDim,
                blockDim*s.r_blk*(s.num_big_blocks_dim-s.i_big),
                advance_A, buffer);
        if (!p->pvt){   
          lda_cpy(blockDim*s.r_blk,
                  blockDim*s.r_blk*(s.num_big_blocks_dim-s.i_big-1),
                  s.my_num_blocks_dim*blockDim,
                  blockDim*s.r_blk,
                  advance_A+s.my_num_blocks_dim*blockDim*blockDim*s.r_blk, 
                  buffer+blockDim*blockDim*(s.num_big_blocks_dim-s.i_big)
                                                          *s.r_blk*s.r_blk);
        }
      }
      if (p->pvt){      
        MPI_Bcast(buffer, 
              blockDim*blockDim*s.r_blk*(s.num_big_blocks_dim-s.i_big)*s.r_blk, 
              MPI_DOUBLE, 0, cdt_kdir.cm);      
      } else {
        MPI_Bcast(buffer, 
              blockDim*blockDim*s.r_blk*(2*(s.num_big_blocks_dim-s.i_big)-1)*s.r_blk, 
              MPI_DOUBLE, 0, cdt_kdir.cm);      
      }
      if (layerRank > 0){
        lda_cpy(blockDim*s.r_blk*(s.num_big_blocks_dim-s.i_big),
                blockDim*s.r_blk,
                blockDim*s.r_blk*(s.num_big_blocks_dim-s.i_big),
                s.my_num_blocks_dim*blockDim,
                buffer, advance_A);
        if (!p->pvt){   
          lda_cpy(blockDim*s.r_blk,
                  blockDim*s.r_blk*(s.num_big_blocks_dim-s.i_big-1),
                  blockDim*s.r_blk,
                  s.my_num_blocks_dim*blockDim,
                  buffer+blockDim*blockDim*(s.num_big_blocks_dim-s.i_big)
                                                            *s.r_blk*s.r_blk,
                  advance_A+s.my_num_blocks_dim*blockDim*blockDim*s.r_blk);
        }
      }
    }
    assign_trsm_blocks(p, &s);

    RANK_PRINTF(myRank,myRank,"total big blocks: %d\n", s.my_num_big_trsm_blocks);
    RANK_PRINTF(myRank,myRank,"L blocks %d to %d, U blocks %d to %d s.i_big=%d\n",
                s.start_L_big_trsm_block,
                s.start_L_big_trsm_block+s.num_L_big_trsm_blocks, 
                s.start_U_big_trsm_block,
                s.start_U_big_trsm_block+s.num_U_big_trsm_blocks,s.i_big);
    /* Now we iterate over the small blocks inside the big blocks and factorize
       with forward looking updates */
    if (pvt){
      TAU_FSTART(MAIN_LU_LOOP);
      TAU_FSTART(set_id_acc);
      if (s.num_big_blocks_dim > 1){
        for (i=s.i_big*s.r_blk; i<s.my_num_blocks_dim; i++){
          for (j=0; j<blockDim; j++){
            pvt_acc[i*blockDim+j] = i*blockDim*num_pes_dim + p->myRow*blockDim + j;
          }
        }
      }
      TAU_FSTOP(set_id_acc);
      for (s.i_sm=0; s.i_sm < num_small_in_big_blk; s.i_sm++){
        TAU_FSTART(set_id_buf);
        for (i=s.i_big*s.r_blk; i<s.my_num_blocks_dim; i++){
          for (j=0; j<blockDim; j++){
            pvt_buffer[i*blockDim+j] = i*blockDim*num_pes_dim + p->myRow*blockDim + j;
          }
        }
        TAU_FSTOP(set_id_buf);
        RANK_PRINTF(myRank,0,"Working on small block %d\n", s.i_sm);
        TAU_FSTART(pivot_step);
        pivot_step(p , &s, 
                   pvt_buffer,
                   pvt_acc, 
                   pvt_buffer_adj+blockDim, 
                   pvt_buffer_adj,
                   mat_A+s.i_big*s.r_blk*s.my_num_blocks_dim*blockDim*blockDim, 
                   buffer);
        TAU_FSTOP(pivot_step);
        TAU_FSTART(panel_trsm);
        panel_trsm(p,
                   &s,
                   1,
                   pvt_buffer,
                   pvt_acc,
                   pvt_buffer_adj,
                   buf_update_L,
                   buf_update_U,
                   advance_A,
                   mat_A,
                   buffer);
        TAU_FSTOP(panel_trsm);
        
        TAU_FSTART(L_panel_updt);
        panel_updt(p, &s, 1, buf_update_L, buf_update_U, advance_A);
        TAU_FSTOP(L_panel_updt);
      }
      if (c_rep > 1){
        TAU_FSTART(L_xchng_blks);
        xchng_blks(p, &s, 1, advance_A, buffer, pvt_acc);
        TAU_FSTOP(L_xchng_blks);
      }

      if (s.num_big_blocks_dim > 1){
        TAU_FSTART(pvt_collc);
        pvt_collc(mat_A+(s.i_big+(s.i_big+1)*s.my_num_blocks_dim*blockDim)
                        *blockDim*s.r_blk,
                  mat_A+s.i_big*blockDim*s.r_blk,
                  buffer,
                  blockDim,
                  blockDim*(s.my_num_blocks_dim-s.i_big*s.r_blk),
                  blockDim*(s.my_num_blocks_dim-(s.i_big+1)*s.r_blk),
                  blockDim*s.i_big*s.r_blk,
                  blockDim*s.my_num_blocks_dim,
                  s.i_big*blockDim*s.r_blk,
                  s.i_big*p->big_blockDim,
                  mat_pvt+s.i_big*blockDim*s.r_blk,
                  pvt_acc+s.i_big*blockDim*s.r_blk,
                  pvt_buffer_adj,
                  p->myRow,
                  p->num_pes_dim,
                  p->cdt_col);
        TAU_FSTOP(pvt_collc);
      }
#ifdef OFFLOAD_FAT_GEMM
      if (p->pvt){
#ifdef ASYNC_GEMM
        if (start_signal){
          TAU_FSTART(wait_for_signal);
          wait_gemm();
          TAU_FSTOP(wait_for_signal);
        }
#endif

        TAU_FSTART(download_pivoted_U);
        int num_active_blk_dim = s.r_blk*(s.num_big_blocks_dim-s.i_big);
        int64_t advance_offset = (s.i_big-1)*s.r_blk*s.my_num_blocks_dim*blockDim*blockDim
                                +(s.i_big-1)*s.r_blk*blockDim;
        download_lda_cpy(s.r_blk*blockDim,
                         (num_active_blk_dim-s.r_blk)*blockDim,
                         s.my_num_blocks_dim*blockDim,
                         s.my_num_blocks_dim*blockDim,
                         (2*s.my_num_blocks_dim*blockDim+1)*s.r_blk*blockDim+advance_offset,
                         advance_A+s.my_num_blocks_dim*blockDim*s.r_blk*blockDim,
                         OFF_A);
        TAU_FSTOP(download_pivoted_U);
      }
#endif
      if (c_rep > 1 && s.i_big < s.num_big_blocks_dim-1){
        TAU_FSTART(reduce_U);
        s.i_big--;
        advance_A = mat_A+s.i_big*s.r_blk*s.my_num_blocks_dim*blockDim*blockDim
                         +s.i_big*s.r_blk*blockDim;
        red_panel(p, &s, 2, advance_A, buffer);
        s.i_big++;
        advance_A = mat_A+s.i_big*s.r_blk*s.my_num_blocks_dim*blockDim*blockDim
                         +s.i_big*s.r_blk*blockDim;
        TAU_FSTOP(reduce_U);
      }
      for (s.i_sm=0; s.i_sm < num_small_in_big_blk; s.i_sm++){
        RANK_PRINTF(myRank,0,"Working on small block %d\n", s.i_sm);
    
        TAU_FSTART(panel_U_trsm);
        panel_trsm(p,
                   &s,
                   2,
                   pvt_buffer,
                   mat_pvt,
                   pvt_buffer_adj,
                   buf_update_L,
                   buf_update_U,
                   advance_A,
                   mat_A,
                   buffer);
        TAU_FSTOP(panel_U_trsm);
        
        TAU_FSTART(panel_U_updt);
        panel_updt(p, &s, 2, buf_update_L, buf_update_U, advance_A);
        TAU_FSTOP(panel_U_updt);
      }
  
      if (c_rep > 1){
        TAU_FSTART(xchng_blocks_U);
        xchng_blks(p, &s, 2, advance_A, buffer, NULL);
        TAU_FSTOP(xchng_blocks_U);
      }
      if (s.i_big < s.num_big_blocks_dim-1){
        RANK_PRINTF(myRank,0,"Updating schur complement\n");
        TAU_FSTART(schur_upd);
        schur_upd(p, &s, advance_A, buffer);
        TAU_FSTOP(schur_upd);
      
        RANK_PRINTF(myRank,0,"Reducing next big anels of Schur complement\n");
        TAU_FSTART(red_panel_L);
        red_panel(p, &s, 1, advance_A, buffer);
        TAU_FSTOP(red_panel_L);
      }
      TAU_FSTOP(MAIN_LU_LOOP);
    } else {
      for (s.i_sm=0; s.i_sm < num_small_in_big_blk; s.i_sm++){
        RANK_PRINTF(myRank,0,"Working on small block %d\n", s.i_sm);
        TAU_FSTART(panel_trsm);    
        panel_trsm(p,
                   &s,
                   3,
                   pvt_buffer,
                   mat_pvt,
                   pvt_buffer,
                   buf_update_L,
                   buf_update_U,
                   advance_A,
                   mat_A,
                   buffer);
        TAU_FSTOP(panel_trsm);    
        
        TAU_FSTART(panel_updt);    
        panel_updt(p, &s, 3, buf_update_L, buf_update_U, advance_A);
        TAU_FSTOP(panel_updt);    
      }
      if (c_rep > 1){
        TAU_FSTART(xchng_blks);    
        xchng_blks(p, &s, 3, advance_A, buffer, NULL);
        TAU_FSTOP(xchng_blks);    
      }
      if (s.i_big < s.num_big_blocks_dim-1){
        RANK_PRINTF(myRank,0,"Updating schur complement\n");
        TAU_FSTART(schur_upd);    
        schur_upd(p, &s, advance_A, buffer);
        TAU_FSTOP(schur_upd);    
      
        RANK_PRINTF(myRank,0,"Reducing next big panels of Schur complement\n");
        TAU_FSTART(red_panel);    
        red_panel(p, &s, 3, advance_A, buffer);
        TAU_FSTOP(red_panel);    
      }
    }
  }
  TAU_FSTOP(LU);
#ifdef OFFLOAD
  free_offload_A();//((int64_t)s.my_num_blocks_dim*blockDim)*s.my_num_blocks_dim*blockDim);
  free_offload_transfer();//((int64_t)s.my_num_blocks_dim*blockDim)*s.my_num_blocks_dim*blockDim);
  free_offload_L();//num_panels*blockDim*blockDim*s.my_num_blocks_dim*s.r_blk);
  free_offload_U();//num_panels*blockDim*blockDim*s.my_num_blocks_dim*s.r_blk);
#endif
} /* end function lu_25d_pvt */


