#ifndef __LU_25D_PVT_H__
#define __LU_25D_PVT_H__

#include "../shared/comm.h"
#include "../shared/util.h"

//#define SHARE_MIC 2

typedef struct lu_25d_pvt_params {
  int   pvt;  /* 1-> do pivoting, 0-> no pivoting */
  int   is_tnmt_pvt;
  int   myRank; 
  int   c_rep;
  int   matrixDim;
  int   blockDim;
  int   big_blockDim;
  int   num_pes_dim;
  int   layerRank;
  int   myRow;
  int   myCol;
  CommData  cdt_row;
  CommData  cdt_col;
  CommData  cdt_kdir;
  CommData  cdt_kcol;
} lu_25d_pvt_params_t;


void lu_25d_pvt(lu_25d_pvt_params_t   *p, 
                double                *mat_A,
                int                   *mat_pvt,
                int                   *pvt_buffer,
                double                *buffer,
                int                   is_alloced=0);

#endif //__LU_25D_PVT_H__
