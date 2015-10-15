#ifndef __CANDMC_H__
#define __CANDMC_H__

/* SUMMA and Cannon */
#include "../alg/MM/topo_pdgemm/topo_pdgemm_algs.h"

/* Split-dimensional Cannon's algorithm for multidimensional torus networks */
#include "../alg/MM/splitdim_cannon/spcannon.h"

/* tournament (CA) pivoting for panels of  matrices */
#include "../alg/LU/tnmt_pvt.h"

/* partial pivoting for panels of matrices */
#include "../alg/LU/partial_pvt.h"

/* 2.5D LU algorithms for full matrices */
#include "../alg/LU/lu_25d_pvt.h"

/* binary tree TSQR on panels of matrices */
#include "../alg/QR/tsqr/bitree_tsqr.h"

/* butterfly tree TSQR on panels of matrices */
#include "../alg/QR/tsqr/butterfly_tsqr.h"

/* TSQR with Householder vector reconstruction */
#include "../alg/QR/hh_recon/hh_recon.h"

/* 2D QR algorithms for full matrices */
#include "../alg/QR/qr_2d/qr_2d.h"

/* 2D QR with Yamamoto's basis-kernel representation for full matrices */
#include "../alg/QR/qr_2d/qr_y2d.h"

/* 2D QR algorithms for full matrices */
#include "../alg/SE/CANSE.h"

#endif
