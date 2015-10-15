#ifndef __TNMT_PVT_H__
#define __TNMT_PVT_H__

#include "../shared/comm.h"
//#define PARTIAL_PVT

//#define USE_GATHER 1

#ifdef USE_GATHER
#if USE_GATHER
#define USE_GATHER_SB 1
#define USE_GATHER_BB 1
#else
#define USE_GATHER_SB 0
#define USE_GATHER_BB 0
#endif
#endif


#ifndef USE_GATHER_SB
#define USE_GATHER_SB 0
#endif

#ifndef USE_GATHER_BB
#define USE_GATHER_BB 1
#endif

#ifndef BGP
extern "C"
void dlaswp_(const int *n,      double *A,
             const int *lda,    const int *k1,
             const int *k2,     const int *IPIV,
             const int *inc);
#endif

void cdlaswp(const int n,       double *A,
             const int lda,     const int k1,
             const int k2,      const int *IPIV,
             const int inc);


void pivot_conv_direct(const int n, const int *P1, int *P2);
/* Performs swaps according to IPIV on pivot matrix P */
void pivot_conv(const int n, const int *IPIV, int *P);

/* Pivots matrix A according to pivot matrix P, puts result into B */
void pivot_mat(const int nrow, const int ncol, const int *P, const double *A, double *B);

/* Forms pivot matrix P from 'best rows' P_br */
void br_to_pivot(const int nbr, const int lda, const int *P_br, int *P);

/* Inverts 'best rows' P_br to figure out where out-pivoted rows of A should go */
void inv_br(const int nbr, 
            const int off, 
            const int *P_br, 
            int *     P,
            int *     P_loc=NULL);


template <int row_major>                    /* is row major ? */
void local_tournament(double *A,        /* n by b matrix */
                      double *R,        /* input:  n by b buffer of opaque memory 
                                           output: b by b best rows; n-b by b buffer of opaque memory*/
                      int *P,           /* output: b length pivot array */
                      int n,
                      int b,
                      int lda_A);

/* Perform tournament pivoting over a ring of processors */
void tnmt_pvt_1d(double         *R,     /* input:  b by b matrix of my best rows */
                 double         *R_out, /* out: b by b  matrix of best rows */
                 int            *P_in,  /* in:  b by 1 global ranks of my best rows */
                 double         *R_buf, /* out: 2b by b buffer for rows  */
                 int            *P_out, /* 3b by 1 buffer for local best rank calc 
                                           out: b by 1 (2b by 1 buffer) global 
                                                ranks of best rows in communicator */
                 const int      b,
                 const int      myRank,
                 const int      pe_start,
                 const int      numPes,
                 const int      root,
                 CommData       cdt);


/* Apply a distributed pivot matrix over a column of processors */
void par_pivot(double           *A,     /* input: matrix  */
               double           *buffer,/* buffer space */
               const int        npiv,   /* number of rows to pivot */
               const int        ncol,   /* number of columns */
               const int        b,      /* dimension of small block */
               const int        lda_A,  /* lda of A */
               const int        idx_off,/* local offset of A and P_* from top */
               const int        glb_off,/* global offset from top */
               int              *P_r,   /* old->new row source indices */
               int              *P_app, /* permutation matrix to apply */
                                        /* owned by root only */
                                        /* 3*npiv length */
               const int        myRank, /* rank in processor oclumn comm */
               const int        lrRank, /* rank in processor oclumn comm */
               const int        numPes, /* num pes in column */
               const int        numLrs, /* num of layers */
               const int        root,   /* rank in processor oclumn comm */
               const int        st_blk, /* first block this layer owns */       
               const int        num_blk,/* number of blocks this layer owns */
               const CommData   cdt_col,  /* column communiccator */
               const CommData   cdt_col_kdir,  /* kdir communiccator */
               int *            pvt_buffer=NULL,
               int const        is_tnmt_pvt=1,
               int const        nloaded=0);//rows already uploaded (only used in skinyy offload  mode

/* Collect rows of U */
void pvt_collc(double           *A_fw,  /* input: next U offset by idx_off*/
               double           *A_bw,  /* input: matrix A offset by idx_off */
               double           *buffer,/* buffer space */
               const int        b,      /* block size */
               const int        nrow,   /* number of rows involved */
               const int        ncol_fw,/* number of columns after current i_big */
               const int        ncol_bw,/* number of columns before current i_big */
               const int        lda_A,  /* lda of A */
               const int        idx_off,/* local offset of A and P_* from top */
               const int        glb_off,/* global offset from top */
               int              *P_r,   /* old->new row source indices */
               int              *P_mine,/* The rows I need */
               int              *P_buf, /* buffer space */
               const int        myRank, /* rank in processor oclumn comm */
               const int        numPes, /* num pes in column */
               const CommData   cdt);  /* column communiccator */


#define local_tournament_row_maj(A,R,P,n,b,lda_A)       \
  do {                                                  \
    local_tournament<1>(A,R,P,n,b,lda_A);               \
  } while (0)

#define local_tournament_col_maj(A,R,P,n,b,lda_A)       \
  do {                                                  \
    local_tournament<0>(A,R,P,n,b,lda_A);               \
  } while (0)


#endif
