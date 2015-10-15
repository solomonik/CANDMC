#include <mpi.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../shared/util.h"
#include "../shared/comm.h"
//#include "../shared/seq_lu.h"
#include "tnmt_pvt.h"
#include <algorithm>
#ifdef OFFLOAD
#include "lu_offload.h"
#endif


void cdlaswp(const int n,       double *A,
             const int lda,     const int k1,
             const int k2,      const int *IPIV,
             const int inc){
#if defined(BGP) || defined(BGQ)
  int i,j;
  double swp;
  for (i=k1-1; i<k2; i++){
    for (j=0; j<n; j++){
      swp                = A[j*lda+IPIV[i]-1];
      A[j*lda+IPIV[i]-1] = A[j*lda+i];
      A[j*lda+i]         = swp;
    }
  }
#else
  dlaswp_(&n,A,&lda,&k1,&k2,IPIV,&inc);
#endif
}

void pivot_conv_direct(const int n, const int *P1, int *P2){
  int i;
  int * P2_cpy = (int*)malloc(sizeof(int)*n);
  memcpy(P2_cpy, P2, sizeof(int)*n);
  for (i=0; i<n; i++){
    P2[i] = P2_cpy[P1[i]];
  }
  free(P2_cpy);
}

void pivot_conv(const int n, const int *P1, int *P2){
  int tmp,i;
  for (i=0; i<n; i++){
    tmp = P2[P1[i]-1];
    P2[P1[i]-1] = P2[i];
    P2[i] = tmp;
  }
}

void pivot_mat(const int nrow, const int ncol, const int *P, const double *A, double *B){
  int row,col;
  for (col=0; col<ncol; col++){
    for (row=0; row<nrow; row++){
      B[col*nrow +row] = A[col*nrow + P[row]];
    }
  }
}


/* Forms pivot matrix P from 'best rows' P_br */
void br_to_pivot(const int nbr, const int lda, const int *P_br, int *P){
  int i,j;
  memset(P,0,nbr*sizeof(int));
  for (i=0; i<nbr; i++){
    if (P_br[i] < nbr) P[P_br[i]] = -1;
  }
  for (i=nbr; i<lda; i++){
    P[i] = i;
  }
  i=0,j=0;
  while (i < nbr && j < nbr){
    while (i<nbr && P[i] == -1) i++; 
    while (j<nbr && P_br[j] < nbr) j++;
    assert((i<nbr && j<nbr) || (i==nbr && j==nbr));
    if (i < nbr){       
      P[P_br[j]] = i;
      i++,j++;
    }
  }
  memcpy(P,P_br,nbr*sizeof(int));    
}

/* Inverts 'best rows' P_br to figure out where out-pivoted rows of A should go */
void inv_br(const int nbr, 
            const int off, 
            const int *P_br, 
            int *     P,
            int *     P_loc){
  int i,j;
  std::fill(P,P+nbr,-1);
  for (i=0; i<nbr; i++){
    if (P_br[i] < nbr+off) {
      P[P_br[i]-off] = P_br[i];
      if (P_loc != NULL) 
        P_loc[i] = P_br[i]-off;
    } else {
      if (P_loc != NULL) P_loc[i] = i;
    }
    assert(P_br[i] >= off);
  }
  i=0,j=0;
  while (i < nbr && j < nbr){
    while (i<nbr && P[i] != -1) i++; 
    while (j<nbr && P_br[j] < nbr+off) j++;
    assert((i<nbr && j<nbr) || (i==nbr && j==nbr));
    if (i < nbr){       
      P[i] = P_br[j];
      i++,j++;
    }
  }
}

//static 
void pack_bwd(double *A, int const nrow, int const lda, int const ncol){
  int i;
  for (i=1; i<ncol; i++){
    memcpy(A+nrow*i,A+lda*i,nrow*sizeof(double));
  }
}


/* Perform tournament pivoting sequentially */
template <int row_major>                    /* is row major ? */
void local_tournament(double *A, /* n by b matrix */
                      double *R, /* input:  lda_A by b buffer of opaque memory 
                                    output: b by b best rows; 
                                            n-b by b buffer of opaque memory*/
                      int *P,    /* output: b length pivot array */
                      int n,
                      int b,
                      int lda_A){
  
  int info,row,i;

  lda_cpy(n,b,lda_A,n,A,R);
  cdgetrf(n,b,R,n,P,&info);

  lda_cpy(n,b,lda_A,n,A,R);
  if (row_major){
    assert(0);
    for (i=0; i<b; i++){
      memcpy(R+row*b,R+P[row]*b,b*sizeof(double));
    }
  } else {
    cdlaswp(b,R,n,1,b,P,1);
    if (n>b)
      pack_bwd(R,b,n,b);
    /*for (col=0; col<b; col++){
      for (row=0; row<b; row++){
        swp = R[col*b+row];
        R[col*b+row] = R[col*n + P[row]];
        R[col*n + P[row]] = swp;
      }
    }*/
  } 
}

template void local_tournament<0>(double*,double*,int*,int,int,int);
template void local_tournament<1>(double*,double*,int*,int,int,int);

/* we receive a contiguous buffer b-by-b (A2) which is the block below.
 * To get a 2b-by-b buffer, we need to combine this buffer with our original
 * block. Since we are working with column-major ordering we need to interleave
 * the blocks. Thats what this function does. If ordering=0, then A1 is top block,
 * otherwise, A2 is top block.  */
static
void coalesce_bwd(double const  *A1,    
                  double        *A2,
                  double const  ordering,
                  int const     b){
  int i;
  if (ordering) {
    for (i=b-1; i>=0; i--){
      memcpy(A2+2*i*b+b, A1+i*b, b*sizeof(double));
      if (i!=0) memcpy(A2+2*i*b, A2+i*b, b*sizeof(double));
    }
  } else {
    for (i=b-1; i>=0; i--){
      memcpy(A2+2*i*b+b, A2+i*b, b*sizeof(double));
      memcpy(A2+2*i*b,   A1+i*b, b*sizeof(double));
    }
  }
}


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
                 CommData       cdt){
  int comm_pe, np;
  int req_id = 0 ;
  MPI_Request req;

  int myr;
  if (myRank < pe_start)
    myr=-1;
  else {
    myr = myRank-root;
    if (myr < 0) myr+=numPes;
  }

  double * R_curr;
  if (myr >= 0){
    memcpy(P_out,P_in,b*sizeof(int));
    if (numPes == 1){
      memcpy(R_out,R,b*b*sizeof(double));
      DEBUG_PRINTF("WARNING: pivoting with 1 processor doesnt do shit\n");
      return;
    }
    R_curr = R;
  }
  TAU_FSTART(tnmt_pvt_1d);

  MPI_Status stat;
  /* Tournament tree is a butterfly */
  for (np = numPes; np > 1; np = np/2+(np%2)){
    /* If I am in second half of processor list, simply send my data to lower half */
    if ((myr > np/2 || myr*2 == np) && myr < np){
      comm_pe = myr-(np+1)/2;
      if (comm_pe >= numPes - root + pe_start) 
        comm_pe = comm_pe - numPes + root;
      else 
        comm_pe = comm_pe + root;
      if (np%2 == 0 || myr != np/2){
        /* FIXME: compress these two messages into one */
        MPI_Send(R_curr, b*b, MPI_DOUBLE, comm_pe, req_id, cdt.cm);
        MPI_Send(P_out, b,   MPI_INT, comm_pe, req_id, cdt.cm);
      } 
    } else if (myr < np/2 && myr >= 0) {
      comm_pe = myr+(np+1)/2;
      if (comm_pe >= numPes - root + pe_start) 
        comm_pe = comm_pe - numPes + root;
      else 
        comm_pe = comm_pe + root;

      MPI_Irecv(R_buf, b*b, MPI_DOUBLE, comm_pe, req_id, cdt.cm, &req);
      MPI_Wait(&req, &stat);
      RANK_PRINTF(myRank,myRank,"R_buf[0] = %lf\n", R_buf[0]);
      MPI_Irecv(P_out+b, b, MPI_INT, comm_pe, req_id, cdt.cm, &req);
      MPI_Wait(&req, &stat);
      coalesce_bwd(R_curr,R_buf,0,b);

      local_tournament_col_maj(R_buf,R_out,P_out+2*b,2*b,b,2*b);
      //pack_bwd(R_out,b,2*b,b);
      pivot_conv(b,P_out+2*b,P_out);

      R_curr = R_out;
    }
    //FIXME: add another tag
    MPI_Barrier(cdt.cm);
  }   
  TAU_FSTOP(tnmt_pvt_1d);
}


#if USE_GATHER_SB
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
               const CommData_t cdt_col,  /* column communiccator */
               const CommData_t cdt_kdir,  /* kdir communiccator */
               int *            pvt_buffer,
               int const        is_tnmt_pvt){
  int row, col, row_dest, row_mod, row_blk, nrow_mine, i;
  double * buf_send, * buf_recv, * buf_back;
  int *P_send,*P_recv;
  int * recvcnts, * displs;

#ifdef OFFLOAD_SKINNY_GEMM
  assert(0);
#endif

  if (is_tnmt_pvt)
    assert(0); //not supported

  if (myRank == root){
    /* Figure out what rows to swap out to where */
  /*  RANK_PRINTF(myRank,myRank,"inverting pivot mat, off =%d P_app[0]=%d [1] = %d [2] = %d [3] = %d\n", 
                glb_off, P_app[0], P_app[1], P_app[2], P_app[3]);*/
    inv_br(npiv,glb_off,P_app,P_app+npiv);
/*    RANK_PRINTF(myRank,myRank,"inverting pivot mat, off =%d P_app[0]=%d [1] = %d [2] = %d [3] = %d\n", 
                glb_off, P_app[4], P_app[5], P_app[6], P_app[7]);*/
    /* Copy my rows and their indices to a buffer */
    memcpy(P_app+npiv*2,P_r,npiv*sizeof(int));
  }
  RANK_PRINTF(myRank,myRank,"broadcasting pivots npiv = %d ncol = %d\n",npiv,ncol);
  /* Broadcast the pivot matrix to everyone */
  MPI_Bcast(P_app+npiv,2*npiv,MPI_INT,root,cdt_col.cm);
  RANK_PRINTF(myRank,myRank,"broadcasted pivots\n");

  P_send = P_app+4*npiv; 
  P_recv = P_send+2*npiv; 
  recvcnts = P_recv+2*npiv;
  displs = recvcnts+numPes;
  buf_send = buffer;
  buf_recv = buf_send + npiv*ncol;
  buf_back = buf_recv + npiv*ncol;
 
  nrow_mine = 0;
  for (row=0; row<npiv; row++){
    row_dest = P_app[npiv+row];
    row_mod = row_dest%(b*numPes);
    row_blk = row_dest/(b*numPes);
    /* If the row belongs to me */
    if (row_mod >= myRank*b && row_mod < (myRank+1)*b && 
        row_blk >= st_blk   && row_blk < st_blk+num_blk){
      lda_cpy(1,ncol,lda_A,1,
              A+row_blk*b+row_mod-myRank*b-idx_off,
              buf_send+ncol*nrow_mine);
      P_send[2*nrow_mine] = row;
      P_send[2*nrow_mine+1] = P_r[row_blk*b+row_mod-idx_off-myRank*b];
      P_recv[nrow_mine]     = row_blk*b+row_mod-idx_off-myRank*b;
      P_r[row_blk*b+row_mod-idx_off-myRank*b] = P_app[npiv*2+row];
      nrow_mine++;
    }
  }
  nrow_mine = nrow_mine*2;
  MPI_Gather(&nrow_mine,1,MPI_INT,
         recvcnts,1,MPI_INT,root,cdt_col.cm);
  if (myRank == root){
    displs[0] = 0;
    for (i=1; i<numPes; i++){
      displs[i] = displs[i-1] + recvcnts[i-1];
    }
  }
  MPI_Gatherv(P_send,nrow_mine,MPI_INT,
          P_app,recvcnts,displs,MPI_INT,root,cdt_col.cm);
  if (myRank == root){
    displs[0] = 0;
    recvcnts[0] = ncol*recvcnts[0]/2;
    for (i=1; i<numPes; i++){
      displs[i] = displs[i-1] + recvcnts[i-1];
      recvcnts[i] = ncol*recvcnts[i]/2;
    }
  }
  MPI_Gatherv(buf_send,ncol*nrow_mine/2,MPI_DOUBLE,
          buf_recv,recvcnts,displs,MPI_DOUBLE,root,cdt_col.cm);
    
  if (myRank == root){
    for (row=0; row<npiv; row++){
      row_dest = P_app[2*row];
      lda_cpy(1,ncol,lda_A,1,A+row_dest,buf_send+row*ncol);
    }
  }
  SCATTERV(buf_send,recvcnts,displs,MPI_DOUBLE,
           buf_back,ncol*nrow_mine/2,MPI_DOUBLE,root,cdt_col);
  for (row=0; row<nrow_mine/2; row++){
    row_dest = P_recv[row];
    lda_cpy(1,ncol,1,lda_A,buf_back+row*ncol,A+row_dest);
  }
  if (numLrs > 1 && myRank == root){
    nrow_mine = ((displs[numPes-1]+recvcnts[numPes-1])/ncol)*2;
    MPI_Gather(&nrow_mine,1,MPI_INT,
           recvcnts,1,MPI_INT,0,cdt_kdir.cm);
    displs[0] = 0;
    for (i=1; i<numLrs; i++){
      displs[i] = displs[i-1] + recvcnts[i-1];
    }
    MPI_Gatherv(P_app,nrow_mine,MPI_INT,
            P_send,recvcnts,displs,MPI_INT,0,cdt_kdir.cm);
  
    displs[0] = 0;
    recvcnts[0] = ncol*recvcnts[0]/2;
    for (i=1; i<numLrs; i++){
      displs[i] = displs[i-1] + recvcnts[i-1];
      recvcnts[i] = ncol*recvcnts[i]/2;
    }
    MPI_Gatherv(buf_recv,ncol*nrow_mine/2,MPI_DOUBLE,
            buf_send,recvcnts,displs,MPI_DOUBLE,0,cdt_kdir.cm);
    if (lrRank == 0){
      for (row=0; row<npiv; row++){
        row_dest = P_send[2*row];
        P_r[row_dest] = P_send[2*row+1];
        lda_cpy(1,ncol,1,lda_A,buf_send+row*ncol,A+row_dest);
      }
    }
  } else if (myRank == root){
    for (row=0; row<npiv; row++){
      row_dest = P_app[2*row];
      P_r[row_dest] = P_app[2*row+1];
      lda_cpy(1,ncol,1,lda_A,buf_recv+row*ncol,A+row_dest);
    }
  }
}

#else

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
               const CommData   cdt_kdir,  /* kdir communiccator */
               int *            pvt_buffer,
               int const        is_tnmt_pvt,
               int const        nloaded){
  int row, col, row_dest, row_mod, row_blk, i, r, isv;
  double * buffer2 = buffer + npiv*ncol;

  if (myRank == root){
    /* Figure out what rows to swap out to where */
    RANK_PRINTF(myRank,myRank,"inverting pivot mat, off =%d P_app[0]=%d [1]=%d\n", 
                glb_off, P_app[0], P_app[1]);
    if (is_tnmt_pvt)
      inv_br(npiv,glb_off,P_app,P_app+npiv,pvt_buffer);
    else
      memcpy(P_app+npiv,P_app,npiv*sizeof(int));
    RANK_PRINTF(myRank,myRank,"inverting pivot mat, off =%d P_app[0]=%d\n", glb_off, P_app[0]);
    /* Copy my rows and their indices to a buffer */
    memcpy(P_app+npiv*2,P_r,npiv*sizeof(int));
    lda_cpy(npiv, ncol, lda_A, npiv, A, buffer);
  }
  RANK_PRINTF(myRank,myRank,"broadcasting pivots npiv = %d ncol = %d\n",npiv,ncol);
  /* Broadcast top rows and their indices to everyone */
  MPI_Bcast(P_app+npiv,2*npiv,MPI_INT,root,cdt_col.cm);
  MPI_Bcast(buffer,npiv*ncol,MPI_DOUBLE,root,cdt_col.cm);
  RANK_PRINTF(myRank,myRank,"broadcasted pivots\n");
#ifdef OFFLOAD_SKINNY_GEMM
  int wrcount = 0;
  int * rindices = (int*)malloc(sizeof(int)*npiv);
  int * bindices = (int*)malloc(sizeof(int)*npiv);
  double * wbuffer = (double*)malloc(sizeof(double)*npiv*(ncol-nloaded));
#endif
  
  for (row=0; row<npiv; row++){
    row_dest = P_app[npiv+row];
    row_mod = row_dest%(b*numPes);
    row_blk = row_dest/(b*numPes);
    /* If the row belongs to me */
    if (row_mod >= myRank*b && row_mod < (myRank+1)*b && 
        row_blk >= st_blk   && row_blk < st_blk+num_blk){
      /* Copy the row to my matrix and swap out my row into buffer */
      P_app[row] = P_r[row_blk*b+row_mod-idx_off-myRank*b];
#ifdef OFFLOAD_SKINNY_GEMM
      if (myRank == root && row_blk == st_blk){
        for (col=0; col<ncol; col++){
          buffer2[col*npiv+row] = A[col*lda_A+row_blk*b+row_mod-myRank*b-idx_off];
        }
      } else {
        rindices[wrcount] = row_blk*b+row_mod-myRank*b+nloaded*lda_A;
        bindices[wrcount] = row;
        for (col=0; col<nloaded; col++){
          buffer2[col*npiv+row] = A[col*lda_A+row_blk*b+row_mod-myRank*b-idx_off];
        }
      }
#else
      for (col=0; col<ncol; col++){
        buffer2[col*npiv+row] = A[col*lda_A+row_blk*b+row_mod-myRank*b-idx_off];
      }
#endif

      r = row;
      if (row_dest-glb_off>=npiv){
        do {
          isv = 0;
          for (i=0; i<npiv; i++) {
            if (P_app[npiv+i] == r+glb_off) {
              isv = 1;
              r = i;
              break;
            }
          }
        } while(isv);
      }
      P_r[row_blk*b+row_mod-idx_off-myRank*b] = P_app[r+npiv*2];
#ifdef OFFLOAD_SKINNY_GEMM
      if (myRank == root && row_blk == st_blk){
        for (col=0; col<ncol; col++){
          A[col*lda_A+row_blk*b+row_mod-myRank*b-idx_off] = 
            buffer[col*npiv+r];
        }
      } else {
        lda_cpy(1,ncol-nloaded,npiv,1,buffer+r+nloaded*npiv,wbuffer+wrcount*(ncol-nloaded));
        wrcount++;
        for (col=0; col<nloaded; col++){
          A[col*lda_A+row_blk*b+row_mod-myRank*b-idx_off] = 
            buffer[col*npiv+r];
        }
      }
#else
      for (col=0; col<ncol; col++){
        A[col*lda_A+row_blk*b+row_mod-myRank*b-idx_off] = 
          buffer[col*npiv+r];
      }
#endif
    }
    else {
      /* Otherwise set buffer to 0 so we can reduce successfully */
      P_app[row] = 0;
      for (col=0; col<ncol; col++){
        buffer2[col*npiv+row] = 0.0;
      }
    }
  }
#ifdef OFFLOAD_SKINNY_GEMM
  //do sparse read and write over PCI
  if (ncol > nloaded){
    offload_sparse_rw(wrcount,ncol-nloaded,lda_A,
                      wbuffer,ncol-nloaded,rindices,OFF_A,'s');
    for (i=0; i<wrcount; i++){
      lda_cpy(1,ncol-nloaded,1,npiv,
              wbuffer+(ncol-nloaded)*i,
              buffer2+bindices[i]+npiv*nloaded);
    }
  }
  free(rindices);
  free(bindices);
  free(wbuffer);
#endif
  /* Reduce swapped-in rows and their indices so the root has them */
  MPI_Reduce(buffer2,buffer,npiv*ncol,MPI_DOUBLE,MPI_SUM,root,cdt_col.cm);
  MPI_Reduce(P_app,P_app+2*npiv,npiv,MPI_INT,MPI_SUM,root,cdt_col.cm);
  /* Substitute swapped-in rows at the root */
  if (myRank == root){
    memcpy(P_r,P_app+npiv*2,npiv*sizeof(int));
    lda_cpy(npiv, ncol, npiv, lda_A, buffer, A);
  }
}
#endif


#if USE_GATHER_BB
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
               const CommData   cdt){  /* column communiccator */
  int row, row_dest, row_mod, row_blk, i, ib, nrow_mine;

  int ncol_tot = ncol_bw + ncol_fw;

  int nrow_get, nrow_recv;
  int * P_recv, * P_send, * P_tots, * recvcnts, * displs;
  double * buffer_send;

#ifdef OFFLOAD_SKINNY_GEMM
  if (myRank == 0){
    printf("Compiled with OFFLOAD_SKINNY_GEMM defined, therefore must");
    printf(" use only one level of blocking\n");
  }
  ABORT;
#endif

  nrow_get = 0;

  for (row=0; row<nrow; row++){
    ib = (row%b)+b*numPes*(row/b)+glb_off+myRank*b;
    if (P_mine[row] != ib){
      P_buf[2*nrow_get] = P_mine[row];
      P_buf[2*nrow_get+1] = row;
      nrow_get++;
    }
  }
  buffer_send = buffer+nrow_get*ncol_tot;
  P_tots = P_buf+nrow_get*2;
  recvcnts = P_tots+numPes;
  displs = recvcnts+numPes;
  P_buf[2*nrow_get+numPes] = nrow_get;
  MPI_Allgather(P_buf+2*nrow_get+numPes,1,MPI_INT,
            P_buf+2*nrow_get,1,MPI_INT,cdt.cm);

#ifdef OFFLOAD_FAT_GEMM
  int * rindices = (int*)malloc(sizeof(int)*nrow);
#endif

  for (i=0; i<numPes; i++){
    if (i==myRank)
      P_recv = P_buf;
    else
      P_recv = displs+numPes;
    nrow_recv = P_tots[i];
    if (nrow_recv > 0){
      P_send = displs+2*nrow_recv+numPes;
      MPI_Bcast(P_recv,2*nrow_recv,MPI_INT,i,cdt.cm);
      nrow_mine = 0;
      for (row=0; row<nrow_recv; row++){
        row_dest = P_recv[2*row];
        row_mod = row_dest%(numPes*b);
        if (row_mod >= myRank*b && row_mod < (myRank+1)*b){
          row_blk = row_dest/(numPes*b);
          DEBUG_PRINTF("[%d] sending row %d to %d\n",myRank,row_blk*b+row_mod,i);
          lda_cpy(1,ncol_bw,lda_A,1,
                  A_bw+(row_blk*b+row_mod-myRank*b)-idx_off,
                  buffer_send+nrow_mine*ncol_tot);
#ifdef OFFLOAD_FAT_GEMM
          rindices[nrow_mine] = (row_blk*b+row_mod-myRank*b)+(int)(A_fw-A_bw);
#else
          lda_cpy(1,ncol_fw,lda_A,1,
                  A_fw+(row_blk*b+row_mod-myRank*b)-idx_off,
                  buffer_send+nrow_mine*ncol_tot+ncol_bw);
#endif
          P_send[nrow_mine*2] = P_r[row_blk*b+row_mod-myRank*b-idx_off];
          P_send[nrow_mine*2+1] = P_recv[2*row+1];
          nrow_mine++;
        }
      }
#ifdef OFFLOAD_FAT_GEMM
      offload_sparse_rw(nrow_mine,ncol_fw,lda_A,
                        buffer_send+ncol_bw,ncol_tot,rindices,OFF_A,'r');
#endif


      int nr2 = nrow_mine*2;
      MPI_Gather(&nr2,1,MPI_INT,
             recvcnts,1,MPI_INT,i,cdt.cm);
      if (myRank == i){
        displs[0] = 0;
        for (row=1; row<numPes; row++){
          displs[row] = displs[row-1] + recvcnts[row-1];
        }
      }
      MPI_Gatherv(P_send,2*nrow_mine,MPI_INT,
              P_buf,recvcnts,displs,MPI_INT,i,cdt.cm);
      if (myRank == i){
        displs[0] = 0;
        recvcnts[0] = ncol_tot*recvcnts[0]/2;
        for (row=1; row<numPes; row++){
          displs[row] = displs[row-1] + recvcnts[row-1];
          recvcnts[row] = ncol_tot*recvcnts[row]/2;
        }
      }
      MPI_Gatherv(buffer_send,ncol_tot*nrow_mine,MPI_DOUBLE,
                  buffer,recvcnts,displs,MPI_DOUBLE,i,cdt.cm);
      
    }
  }
  for (row=0; row<nrow_get; row++){
    DEBUG_PRINTF("[%d] P_mine[%d] = %d\n",myRank,row,P_mine[row]);
    row_dest = P_buf[row*2+1];
    P_r[row_dest] = P_buf[2*row];
    lda_cpy(1,ncol_bw,1,lda_A,
            buffer+row*ncol_tot,
            A_bw+row_dest);
#ifdef OFFLOAD_FAT_GEMM
    rindices[row] = row_dest+idx_off+(int)(A_fw-A_bw);
#else
    lda_cpy(1,ncol_fw,1,lda_A,
            buffer+row*ncol_tot+ncol_bw,
            A_fw+row_dest);
#endif
  }
#ifdef OFFLOAD_FAT_GEMM
    offload_sparse_rw(nrow_get,ncol_fw,lda_A,
                      buffer+ncol_bw,ncol_tot,rindices,OFF_A,'w');
#endif
}
#else

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
               const CommData   cdt){  /* column communiccator */
  int row, row_dest, row_mod, row_blk, i, ib;

  int ncol_tot = ncol_bw + ncol_fw;

  int nrow_get, nrow_recv;
  int * P_recv, * P_send;
  double * buffer_send;

#ifdef OFFLOAD_SKINNY_GEMM
  ABORT;
#endif
#ifdef OFFLOAD_FAT_GEMM
  ABORT;
#endif

  nrow_get = 0;

  for (row=0; row<nrow; row++){
    ib = (row%b)+b*numPes*(row/b)+glb_off+myRank*b;
    if (P_mine[row] != ib){
      P_buf[nrow_get] = P_mine[row];
      nrow_get++;
    }
  }
  buffer_send = buffer+nrow_get*ncol_tot;

  for (i=0; i<numPes; i++){
    if (i==myRank){
      nrow_recv = nrow_get;
      P_buf[nrow_get] = nrow_get;
      MPI_Bcast(P_buf+nrow_get,1,MPI_INT,i,cdt.cm);
      P_recv = P_buf;
    } else {
      MPI_Bcast(P_buf+nrow_get,1,MPI_INT,i,cdt.cm);
      nrow_recv = P_buf[nrow_get];
      P_recv = P_buf+nrow_get;
    }
    if (nrow_recv > 0){
      P_send = P_recv+nrow_recv+numPes;
      MPI_Bcast(P_recv,nrow_recv,MPI_INT,i,cdt.cm);
      std::fill(buffer_send,buffer_send+nrow_recv*ncol_tot,0.0);
      std::fill(P_send,P_send+nrow_recv,0);
      for (row=0; row<nrow_recv; row++){
        row_dest = P_recv[row];
        row_mod = row_dest%(numPes*b);
        if (row_mod >= myRank*b && row_mod < (myRank+1)*b){
          DEBUG_PRINTF("[%d] sending row %d to %d\n",myRank,row_blk*b+row_mod,i);
          row_blk = row_dest/(numPes*b);
          lda_cpy(1,ncol_bw,lda_A,1,
                  A_bw+(row_blk*b+row_mod-myRank*b)-idx_off,
                  buffer_send+row*ncol_tot);
          lda_cpy(1,ncol_fw,lda_A,1,
                  A_fw+(row_blk*b+row_mod-myRank*b)-idx_off,
                  buffer_send+row*ncol_tot+ncol_bw);
          P_send[row] = P_r[row_blk*b+row_mod-myRank*b-idx_off];
        }
      }
      MPI_Reduce(buffer_send,buffer,nrow_recv*ncol_tot,MPI_DOUBLE,MPI_SUM,i,cdt.cm);
      MPI_Reduce(P_send,P_recv,nrow_recv,MPI_INT,MPI_SUM,i,cdt.cm);
    }
  }
  nrow_get = 0;
  for (row=0; row<nrow; row++){
    ib = (row%b)+b*numPes*(row/b)+glb_off+myRank*b;
    DEBUG_PRINTF("[%d] P_mine[%d] = %d\n",myRank,row,P_mine[row]);
    if (P_mine[row] != ib){
      DEBUG_PRINTF("[%d] set pivot %d to %d %lf\n",myRank,row,P_buf[nrow_get], buffer[0]);
      P_r[row] = P_buf[nrow_get];
      lda_cpy(1,ncol_bw,1,lda_A,
              buffer+nrow_get*ncol_tot,
              A_bw+row);
      lda_cpy(1,ncol_fw,1,lda_A,
              buffer+nrow_get*ncol_tot+ncol_bw,
              A_fw+row);
      nrow_get++;
    }
  }
}

#endif
