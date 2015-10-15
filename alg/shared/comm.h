/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __COMM_H__
#define __COMM_H__

#define USE_MPI

#include <assert.h>

#ifdef USE_MPI
/*********************************************************
 *                                                       *
 *                      MPI                              *
 *                                                       *
 *********************************************************/
#include "mpi.h"
#include "util.h"
//latency time per message
#define COST_LATENCY 1.e-6
//memory bandwidth: time per per byte
#define COST_MEMBW 1.e-9
//network bandwidth: time per byte
#define COST_NETWBW 5.e-10
//flop cost: time per flop
#define COST_FLOP 2.e-11
//flop cost: time per flop
#define COST_OFFLOADBW 5.e-10


//typedef MPI_Comm COMM;

typedef class CommData {
  public:
  MPI_Comm cm;
  int np;
  int rank;
  int color;
  int alive;

  double estimate_bcast_time(long_int msg_sz) {
#ifdef BGQ
    return msg_sz*(double)COST_NETWBW+COST_LATENCY;
#else
    return msg_sz*(double)log2((double)np)*COST_NETWBW;
#endif
  }
  
  double estimate_allred_time(long_int msg_sz) {
#ifdef BGQ
    return msg_sz*(double)(2.*COST_MEMBW+COST_NETWBW)+COST_LATENCY;
#else
    return msg_sz*(double)log2((double)np)*(2.*COST_MEMBW+COST_FLOP+COST_NETWBW);
#endif
  }
  
  double estimate_alltoall_time(long_int chunk_sz) {
    return chunk_sz*np*log2((double)np)*COST_NETWBW+2.*log2((double)np)*COST_LATENCY;
  }
  
  double estimate_alltoallv_time(long_int tot_sz) {
    return 2.*tot_sz*log2((double)np)*COST_NETWBW+2.*log2((double)np)*COST_LATENCY;
  }
} CommData_t;

//2d procesor grid local processor view
class pview {
  public:
  //current root row
  int rrow;
  //current root col
  int rcol;
  //row communicatir
  CommData_t crow;
  //column communicator
  CommData_t ccol;
  //diagonal communicator
  CommData_t cdiag;
  //world communicator
  CommData_t cworld;
#ifdef USE_SCALAPACK
  //scalapack context for 2D grid
  int ictxt;
#endif
};


//3d procesor grid local processor view
class pview_3d {
  public:
  //context for 2D rectangular folding of 3D grid
  pview prect;

  //context for my 2D layer
  pview plyr;

  //layer (replication dimension) communicator
  CommData_t clyr;

  //column-layers of the processor grid (there is crow.np of these in total)
  CommData_t cworld;
};

#ifdef PRINTALL
#define CPRINTF(cdt,...) \
  do { if (cdt.rank == 0) printf(__VA_ARGS__); } while (0)
#else
#define CPRINTF(...)
#endif

#define POST_BCAST(buf, sz, type, root, cdt, bcast_req)  \
  do {                                                   \
  MPI_Bcast(buf, sz, type, root, cdt.cm); } while(0)

#define WAIT_BCAST(cdt, bcast_req)


#define SET_COMM(_cm, _rank, _np, _cdt) \
do {                                    \
    _cdt.cm    = _cm;                  \
    _cdt.rank  = _rank;                \
    _cdt.np    = _np;                  \
    _cdt.alive = 1;                    \
} while (0)
    
#define RINIT_COMM(numPes, myRank, nr, nb, cdt)                 \
  do {                                                          \
    INIT_COMM(numPes, myRank, nr, cdt);                         \
  } while(0)

#define INIT_COMM(numPes, myRank, nr, cdt)                      \
  do {                                                          \
  MPI_Init(&argc, &argv);                                       \
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);                       \
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);                       \
  SET_COMM(MPI_COMM_WORLD, myRank, numPes, cdt);                 \
  } while(0)


#define COMM_EXIT                               \
  do{                                           \
  MPI_Finalize(); } while(0)            

#define SETUP_SUB_COMM(cdt_master, cdt, commrank, bcolor, p)        \
  do {                                                              \
  cdt.rank     = commrank;                                          \
  cdt.np       = p;                                                 \
  cdt.color    = bcolor;                                            \
  cdt.alive    = 1;                                                 \
  MPI_Comm_split(cdt_master.cm,                                     \
                 bcolor,                                            \
                 commrank,                                          \
                 &cdt.cm); } while(0)

#define SETUP_SUB_COMM_SHELL(cdt_master, cdt, commrank, bcolor, p)  \
  do {                                                              \
  cdt.rank     = commrank;                                          \
  cdt.np       = p;                                                 \
  cdt.color    = bcolor;                                            \
  cdt.alive    = 0;                                                 \
  } while(0)

#define SHELL_SPLIT(cdt_master, cdt) \
  do {                                                              \
  cdt.alive    = 1;                                                 \
  MPI_Comm_split(cdt_master.cm,                                     \
                 cdt.color,                                         \
                 cdt.rank,                                          \
                 &cdt.cm); } while(0)


#define RSETUP_KDIR_COMM(myRank, p, c, cdt, commrank, color)	\
  do {									\
  commrank 	= myRank/(p/c);						\
  color 	= myRank%(p/c);						\
  cdt.rank 	= commrank;						\
  cdt.np 	= c;							\
  MPI_Comm_split(MPI_COMM_WORLD, 					\
		 color, 						\
		 commrank, 						\
		 &(cdt.cm)); } while(0)


#define RSETUP_LAYER_COMM(pesdim, commrank, color, cdt_row, cdt_col, row, col)			\
  do {									\
  MPI_Comm MPI_INTRALAYER_COMM;						\
  MPI_Comm_split(MPI_COMM_WORLD, commrank, color, &MPI_INTRALAYER_COMM);\
  row = color / pesdim;							\
  col = color % pesdim;							\
  MPI_Comm_split(MPI_INTRALAYER_COMM, myRow, myCol, &(cdt_row.cm));	\
  MPI_Comm_split(MPI_INTRALAYER_COMM, myCol, myRow, &(cdt_col.cm));	\
  cdt_row.np = pesdim;							\
  cdt_row.rank = col;							\
  cdt_col.np = pesdim;							\
  cdt_col.rank = row;							\
  } while(0)



#define FREE_CDT(cdt)			\
  do {					\
  MPI_Comm_free(&(cdt->cm)); } while(0)

#endif
  
#endif
