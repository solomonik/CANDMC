/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include "CANDMC.h"
#include "../../alg/shared/util.h"

static
char* getopt(char ** begin, char ** end, const std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

int main(int argc, char **argv) {
  int myRank, numPes, n, b, pr, pc, ipr, ipc, i, j, loc_off, m, b_agg;
  double * loc_A, * full_A, * scala_EL, * full_EL;
  double * work;
  int * iwork;
  int ictxt_lyr, ictxt_rect, info, iam, inprocs, lwork;
  int desc_A[9], desc_EC[9];
  int64_t liwork;
  int c_rep, num_lyrPes, bw;

  CommData_t cdt_glb, cdt_lyr, cdt_rep, cdt_supercol;
  CommData_t cdt_row, cdt_col;
  INIT_COMM(numPes, myRank, 1, cdt_glb);

  if (myRank == 0)
    printf("Usage: %s -n 'matrix dimension' -b 'distribution blocking factor' -bw 'matrix bandwidth to reduce to' -b_agg 'aggregation blocking factor' -c_rep 'replication factor'\n", argv[0]);

  if (     getopt(argv, argv+argc, "-c_rep") &&
      atoi(getopt(argv, argv+argc, "-c_rep")) > 0 ){
    c_rep = atoi(getopt(argv, argv+argc, "-c_rep"));
  } else c_rep = 1; 
  if (numPes % c_rep > 0){
    if (myRank == 0)
      printf("replication factor must divide into number of processors, terminating...\n");
    return 0;
  }
  num_lyrPes = numPes/c_rep;
  pr = sqrt(num_lyrPes);
  while (num_lyrPes%pr!=0) pr++;
  if (pr*pr != num_lyrPes){
    if (myRank == 0)
      printf("Full to banded test needs base square processor grid, terminating...\n");
    return 0;
  }
  if (     getopt(argv, argv+argc, "-b") &&
      atoi(getopt(argv, argv+argc, "-b")) > 0 )
    b = atoi(getopt(argv, argv+argc, "-b"));
  else 
    b = 4;
  if (     getopt(argv, argv+argc, "-n") &&
      atoi(getopt(argv, argv+argc, "-n")) > 0 )
    n = atoi(getopt(argv, argv+argc, "-n"));
  else 
    n = 4*b*pr;

  if (     getopt(argv, argv+argc, "-b_agg") &&
      atoi(getopt(argv, argv+argc, "-b_agg")) > 0 )
    b_agg = atoi(getopt(argv, argv+argc, "-b_agg"));
  else 
    b_agg = std::min(n,16);
  
  if (     getopt(argv, argv+argc, "-bw") &&
      atoi(getopt(argv, argv+argc, "-bw")) > 0 )
    bw = atoi(getopt(argv, argv+argc, "-bw"));
  else 
    bw = 8;

  if (myRank == 0)
    printf("Executed as '%s -n %d -b %d -bw %d -pr %d -b_agg %d -c_rep %d'\n", 
            argv[0], n, b, bw, pr, b_agg, c_rep);

  int spr = pr*c_rep;

  if (num_lyrPes % pr != 0) {
    if (myRank == 0){
      printf("%d mod %d != 0 Number of processor grid ", num_lyrPes, pr);
      printf("rows must divide into number of processors in layer\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (n % spr != 0) {
    if (myRank == 0){
      printf("%d mod %d != 0 Number of processor grid ", n, spr);
      printf("rows*c_rep must divide into the matrix dimension\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  pc = num_lyrPes / pr;
  if (num_lyrPes % pr != 0) {
    if (myRank == 0){
      printf("%d mod %d != 0 Number of processor grid ", n, pc);
      printf("columns must divide into the matrix dimension\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  
  if (myRank == 0){ 
    printf("Testing full to band symmetric reduction ");
    printf("%d-by-%d matrix with block size %d\n",n,n,b);
    printf("Using %d processors in %d-by-%d-by-%d grid.\n", numPes, pr, pc, c_rep);
  }

  loc_A    = (double*)malloc(n*n*sizeof(double)/num_lyrPes);
  scala_EL = (double*)malloc(n*sizeof(double));
  full_A  = (double*)malloc(n*n*sizeof(double));
  full_EL = (double*)malloc(n*sizeof(double));
  lwork       = MAX(5*n*n/num_lyrPes,30*n);
  //liwork > 6*NNP, where  NNP = MAX( N, NPROW*NPCOL + 1, 4 )
  liwork      = 6*MAX(5*n, MAX(4, spr*pc+1));
  work        = (double*)malloc(lwork*sizeof(double));
  iwork       = (int*)malloc(liwork*sizeof(int));

  srand48(646);

  char cC = 'C';
  Cblacs_pinfo(&iam,&inprocs);
  Cblacs_get(-1, 0, &ictxt_rect);
  Cblacs_gridinit(&ictxt_rect, &cC, spr, pc);
  Cblacs_get(-1, 0, &ictxt_lyr);
  int gmap[spr*pc];
  for (i=0; i<spr; i++){
    for (j=0; j<pc; j++){
      gmap[i+j*spr] = i+j*spr;
    }
  }
  Cblacs_gridmap(&ictxt_lyr, gmap+pr*((myRank/pr)%c_rep), spr, pr, pc);
  cdescinit(desc_A, n, n,
		        b, b,
		        0, 0,
		        ictxt_lyr, n/pr, 
				    &info);
  cdescinit(desc_EC, n, n,
		        b, b,
		        0, 0,
		        ictxt_lyr, n/pr, 
				    &info);
  LIBT_ASSERT(info==0);


  if (myRank == 0) printf("Testing 3D 2-level full-to-band algorithm\n");
  srand48(657);
  
  ipc = myRank / spr;
  ipr = myRank % pr;
  init_dist_sym_matrix(n,ipr,pr,ipc,pc,b,full_A,loc_A); 
  cdsyevx( 'N', 'A', 'L', n, full_A, n, 0.0, 0.0, 0, 0, 0.0,
           &m, full_EL, NULL, n, work, lwork, iwork,
           NULL, &info);

  SETUP_SUB_COMM(cdt_glb, cdt_supercol, 
                 myRank%spr, 
                 ipc, 
                 spr);
  SETUP_SUB_COMM(cdt_glb, cdt_col, 
                 ipr, 
                 myRank/pr, 
                 pr);
  SETUP_SUB_COMM(cdt_glb, cdt_row, 
                 ipc, 
                 myRank%spr, 
                 pc);
  SETUP_SUB_COMM(cdt_glb, cdt_rep, 
                 cdt_supercol.rank/pr, 
                 cdt_row.rank*cdt_col.np + cdt_col.rank,
                 c_rep);
  SETUP_SUB_COMM(cdt_glb, cdt_lyr, 
                 cdt_row.rank*cdt_col.np + cdt_col.rank,
                 cdt_rep.rank,
                 pr*pc);
  //printf("Processor %d (by rank ) is in (2D lyr) row %d (2D lyr) col %d superrow %d layer %d and layer rank %d\n",
   //      myRank, cdt_col.rank, cdt_row.rank, cdt_supercol.rank, cdt_rep.rank, cdt_lyr.rank);
  //printf("Processor %d (by color) is in (2D lyr) row %d (2D lyr) col %d superrow %d layer %d and layer rank %d\n",
     //    myRank, cdt_row.color%pr, cdt_col.color/c_rep, cdt_row.color, cdt_lyr.color, cdt_rep.color);
  srand48(657);
  init_dist_sym_matrix(n,ipr,pr,ipc,pc,b,full_A,loc_A); 
  pview pv_2d;
  pv_2d.rrow = 0;
  pv_2d.rcol = 0;
  pv_2d.crow = cdt_row;
  pv_2d.ccol = cdt_col;
  pv_2d.cworld = cdt_lyr;
  pv_2d.ictxt = ictxt_lyr;
  
  pview pv_2drect;
  pv_2drect.rrow = 0;
  pv_2drect.rcol = 0;
  pv_2drect.crow = cdt_row;
  pv_2drect.ccol = cdt_supercol;
  pv_2drect.cworld = cdt_glb;
  pv_2drect.ictxt = ictxt_rect;

  pview_3d pv;
  pv.prect = pv_2drect;
  pv.plyr = pv_2d;
  pv.clyr = cdt_rep;
  pv.cworld = cdt_glb;

  sym_full2band_3d(loc_A, n/pr, n, b_agg, bw, b, &pv);
  loc_off = 0;
  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      if ((i/b)%pc == ipc && (j/b)%pr == ipr){
        if (i-j > b_agg || j-i > b_agg) 
          loc_A[loc_off] = 0.0;
        loc_off++;
      }
    }
  }
//  if (pv.cworld.rank == 0)
//   printf("A:\n");
//  print_dist_mat(n, n, b, pv.ccol.rank, pv.ccol.np, 0, 
//                          pv.crow.rank, pv.crow.np, 0,
//                 pv.cworld.cm, loc_A, n/pr);

//  print_matrix(loc_A, n, n);
  if (cdt_rep.rank == 0){
    cpdsyevx('N', 'A', 'L', n, loc_A, 1, 1, desc_A, 0.0, 0.0, 0, 0, 0.0,
             &m, 0, scala_EL, 0.0, NULL, 0, 0, NULL, work, lwork, iwork,
             liwork, NULL, NULL, NULL, &info);

    if (myRank == 0){
      int pass = 1;
      for (i=0; i<n; i++){
        if (fabs(full_EL[i] - scala_EL[i]) > 1.E-10){
          printf("incorrect eigenvalue %d, band matrix had eigenvalue %E, original matrix had %E\n",
                  i, scala_EL[i], full_EL[i]);
          pass = 0;
        }
      }
      if (pass)
        printf("Verification of eigenvalues of banded matrix completed without error\n");
      else
        printf("Verification of eigenvalues of banded matrix completed with ERRORS!\n");
    }
  }

  MPI_Finalize();
  return 0;
} /* end function main */


