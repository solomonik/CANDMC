/* Copyright (c) Edgar Solomonik 2015, all rights reserved. This code is part of the CANDMC repository, protected under a two-clause BSD license. */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include "CANDMC.h"
#include "../../alg/shared/util.h"

#define NUM_ITER 3
static
char* getopt(char ** begin, char ** end, const std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

int main(int argc, char **argv) {
  int myRank, numPes, pr, pc, ipr, ipc, j, iter, niter;
  int64_t n, b, i, b_agg, bw;
  double * loc_A;
  int ictxt_lyr, ictxt_rect, info, iam, inprocs;
  int desc_A[9], desc_EC[9];
  int c_rep, num_lyrPes;

  CommData_t cdt_glb, cdt_lyr, cdt_rep, cdt_supercol;
  CommData_t cdt_row, cdt_col;
  INIT_COMM(numPes, myRank, 1, cdt_glb);

  if (myRank == 0)
    printf("Usage: %s -n 'matrix dimension' -b 'distribution blocking factor' -bw 'matrix bandwidth to reduce to' -b_agg 'aggregation blocking factor' -c_rep 'replication factor' -niter 'number of iterations' \n", argv[0]);

  if (     getopt(argv, argv+argc, "-niter") &&
      atoi(getopt(argv, argv+argc, "-niter")) > 0 )
    niter = atoi(getopt(argv, argv+argc, "-niter"));
  else 
    niter = NUM_ITER;

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
    b_agg = std::min(n,(int64_t)16);
  
  if (     getopt(argv, argv+argc, "-bw") &&
      atoi(getopt(argv, argv+argc, "-bw")) > 0 )
    bw = atoi(getopt(argv, argv+argc, "-bw"));
  else 
    bw = 8;

  if (myRank == 0)
    printf("Executed as '%s -n %ld -b %ld -bw %ld -pr %d -b_agg %ld -c_rep %d'\n", 
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
      printf("%ld mod %d != 0 Number of processor grid ", n, spr);
      printf("rows*c_rep must divide into the matrix dimension\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  pc = num_lyrPes / pr;
  if (num_lyrPes % pr != 0) {
    if (myRank == 0){
      printf("%ld mod %d != 0 Number of processor grid ", n, pc);
      printf("columns must divide into the matrix dimension\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  
  if (myRank == 0){ 
    printf("Benchmarking 2.5D full to band symmetric reduction ");
    printf("%ld-by-%ld matrix with block size %ld\n",n,n,b);
    printf("Using %d processors in %d-by-%d-by-%d grid.\n", numPes, pr, pc, c_rep);
  }

  loc_A    = (double*)malloc(n*n*sizeof(double)/num_lyrPes);


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


  
  ipc = myRank / spr;
  ipr = myRank % pr;
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

  TAU_PROFILE_TIMER(timer, "main", "int (int, char**)", TAU_USER);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_SET_NODE(myRank);
  TAU_PROFILE_SET_CONTEXT(0);


  //init_dist_sym_matrix(n,ipr,pr,ipc,pc,b,full_A,loc_A); 
  srand48(666*myRank);
  CTF_Timer_epoch ep1("full2band_2.5D"); 
  ep1.begin();
  double time = MPI_Wtime();
  for (iter=0; iter<niter; iter++){
    //FIXME: nonsymmetric
    CTF_Timer ti("initialization of data");
    ti.start();
    if (cdt_rep.rank == 0){
      for (i=0; i<n*n*c_rep/numPes; i++){
        loc_A[i] = drand48();
      }
    }
    ti.stop();
    CTF_Timer trep("replication of initial A");
    trep.start();
    MPI_Bcast(loc_A, n*n*c_rep/numPes, MPI_DOUBLE, 0, cdt_rep.cm);
    trep.stop();
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
  }
  time = MPI_Wtime()-time;
  ep1.end();
  if(myRank == 0){
    printf("Completed %u iterations of 2.5D full to band\n", iter);
    printf("2.5D CANDMC full to band n = %ld bw = %ld b_agg = %ld b = %ld p = %d c = %d: sec/iteration: %lf ", n, bw, b_agg, b, numPes, c_rep, time/niter);
    printf("Gigaflops (4/3)n^3: %lf\n", ((4./3.)*n*n*n)/(time/niter)*1E-9);
  }
  free(loc_A);



  MPI_Finalize();
  return 0;
} /* end function main */


