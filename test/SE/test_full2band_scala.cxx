#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include <assert.h>
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
  int myRank, numPes, n, b, pr, pc, ipr, ipc, i, j, loc_off, m, nz, b_agg;
  double * loc_A, * full_A, * scala_EL, * full_EL, * loc_EC, * full_EC;
  double * scala_D, * scala_E, * full_D, * full_E, * work, * gap;
  double * scala_T, * full_T;
  int * iwork, * iclustr, * ifail;
  int icontxt, info, iam, inprocs, lwork;
  char cC = 'C';
  int desc_A[9], desc_EC[9];
  double time;
  int64_t liwork;

  CommData_t cdt_glb;
  CommData_t cdt_row, cdt_col;
  INIT_COMM(numPes, myRank, 1, cdt_glb);
  CommData_t cdt_diag;

  if (myRank == 0)
    printf("Usage: %s -n 'matrix dimension' -b 'distribution blocking factor' -b_agg 'aggregation blocking factor'\n", argv[0]);

  if (     getopt(argv, argv+argc, "-pr") &&
      atoi(getopt(argv, argv+argc, "-pr")) > 0 ){
    pr = atoi(getopt(argv, argv+argc, "-pr"));
  } else {
    pr = sqrt(numPes);
    while (numPes%pr!=0) pr++;
  }
  if (pr != sqrt(numPes)){
    if (myRank == 0)
      printf("Full to banded test needs square processor grid, terminating...\n");
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
    n = 2*b*pr;
  if (     getopt(argv, argv+argc, "-b_agg") &&
      atoi(getopt(argv, argv+argc, "-b_agg")) > 0 )
    b_agg = atoi(getopt(argv, argv+argc, "-b_agg"));
  else 
    b_agg = 8;

  if (myRank == 0)
    printf("Executed as '%s -n %d -b %d -pr %d -b_agg %d'\n", 
            argv[0], n, b, pr, b_agg);


  if (numPes % pr != 0) {
    if (myRank == 0){
      printf("%d mod %d != 0 Number of processor grid ", numPes, pr);
      printf("rows must divide into number of processors\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (n % pr != 0) {
    if (myRank == 0){
      printf("%d mod %d != 0 Number of processor grid ", n, pr);
      printf("rows must divide into the matrix dimension\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  pc = numPes / pr;
  if (numPes % pr != 0) {
    if (myRank == 0){
      printf("%d mod %d != 0 Number of processor grid ", n, pc);
      printf("columns must divide into the matrix dimension\n");
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  ipc = myRank / pr;
  ipr = myRank % pr;
  
  
  if (myRank == 0){ 
    printf("Testing full to band symmetric reduction ");
    printf("%d-by-%d matrix with block size %d\n",n,n,b);
    printf("Using %d processors in %d-by-%d grid.\n", numPes, pr, pc);
  }

  loc_A    = (double*)malloc(n*n*sizeof(double)/numPes);
  loc_EC   = (double*)malloc(n*n*sizeof(double)/numPes);
  scala_EL = (double*)malloc(n*sizeof(double));
  scala_D  = (double*)malloc(n*sizeof(double));
  scala_E  = (double*)malloc(n*sizeof(double));
  scala_T  = (double*)malloc(n*sizeof(double));
  full_A  = (double*)malloc(n*n*sizeof(double));
  full_EC = (double*)malloc(n*n*sizeof(double));
  full_EL = (double*)malloc(n*sizeof(double));
  full_D  = (double*)malloc(n*sizeof(double));
  full_E  = (double*)malloc(n*sizeof(double));
  full_T  = (double*)malloc(n*sizeof(double));
  lwork = 6*MAX(n*n/numPes,MAX(30*n, MAX(4, pr*pc+1)));
  work        = (double*)malloc(lwork*sizeof(double));
  liwork      = 6*MAX(5*n, MAX(4, pr*pc+1));
  iwork       = (int*)malloc(liwork*sizeof(int));
  ifail       = (int*)malloc(n*sizeof(int));
  iclustr     = (int*)malloc(2*pr*pc*sizeof(int));
  gap         = (double*)malloc(pr*pc*sizeof(double));

  srand48(666);

  Cblacs_pinfo(&iam,&inprocs);
  Cblacs_get(-1, 0, &icontxt);
  Cblacs_gridinit(&icontxt, &cC, pr, pc);
  cdescinit(desc_A, n, n,
		        b, b,
		        0, 0,
		        icontxt, n/pr, 
				    &info);
  cdescinit(desc_EC, n, n,
		        b, b,
		        0, 0,
		        icontxt, n/pr, 
				    &info);
  assert(info==0);


  if (myRank == 0) printf("Testing 2D 2-level full-to-band algorithm\n");
  srand48(667);
  
  init_dist_sym_matrix(n,ipr,pr,ipc,pc,b,full_A,loc_A); 
  cdsyevx( 'N', 'A', 'L', n, full_A, n, 0.0, 0.0, 0, 0, 0.0,
           &m, full_EL, NULL, n, work, lwork, iwork,
           NULL, &info);

  SETUP_SUB_COMM(cdt_glb, cdt_row, 
                 myRank/pr, 
                 myRank%pr, 
                 pc);
  SETUP_SUB_COMM(cdt_glb, cdt_col, 
                 myRank%pr, 
                 myRank/pr, 
                 pr);
  if (ipr == ipc){
    SETUP_SUB_COMM(cdt_glb, cdt_diag, 
                   ipr, 
                   0, 
                   pr);
  } else {
    SETUP_SUB_COMM(cdt_glb, cdt_diag, 
                   myRank, 
                   1, 
                   pr);
  }

  srand48(667);
  init_dist_sym_matrix(n,ipr,pr,ipc,pc,b,full_A,loc_A); 
  pview pv;
  pv.rrow = 0;
  pv.rcol = 0;
  pv.crow = cdt_row;
  pv.ccol = cdt_col;
  pv.cdiag = cdt_diag;
  pv.cworld = cdt_glb;
  sym_full2band_scala(loc_A, n/pr, n, b_agg, b, &pv, desc_A, loc_A);
  loc_off = 0;
  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      if ((i/b)%pc == ipc && (j/b)%pr == ipr){
        if (abs(i-j) > b_agg) 
          loc_A[loc_off] = 0.0;
        loc_off++;
      }
    }
  }
  //if (myRank == 0)
    //printf("final A:\n");
//  print_dist_mat(n,n,b,ipr,pr,0,ipc,pc,0,cdt_glb.cm,loc_A,n/pr); 
  //print_matrix(loc_A, n, n);
  cpdsyevx('N', 'A', 'L', n, loc_A, 1, 1, desc_A, 0.0, 0.0, 0, 0, 0.0,
           &m, 0, scala_EL, 0.0, NULL, 0, 0, NULL, work, lwork, iwork,
           liwork, NULL, NULL, NULL, &info);

  if (myRank == 0){
    for (i=0; i<n; i++){
      if (fabs(full_EL[i] - scala_EL[i]) > 1.E-6){
        printf("incorrect eigenvalue %d, band matrix had eigenvalue %E, original matrix had %E\n",
                i, scala_EL[i], full_EL[i]);
      }
    }
    printf("Verification of eigenvalues of banded matrix completed\n");
  }

  MPI_Finalize();
  return 0;
} /* end function main */


