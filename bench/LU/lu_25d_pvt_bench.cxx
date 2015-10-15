#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <math.h>
#include "CANDMC.h"
#include "omp.h"

#define MAX_OFF_DIAG    1.1

/* benchmark parallel tournament pivoting
 * n is the test matrix dimension 
 * b_sm is the small block dimension 
 * b_lrg is the large block dimension */
static
void lu_25d_pvt_bench(int64_t const         n,
                      int64_t const         b_sm, 
                      int64_t const         b_lrg, 
                      int const         myRank, 
                      int const         numPes, 
                      int const         c_rep,
                      int const         pvt,
                      int const         num_iter,
                      int const         start_iter,
                      CommData const    cdt){
  
  const CommData_t cdt_glb = cdt; 

  const int64_t matrixDim = n;
  const int64_t blockDim = b_sm;
  const int64_t big_blockDim = b_lrg;

  if (myRank == 0){ 
    printf("BENCHMARKING LU FACOTRIZATION OF SQUARE MATRIX\n");
    printf("MATRIX DIMENSION IS " PRId64 " ", matrixDim);
    printf("BLOCK DIMENSION IS " PRId64 " ", blockDim);
    printf("BIG BLOCK DIMENSION IS " PRId64 "\n", big_blockDim);
    printf("REPLICATION FACTOR, C, IS %d\n", c_rep);
#ifdef USE_DCMF
    printf("USING DCMF FOR COMMUNICATION\n");
#else
    printf("USING MPI FOR COMMUNICATION\n");
#endif
  }

  if (matrixDim < blockDim || matrixDim % blockDim != 0) {
    if (myRank == 0) printf("array_size mod block_size != 0!\n");
    ABORT;
  }

  const int num_blocks_dim = matrixDim/blockDim;
  const int num_pes_dim = sqrt(numPes/c_rep);

  if (myRank == 0){
    printf("NUM X BLOCKS IS %d\n", num_blocks_dim);
  //  printf("NUM Y BLOCKS IS %d\n", num_blocks_dim);
    printf("NUM X PROCS IS %d\n", num_pes_dim);
  //  printf("NUM Y PROCS IS %d\n", num_pes_dim);
  }

  if (num_pes_dim*num_pes_dim != numPes/c_rep || numPes%c_rep > 0 || c_rep > num_pes_dim){
    if (myRank == 0) 
      printf("ERROR: PROCESSOR GRID MISMATCH\n");
    ABORT;
  }
  if (big_blockDim%(num_pes_dim*blockDim)!=0  || big_blockDim<=0 || big_blockDim>matrixDim){
    if (myRank == 0) 
      printf("ERROR: BIG BLOCK DIMENSION MUST BE num_pes_dim*blockDim*x for some x.\n");
    ABORT;
  }

  if (num_blocks_dim%num_pes_dim != 0){
    if (myRank == 0) 
      printf("NUMBER OF BLOCKS MUST BE DIVISBLE BY THE 2D PROCESSOR GRID DIMENSION\n");
    ABORT;
  }
 
  int layerRank, intraLayerRank, myRow, myCol;

  CommData_t cdt_kdir;
  CommData_t cdt_row;
  CommData_t cdt_col;
  CommData_t cdt_kcol;

  RANK_PRINTF(0,myRank," set up kcol comms\n");
  RSETUP_KDIR_COMM(myRank, numPes, c_rep, cdt_kdir, layerRank, intraLayerRank);
  RANK_PRINTF(0,myRank," set up kdir comms\n");
  RSETUP_LAYER_COMM(num_pes_dim, layerRank, intraLayerRank, cdt_row, cdt_col, myRow, myCol);
  RANK_PRINTF(0,myRank," set up layer comms\n");
  SETUP_SUB_COMM(cdt_glb, cdt_kcol, (layerRank*num_pes_dim+myRow), 
                 myCol, (c_rep*num_pes_dim));
  const int my_num_blocks_dim = num_blocks_dim/num_pes_dim;
  const int num_big_blocks_dim = matrixDim/big_blockDim;
  const int num_small_in_big_blk = big_blockDim/blockDim;
  const int r_blk = num_small_in_big_blk/num_pes_dim;
  int i,j,ib,jb,iter;

  double * mat_A, * buffer;
  int * mat_P, * buf_P;

  double start_t, end_t, sum_t;
  int seed = 1000;

  assert(posix_memalign((void**)&mat_A, 
                        ALIGN_BYTES, 
                        my_num_blocks_dim*my_num_blocks_dim*
                        blockDim*blockDim*sizeof(double)) == 0);
  /* In the current implementation theschur complement update is done all at once.
   * The size of the update can exceed the size of the matrix stored on each processor.
   * This check insures that we have enough buffer space to do it properly. */
  if (num_big_blocks_dim > 1 && num_big_blocks_dim < (num_pes_dim/c_rep + ((num_pes_dim%c_rep) > 0))*2){
    if (myRank == 0) printf("WARNING: USING EXTRA MEMORY (SEE COMMENT IN CODE)\n");
    assert(posix_memalign((void**)&buffer, 
                          ALIGN_BYTES, 
                          r_blk*r_blk*num_pes_dim*2*num_big_blocks_dim*
                          blockDim*blockDim*sizeof(double)/c_rep) == 0);
    
  } else {
    assert(posix_memalign((void**)&buffer, 
                          ALIGN_BYTES, 
                          my_num_blocks_dim*my_num_blocks_dim*
                          blockDim*blockDim*sizeof(double)) == 0);
  }
  assert(posix_memalign((void**)&mat_P, 
                        ALIGN_BYTES, 
                        my_num_blocks_dim*blockDim*sizeof(int)) == 0);
  
  assert(posix_memalign((void**)&buf_P, 
                        ALIGN_BYTES, 
                        30*my_num_blocks_dim*blockDim*sizeof(int)) == 0);

  if (myRank == 0)  {
#ifdef PARTIAL_PVT
    printf("BENCHMARKING LU WITH PARTIAL PIVOTING\n");
#else
    if (pvt) printf("BENCHMARKING LU WITH TOURNAMENT PIVOTING\n");
    else printf("BENCHMARKING LU WITH NO PIVOTING (DIAGONALLY DOMINANT MATRIX)\n");
#endif
  }



  MPI_Barrier(cdt_glb.cm);

  lu_25d_pvt_params_t p;

  p.pvt                 = pvt;
  p.myRank              = myRank;
  p.c_rep               = c_rep;
  p.matrixDim           = matrixDim;
  p.blockDim            = blockDim;
  p.big_blockDim        = big_blockDim;
  p.num_pes_dim         = num_pes_dim;
  p.layerRank           = layerRank;
  p.myRow               = myRow;
  p.myCol               = myCol;
  p.cdt_row             = cdt_row;
  p.cdt_col             = cdt_col;
  p.cdt_kdir            = cdt_kdir;
  p.cdt_kcol            = cdt_kcol;
#ifdef PARTIAL_PVT
  p.is_tnmt_pvt = 0;
#else
  p.is_tnmt_pvt = 1;
#endif

  TAU_FSTART(first_iter);

  sum_t = 0.0;

  for (iter=0; iter<num_iter+start_iter; iter++){       
    srand48(seed*myRank+seed);
    if (layerRank == 0){
      for (ib=0; ib < my_num_blocks_dim; ib++){
        for (i=0; i < blockDim; i++){
          for (jb=0; jb < my_num_blocks_dim; jb++){
            for (j=0; j < blockDim; j++){
              if (pvt){
                mat_A[(ib*blockDim + i)*my_num_blocks_dim*blockDim + jb*blockDim+j] 
                    = drand48();
              } else {
                if (ib == jb && i == j && myRow == myCol)
                  mat_A[(ib*blockDim + i)*my_num_blocks_dim*blockDim + jb*blockDim+j] 
                      = (MAX_OFF_DIAG+drand48())*matrixDim;
                else
                  mat_A[(ib*blockDim + i)*my_num_blocks_dim*blockDim + jb*blockDim+j] 
                                                                       = drand48();
              }
            }
          }
        }
      }
    }
    else { /* If I am not the first layer I will be receiving whatever data I need from
            * layer 0, and my updates will just be accumulating to the Schur complement */
      for (i=0; i < my_num_blocks_dim*my_num_blocks_dim*blockDim*blockDim; i++)
        mat_A[i] = 0.0;
    }
    if (iter >= start_iter){
      if (iter == start_iter){
        TAU_FSTOP(first_iter);
        TAU_FSTART(timer);
      }
      MPI_Barrier(cdt_glb.cm);
      start_t = MPI_Wtime();
    }
    lu_25d_pvt(&p, mat_A, mat_P, buf_P, buffer, iter>0);
    if (iter >= start_iter){
      end_t = MPI_Wtime();
      MPI_Barrier(cdt_glb.cm);
      sum_t +=end_t -start_t;
    }
  }
  TAU_FSTOP(timer);
  
  if (myRank == 0){
    printf("Warmed up for %d iterations, benchmarked %d iterations\n", start_iter, num_iter);
    printf("Time elapsed per iteration: %lf\n", sum_t/num_iter);
    printf("Gigaflops: %lf\n", (2./3)*matrixDim*matrixDim*matrixDim/
                                (sum_t/num_iter)*1E-9);
  }
  FREE_CDT((&cdt_kdir));
  FREE_CDT((&cdt_row));
  FREE_CDT((&cdt_col));
} 


static
char* getCmdOption(char ** begin, char ** end, const std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

int main(int argc, char **argv) {
  int myRank, numPes, c_rep, num_iter, pvt, nwarm;
  int64_t b_sm, b_lrg, n;

  CommData_t cdt_glb;
  RINIT_COMM(numPes, myRank, 4, 4, cdt_glb);

#if 0//SHARE_MIC > 1
  char sout[128];
  int offset = 0;
 // offset += (240/SHARE_MIC)*(myRank%SHARE_MIC);
 // sprintf(sout, "export MIC_KMP_AFFINITY=explicit,granularity=fine,proclist=[%d-%d:1]",offset+1,offset+240/SHARE_MIC);
 // printf("[%d] export MIC_KMP_AFFINITY=explicit,granularity=fine,proclist=[%d-%d:1]",myRank,offset+1,offset+240/SHARE_MIC);
  offset = 0;
  offset += (60/SHARE_MIC)*(myRank%SHARE_MIC);
  sprintf(sout, "MIC_KMP_PLACE_THREADS=30c,4t,%dO",offset);
  if (myRank == 0)
    printf("[%d] export MIC_KMP_PLACE_THREADS=30c,4t,%dO\n",myRank,offset);
  putenv(sout);
  char hostname[128];
  gethostname(hostname, 128);
  if (myRank == 0)
    printf("[%d] hostname is %s\n",myRank,hostname);
  omp_set_num_threads(8);
  if (myRank == 0)
    printf("threads=%d\n",omp_get_max_threads());
#endif

  if (getCmdOption(argv, argv+argc, "-n")){
    n = atoi(getCmdOption(argv, argv+argc, "-n"));
    if (n <= 0) n = 512;
  } else n = 512;

  if (getCmdOption(argv, argv+argc, "-num_iter")){
    num_iter = atoi(getCmdOption(argv, argv+argc, "-num_iter"));
    if (num_iter <= 0) num_iter = 1;
  } else num_iter = 3;

  if (getCmdOption(argv, argv+argc, "-nwarm")){
    nwarm = atoi(getCmdOption(argv, argv+argc, "-nwarm"));
    if (nwarm <= 0) nwarm = 1;
  } else nwarm = 1;

  if (getCmdOption(argv, argv+argc, "-c_rep")){
    c_rep = atoi(getCmdOption(argv, argv+argc, "-c_rep"));
    if (c_rep <= 1) c_rep = 1;
  } else {
    c_rep = 1;
    if (sqrt(numPes/c_rep)*sqrt(numPes/c_rep) != numPes/c_rep){
      if (numPes >= 8 && numPes%2 == 0) c_rep = 2;
    }
  }
  if (getCmdOption(argv, argv+argc, "-b_sm")){
    b_sm = atoi(getCmdOption(argv, argv+argc, "-b_sm"));
    if (b_sm < 1) b_sm = 4;
  } else b_sm = 4;
  if (getCmdOption(argv, argv+argc, "-b_lrg")){
    b_lrg = atoi(getCmdOption(argv, argv+argc, "-b_lrg"));
    if (b_lrg < b_sm*sqrt(numPes/c_rep)) 
      b_lrg = b_sm*sqrt(numPes/c_rep);
  } else 
      b_lrg = b_sm*sqrt(numPes/c_rep);
  if (getCmdOption(argv, argv+argc, "-pvt")){
    pvt = atoi(getCmdOption(argv, argv+argc, "-pvt"));
    if (pvt < 1) pvt = 3;
  } else pvt = 3;


#ifdef USE_MIC
  char * dummy;
  assert(0==posix_memalign((void**)&dummy, 1024, sizeof(double)*12345)); 
  #pragma offload target(mic:0) \
    nocopy(dummy:length(12345) alloc_if(1) free_if(0)) 
  {
  }
#endif
  if (sqrt(numPes/c_rep)*sqrt(numPes/c_rep) != numPes/c_rep){
    if (myRank == 0)
      printf("LU test needs square processor grid... exiting.\n");
    return 0;
  }


#ifdef TAU
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(myRank);
  TAU_PROFILE_SET_CONTEXT(0);
#endif
  

  MPI_Barrier(cdt_glb.cm);
#if (defined(PARTIAL_PVT) || defined(TNMT_PVT))
  lu_25d_pvt_bench(n, b_sm, b_lrg, myRank, numPes, c_rep, 1, num_iter, nwarm, cdt_glb);
#else
#ifdef NO_PVT
  lu_25d_pvt_bench(n, b_sm, b_lrg, myRank, numPes, c_rep, 0, num_iter, nwarm, cdt_glb);
#else
  lu_25d_pvt_bench(n, b_sm, b_lrg, myRank, numPes, c_rep, 1, num_iter, nwarm, cdt_glb);
  if (pvt&0x1)
    lu_25d_pvt_bench(n, b_sm, b_lrg, myRank, numPes, c_rep, 1, num_iter, nwarm, cdt_glb);
  else 
    lu_25d_pvt_bench(n, b_sm, b_lrg, myRank, numPes, c_rep, 0, num_iter, nwarm, cdt_glb);
  if ((pvt&0x2)>>1)
    lu_25d_pvt_bench(n, b_sm, b_lrg, myRank, numPes, c_rep, 0, num_iter, nwarm, cdt_glb);
#endif
#endif
  
  MPI_Barrier(cdt_glb.cm);
  COMM_EXIT;
  return 0;
}
