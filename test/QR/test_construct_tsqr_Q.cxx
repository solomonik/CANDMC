#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <string>
#include <ios>
#include <fstream>
#include "CANDMC.h"
using namespace std;


/**
 * \brief Test construct Q from TSQR
 *
 * \param[in] m number of rows in A
 * \param[in] b number of columns in A
 * \param[in] root is the root process that owns A
 * \param[in] myRank rank in communicator column
 * \param[in] numPes number of processes in column
 * \param[in] comm MPI communicator for column
 **/

#ifdef READ_FILE
void par_tsqr_unit_test(const char * filename, 
                        int const root, 
                        int const myRank, 
                        int const numPes, 
                        CommData_t  cdt){
#else
void par_tsqr_unit_test(int const m,
                        int const b, 
                        int const root, 
                        int const myRank, 
                        int const numPes, 
                        CommData_t  cdt){
#endif
  if (myRank == 0)  printf("unit testing parallel TSQR with m=%d b=%d, Q construction...\n",m,b);
  double *A,*R,*swork,*tree_tau,*tau;
  double *whole_A,*collect_Q,*Q,*Qwork,*tau_tree;
  double norm;
  int i,j,row,col,info,bmb,mb,mbi,offset,pass,ioffset;

#ifdef READ_FILE
  int m,b;
  read_matrix(filename, myRank, numPes, whole_A, m, b, A, mb);
  assert(m%numPes == 0);
#else
  int seed_offset = 99900;
  mb = (m+root*b)/numPes;
  bmb = (m+root*b)/numPes;
  if (myRank < root){
    offset = (numPes-root)*mb+myRank*(mb-b);
    mb-=b;
  } else
    offset = (myRank-root)*mb;
  assert(m%numPes == 0);

  assert(0==(posix_memalign((void**)&whole_A,
          ALIGN_BYTES,
          m*m*sizeof(double))));
  assert(0==(posix_memalign((void**)&A,
          ALIGN_BYTES,
          bmb*b*sizeof(double))));
  for (col=0; col<b; col++){
    for (row=0; row<mb; row++){
      srand48(seed_offset + ((offset + row)+(col*m))*61);
      A[row+col*mb] = drand48();
      /*      if (myRank*mb+row == col)
              A[row+col*mb] += 2*m;*/
    }
  }
  for (col=0; col<b; col++){
    for (row=0; row<m; row++){
      srand48(seed_offset + (row + col*m)*61);
      whole_A[row+col*m] = drand48();
      /*      if (row == col)
              whole_A[row+col*m] += 2*m;*/
    }
  }


#endif
  assert(0==(posix_memalign((void**)&R,
          ALIGN_BYTES,
          b*b*sizeof(double))));
  assert(0==(posix_memalign((void**)&swork,
          ALIGN_BYTES,
          m*b*sizeof(double))));
  assert(0==(posix_memalign((void**)&tau,
          ALIGN_BYTES,
          b*sizeof(double))));
  assert(0==(posix_memalign((void**)&tree_tau,
          ALIGN_BYTES,
          b*sizeof(double))));
  assert(0==(posix_memalign((void**)&Q,
          ALIGN_BYTES,
          m*m*sizeof(double))));
  assert(0==(posix_memalign((void**)&tau_tree,
          ALIGN_BYTES,
          m*b*sizeof(double))));

  if (myRank == root){
    assert(0==(posix_memalign((void**)&collect_Q,
            ALIGN_BYTES,
            numPes*numPes*bmb*bmb*sizeof(double))));
    assert(0==(posix_memalign((void**)&Qwork,
            ALIGN_BYTES,
            m*m*sizeof(double))));
  }



  //get tournament R
  bitree_tsqr(A,mb,R,tree_tau,m,b,myRank,numPes,root,0,cdt,1,tau_tree);
  MPI_Gather(A, bmb*b, MPI_DOUBLE, collect_Q, bmb*b, MPI_DOUBLE, root, cdt.cm);
  if (myRank == root){
    for (i=0; i<numPes; i++){
      mbi = (m+root*b)/numPes;
      if (i < root){
        ioffset = (numPes-root)*mbi+i*(mbi-b);
        mbi-=b;
      } else
        ioffset = (i-root)*mbi;

      lda_cpy(mbi, b, mbi, m, collect_Q+bmb*b*i, Q+ioffset);
    }

    if(m<=10){
      printf("whole_YR is:\n");
      print_matrix(Q,m,b);
    }
    printf("-------------------------------\n");
  }

  //construct Q by applying the tournament tree to the indentity
  construct_Q1(A,mb, tree_tau, Q, mb, m, b, m, myRank, numPes, root, cdt, tau_tree);
/*  if(m<=10){
    printf("[%d] my Q is:\n",myRank);
    print_matrix(Q,mb,m);
  }*/

  //Gather pieces of Q to form whole_Q
  MPI_Gather(Q, bmb*m, MPI_DOUBLE, collect_Q, bmb*m, MPI_DOUBLE, root, cdt.cm);


  if (myRank == root){
    for (i=0; i<numPes; i++){
      mbi = (m+root*b)/numPes;
      if (i < root){
        ioffset = (numPes-root)*mbi+i*(mbi-b);
        mbi-=b;
      } else
        ioffset = (i-root)*mbi;

      lda_cpy(mbi, m, mbi, m, collect_Q+bmb*m*i, Q+ioffset);
    }

    if(m<=10){
      printf("whole_Q is:\n");
      print_matrix(Q,m,m);
    }
    printf("-------------------------------\n");

    //compute QQ^T - I
    std::fill(Qwork,Qwork+m*m,0.0);
    for(i=0;i<m;i++)
      Qwork[i+i*m] =-1.0;

    cdgemm('N','T',m,m,m,1.0,Q,m, Q,m,1.0,Qwork,m);
    //get the norm
    norm = cdlange('F',m,m, Qwork,m,NULL);
    pass = 0;
    if(norm <= 1.E-6)
      pass=1;
    
    printf("||QQ^T - I|| = %.6E\n",norm);

    //compute QR - A
    std::fill(Qwork,Qwork+m*m,0.0);
    lda_cpy(m,b,m,m,whole_A,Qwork);
    cdgemm('N','N',m,b,b,1.0,Q,m, R,b,-1.0,Qwork,m);
    //get the norm
    norm = cdlange('F',m,b, Qwork,m,NULL);
    if(norm <= 1.E-6 && pass)
      pass=1;

    printf("TSQR ||QR - A|| = %.6E\n",norm);

    cdgeqrf(m,b,whole_A,m,tau,swork,m*b,&info);
    if(m<=10){
      printf("correct whole_YR is:\n");
      print_matrix(whole_A,m,b);
    }
    cdorgqr( m, m, b, whole_A, m, tau, swork, m*b, &info );
    if(m<=10){
      printf("correct whole_Q is:\n");
      print_matrix(whole_A,m,m);
    }
  }
  MPI_Barrier(cdt.cm);

  free(A);
  free(tau_tree);
  free(R);
  free(Q);
  if (myRank == root){
    free(collect_Q);
    free(Qwork);
  }
  free(swork);
}

int main(int argc, char **argv) {
  int myRank, numPes, m, b, root;

  CommData_t cdt_glb;
  INIT_COMM(numPes, myRank, 1, cdt_glb);

#ifdef READ_FILE
  string filename;
  if (argc != 2 ) 
    printf("Usage: mpirun -np <num procs> ./exe <matrix file>\n");

  if (argc == 2) {
    filename.append(argv[1]);
  }
  root =0;
  par_tsqr_unit_test(filename.c_str(), root, myRank, numPes, cdt_glb);
#else
  if (argc == 2 && argc > 5) 
    printf("Usage: mpirun -np <num procs> ./exe <number of rows> <number of columns>\n");

  if (argc == 1) {
    b = 16;
    m = 32*numPes;
    root = 0;
  }
  if (argc >= 3) {
    m = atoi(argv[1]);
    b = atoi(argv[2]);
    if (argc>3) root = atoi(argv[3]);
    else root = 0;
    assert(m > 0);
    assert(b > 0);
    assert(root >= 0 && root <= numPes);
    assert((m+b*root) % numPes == 0);
    assert((m+b*root) / numPes >= b);
  }
  par_tsqr_unit_test(m, b, root, myRank, numPes, cdt_glb);
#endif


  COMM_EXIT;

  return 0;
}
