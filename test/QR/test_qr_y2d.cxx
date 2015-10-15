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
#include <iostream>

#include "CANDMC.h"

using namespace std;

/**
 * \brief Test 2D QR with Yamamoto's algorithm
 *
 * \param[in] m number of rows in A
 * \param[in] b number of columns in A
 * \param[in] myRank rank in communicator column
 * \param[in] numPes number of processes in column
 * \param[in] req_id request id to use for send/recv
 * \param[in] comm MPI communicator for column
 **/
void qr_y2d_unit_test(int64_t    m,
                      int64_t    k,
                      int64_t    b2,
                      int64_t    b,
                      int64_t    myRank,
                      int64_t    numPes,
                      int64_t    req_id,
                      CommData_t cdt_row,
                      CommData_t cdt_col,
                      CommData_t cdt){
  if (myRank == 0)  
    printf("unit testing parallel 2D QR using Yamamoto's basis kernel representation...\n");
  double *A,*whole_A,*collect_YR,*whole_YR,*swork,*tau,*bw_A;
  double norm, yn2;
  int info;
  int64_t i,j,row,col,mb,kb,pass;
  int64_t npcol, nprow, mycol, myrow;
  npcol = cdt_row.np;
  nprow = cdt_col.np;
  mycol = cdt_row.rank;
  myrow = cdt_col.rank;


  int64_t seed_offset = 99900;

/*#ifdef READ_FILE
  int64_t m,b;
  read_matrix(filename, myRank, numPes, whole_A, m, k, A, mb);
  assert(m%numPes == 0);

#else*/
  mb = m / nprow;
  kb = k / npcol;
  assert(0==(posix_memalign((void**)&whole_YR,
          ALIGN_BYTES,
          m*k*sizeof(double))));
  assert(0==(posix_memalign((void**)&collect_YR,
          ALIGN_BYTES,
          m*k*sizeof(double))));
  assert(0==(posix_memalign((void**)&whole_A,
          ALIGN_BYTES,
          m*k*sizeof(double))));
  assert(0==(posix_memalign((void**)&A,
          ALIGN_BYTES,
          mb*kb*sizeof(double))));

  for (col=0; col<kb; col++){
    for (row=0; row<mb; row++){
      srand48(seed_offset +((myrow*b + (row%b) + (row/b)*b*nprow)
                          + (mycol*b + (col%b) + (col/b)*b*npcol)*m)*61);
      A[row+col*mb] = drand48()-.5;
/*      if (myRank*mb+row == col)
        A[row+col*mb] += 2*m;*/
    }
  }
  for (col=0; col<k; col++){
    for (row=0; row<m; row++){
      srand48(seed_offset + (row + col*m)*61);
      whole_A[row+col*m] = drand48()-.5;
/*      if (row == col)
        whole_A[row+col*m] += 2*m;*/
    }
  }

  assert(0==(posix_memalign((void**)&tau,
          ALIGN_BYTES,
          MAX(k,m)*sizeof(double))));
  assert(0==(posix_memalign((void**)&swork,
          ALIGN_BYTES,
          m*k*sizeof(double))));
  assert(0==(posix_memalign((void**)&bw_A,
          ALIGN_BYTES,
          m*k*sizeof(double))));


/*  if (myRank == 0){
    printf("whole A:\n");
    print_matrix(whole_A, m, k);
  }*/
  //get YR via TSQR and reconstruction
  //QR_tree_2D(A,mb,m,k,b,myRank,numPes,0,0,cdt_row,cdt_col,cdt);
  pview pv;
  pv.rrow = 0;
  pv.rcol = 0;
  pv.crow = cdt_row;
  pv.ccol = cdt_col;
  pv.cworld = cdt;
  //QR_Yamamoto_2D(A,mb,m,k,b,&pv,NULL,0,true);
  QR_Yamamoto_2D_2D(A,mb,m,k,b2,b,&pv,NULL,true);
  /* IMPORTANT: currently need to change sign of last R entry to align with LAPACK */
  if (m==k && myRank == numPes-1){
    A[mb*kb-1] = A[mb*kb-1]*-1.0;
  }
/*  if (myRank == 0){
    printf("my YR part:\n");
    print_matrix(A, mb, kb);
  }*/
  MPI_Gather(A, mb*kb, MPI_DOUBLE,
         collect_YR, mb*kb, MPI_DOUBLE, 0, cdt.cm);
  if (myRank == 0){
    for (i=0; i<numPes; i++){
      //lda_cpy(mb, kb, mb, m, collect_YR+mb*kb*i, 
      //                       whole_YR+m*kb*(i/nprow)+mb*(i%nprow));
      for (j=0; j<kb; j++){
        lda_cpy(b, mb/b, b, b*nprow, 
                collect_YR+mb*(kb*i+j), 
                whole_YR+m*(b*(i/nprow) + (j%b)+(j/b)*(b*npcol))+b*(i%nprow));
      }
    }

    /*for (i=0; i<b; i++){
      tau[i] = 2./(1+cddot(m-i-1, whole_YR+m*i+i+1, 1, 
                                  whole_YR+m*i+i+1, 1));
//          tau[i] = cdlange('F',m-i-1,1, whole_YR + i+1 + i*m,m,NULL);
//          tau[i] = 2.0 / (pow(tau[i],2.0)+1.0);
//          tau[i] = (tau[i]==2.0)?0.0:tau[i];
    }*/
    tau_recon('L', k, m, m, whole_YR, tau);


    double * Q = (double*)malloc(MAX(m,k)*m*sizeof(double));
    double * QQT = (double*)malloc(MAX(m,k)*m*sizeof(double));
    double * tau_lapack = (double*) malloc(m*sizeof(double));


    if (m<=16){
      printf("2D QR Y/R is \n");
      print_matrix(whole_YR,m,k);
      printf("2D QR TAU is \n");
      print_matrix(tau,1,k);
    }



    //compute Q 
    /*lda_cpy(m,b,m,m,whole_A,bw_A);
    cdgeqrf(m,b,bw_A,m,tau,swork,m*b,&info);
    lda_cpy(m,b,m,m,bw_A,whole_YR);*/
    lda_cpy(m,k,m,m,whole_YR,Q);
    cdorgqr( m, m, MIN(m,k), Q, m, tau, QQT, m, &info );

    //A-QR check
    std::fill(bw_A, bw_A+m*k, 0.0);
    copy_upper(whole_YR, bw_A, k, m, m, 0);
    lda_cpy(m,k,m,m,whole_A,QQT);
    cdgemm('N','N',m,k,m,-1.0,Q,m,bw_A,m,1.0,QQT,m);

    /*for(i=0;i<b;i++){
      norm = cdlange('F',m,i+1, QQT,m,NULL);
      cout <<"TSQR   \t"<<norm<< "\t" << *(whole_YR+i+i*m) << endl;  
    }*/
    
    norm = cdlange('F',m,k, QQT,m,NULL);
    cout << "2D QR ||A - QR||_2 is "<< norm <<endl;


    //Do the same with LAPACK
    lda_cpy(m,k,m,m,whole_A,bw_A);
    cdgeqrf(m,k,bw_A,m,tau_lapack,swork,m*k,&info);


    if (m<=16){
      printf("LAPACK Y/R is \n");
      print_matrix(bw_A,m,k);
      printf("LAPACK TAU is \n");
      print_matrix(tau_lapack,1,k);
    }
    


    //compare the Ys
/*    copy_lower(whole_YR, swork, b,m, m, m, 1);
    std::fill(QQT, QQT+m*m, 0.0);
    copy_lower(bw_A, QQT, b,m, m, m, 1);
    for(i=0;i<b;i++){
      cdaxpy(m-i-1, -1.0, QQT+i+1+i*m, 1, swork+i+1+i*m, 1);
//      for(j=i+1;j<m;j++){
//        cout <<  *(swork+j+i*m) << endl;
//      }     

      norm = cdlange('F',m-i-1,1, swork + i*m,m,NULL);
      printf("||Y(:," PRId64 ") - Ylapack|| = %lE\n",i,norm);
    }

    norm = cdlange('F',m,b, swork,m,NULL);
    printf("||Y - Ylapack|| = %lE\n",norm);


    //Compare the Tau's
    cdaxpy(b, -1.0, tau_lapack, 1, tau, 1);
    cout << "-----Tau - Tau_lapack-----"<< endl;
    print_matrix(tau,b,1);
    cout << "--------------------------"<<endl;*/

    assert(info == 0);
    //get Q of LAPACK
    lda_cpy(m,k,m,m,bw_A,Q);
    cdorgqr( m, m, MIN(k,m), Q, m, tau_lapack, QQT, m, &info );
    /*printf("LAPACK first b columns of Q:\n");
    print_matrix(Q,m,b);*/
    //Get R of LAPACK
    std::fill(collect_YR, collect_YR+m*k, 0.0);
    copy_upper(bw_A, collect_YR, k, m, m, 0);
    
    lda_cpy(m,k,m,m,whole_A,QQT);
    cdgemm('N','N',m,k,m,-1.0,Q,m,collect_YR,m,1.0,QQT,m);

    /*for(i=0;i<b;i++){
      norm = cdlange('F',m,i+1, QQT,m,NULL);
      cout <<"LAPACK \t"<<norm<< "\t" << *(bw_A+i+i*m) << endl;  
    }*/

    norm = cdlange('F',m,k, QQT,m,NULL);
    cout << "LAPACK ||A - QR||_2 is "<< norm <<endl;

    /*for(i=0;i<b;i++){
      cout <<"R_LAPACK  - R_TSQR = \t"<<*(bw_A+i+i*m) - *(whole_YR+i+i*m)<< endl;  
    }*/


    free(tau_lapack);
    free(Q);
    free(QQT);


    std::fill(bw_A, bw_A+m*k, 0.0);
    copy_upper(whole_YR, bw_A, k, m, m, 0);
  

    cdormqr('L', 'N', m, k, MIN(k,m), whole_YR, m, tau, bw_A, 
            m, swork, m*k, &info);

//    print_matrix(whole_A, m, b);
    /*print_matrix(tau, 1, b);
    for (i=0; i<b; i++){
      yn2 = cddot(m-i-1, whole_A+m*i+i+1, 1, 
       whole_A+m*i+i+1, 1);
      tau[i] = 1.-(yn2-sqrt(2.-yn2))/(yn2+2);
      tau[i] = 2./(1+cddot(m-i-1, whole_A+m*i+i+1, 1, 
             whole_A+m*i+i+1, 1));
    }*/
  
    for (i=0; i<m*k; i++){
      bw_A[i] = whole_A[i] - bw_A[i];
    }
    /*print_matrix(bw_A, m, b);
    printf("\n\n");*/
    norm = 0.0;  
    for (i=0; i<k; i++){
      norm += sqrt(cddot(m, bw_A+i*m, 1, bw_A+i*m, 1)/cddot(m, whole_A+i*m, 1, whole_A+i*m, 1));
    }

    printf("||A-QR||_2 = %.2E\n", norm);
    
    if (norm > 1.E-9){
      printf("TEST FAILED!!!\n");
    } else {
      printf("Test successful.\n");
    }
  }
  MPI_Barrier(cdt.cm);

  free(A);
  free(whole_A);
  free(whole_YR);
  free(collect_YR);
  free(tau);
  free(swork);
}

int main(int argc, char **argv) {
  int myRank, numPes;
  int64_t m, k, b, b2, nprow, npcol;

  CommData_t cdt_glb;
  CommData_t cdt_row, cdt_col;
  INIT_COMM(numPes, myRank, 1, cdt_glb);

  /*string filename;
  if (argc != 2 ) 
    printf("Usage: mpirun -np <num procs> ./exe <matrix file>\n");

  if (argc == 2) {
    filename.append(argv[1]);
  }
  qr_2d_unit_test(filename.c_str(), myRank, numPes, 0, cdt_glb);
  */

  nprow = sqrt(numPes);
  if (nprow*nprow != numPes) nprow++;
  while (numPes % nprow != 0) nprow++;
  npcol = numPes/nprow;
  if (argc == 1) {
    b = 2;
    m = 8*nprow;
    k = 4*npcol;
    b2 = min(m,k); 
  } else if (argc > 3) {
    m = atoi(argv[1]);
    k = atoi(argv[2]);
    b = atoi(argv[3]);
    if (argc > 4)  nprow = atoi(argv[4]);
    if (argc > 5)  b2 = atoi(argv[5]);
    else b2 = min(m,k);
   } else {
    printf("Usage: mpirun -np <num procs> ./exe <number of rows>");
    printf(" <number of columns> <block size> <nprow> <higher level 2D block size>\n");
    ABORT;
  }
    printf("Usage: mpirun -np <num procs> ./exe <number of rows>");
    printf(" <number of columns> <block size> <nprow> <higher level 2D block size>\n");
    assert(b2%b == 0);
    assert(m > 0);
    assert(b > 0);
    assert(k > 0);
    assert(nprow > 0);
    assert(nprow <= numPes);

  npcol = numPes/nprow;
  if (myRank == 0){
    printf("m=" PRId64 ", k=" PRId64 ", b2 = " PRId64 ", b = " PRId64 ", nprow = " PRId64 ", npcol = " PRId64 "\n",
            m,k,b2,b,nprow,npcol);
  }
  SETUP_SUB_COMM(cdt_glb, cdt_row, 
                 myRank/nprow, 
                 myRank%nprow, 
                 npcol);
  SETUP_SUB_COMM(cdt_glb, cdt_col, 
                 myRank%nprow, 
                 myRank/nprow, 
                 nprow);
  
  qr_y2d_unit_test(m, k, b2, b, myRank, numPes, 0, cdt_row, cdt_col, cdt_glb);


  COMM_EXIT;
  return 0;
}
