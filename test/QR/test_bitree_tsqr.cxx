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

#define ERROR_CHECK

#define EL(A,LDA,I,J)\
  &(A)[(J)*(LDA)+(I)]

using namespace std;

#ifdef ERROR_CHECK
void get_tsqr_Q(double * Q, double * Qwork, double* whole_YR, double * tau, int const m, int const b, double * swork,int const root,int const pe_st, int const numPes,double* whole_A,int verbose){
  int i;
  int mb = m/numPes;

  double * updated_A,*Awork, * T_sqr;


  assert(0==(posix_memalign((void**)&Awork,
          ALIGN_BYTES,
          m*m*sizeof(double))));
  
  assert(0==(posix_memalign((void**)&T_sqr,
          ALIGN_BYTES,
          b*b*sizeof(double))));

  if(verbose){
    assert(0==(posix_memalign((void**)&updated_A,
            ALIGN_BYTES,
            m*b*sizeof(double))));
    lda_cpy(m,b,m,m,whole_A,updated_A);
  }

  //Compute first Qs of the leaves
  for(i=0;i<m;i++)
    Q[i+i*m] =1.0;

  for(int myr= 0; myr<numPes;myr++){
    //recompute TAU
    for(i=0;i<b;i++){
          tau[i] = cdlange('F',mb-i-1,1, EL(whole_YR,m,mb*myr+i+1,i),m,NULL);
          tau[i] = 2.0 / (pow(tau[i],2.0)+1.0);
          tau[i] = (tau[i]==2.0)?0.0:tau[i];
    }

        lda_cpy(mb,b,m,m,EL(whole_YR,m,mb*myr,0),EL(Q,m,mb*myr,mb*myr));
        cdorgqr( mb, mb, b, EL(Q,m,mb*myr,mb*myr), m, tau, Qwork, m, &i );

  }

  if(verbose){
    //try Q^T*A = R
    cdgemm('T','N',m,b,m,1.0,Q,m,updated_A,m,0.0,Qwork,m);
    lda_cpy(m,b,m,m,Qwork,updated_A);
    printf("Grad Updated A After is:\n");
    print_matrix(updated_A,m,b);
  }

  //Now we need to compute Q by going through the tree again
  for (int np = numPes; np > 1; np = np/2+(np%2)){
    for(int myr= 0; myr<numPes;myr++){
      /* If I am in the first half of processor list, I dont have Y/Y' */
      if (myr < np/2 && myr >= 0) {
        int comm_pe = myr+(np+1)/2;
        comm_pe = myr+(np+1)/2;
        comm_pe = comm_pe + root;
        if (comm_pe >= numPes)
          comm_pe = comm_pe - numPes;


        std::fill(swork,swork+2*b*b,0.0);
        for(i=0;i<b;i++)
          swork[i+i*2*b] =1;

        char uplo = 'U';
        int db = 2*b;
        cdlacpy( uplo, b, b, EL(whole_YR,m,mb*comm_pe,0) , m, swork+b, db );

        if(verbose){
          printf("Swork|Y is:\n");
          print_matrix(swork,2*b,b);
        }

        for(i=0;i<b;i++){
          tau[i] = cdlange('F',b,1, EL(swork,2*b,b,i),m,NULL);
          tau[i] = 2.0 / (pow(tau[i],2.0)+1.0);
          tau[i] = (tau[i]==2.0)?0.0:tau[i];
        }

        if(verbose){
          printf("elim(%d,%d) TAU\n",myr,comm_pe);
          print_matrix(tau,1,b);
        }
        cdorgqr( 2*b, 2*b, b, swork, 2*b, tau, Qwork, m, &i );

        //Get small Q
/*        unpack_upper(tree_data, T_sqr, b, b);
        tree_data += upr_sz(b);
        unpack_upper(tree_data, swork+b, b, 2*b);
        cdorgqr( 2*b, 2*b, b, swork, 2*b, T_sqr:, Qwork, m, &i )*/

        if(verbose){
          printf("Compact Q is:\n");
          print_matrix(swork,2*b,2*b);
          
          //check : compute the update compact A
          lda_cpy(b,b,m,2*b,EL(updated_A,m,mb*myr,0),EL(Qwork,2*b,0,0));
          lda_cpy(b,b,m,2*b,EL(updated_A,m,mb*comm_pe,0),EL(Qwork,2*b,b,0));

          printf("Grad Up Compact A is:\n");
          print_matrix(Qwork,2*b,b);

          cdgemm('T','N',2*b,b,2*b,1.0, swork,2*b,Qwork,2*b,0.0,Awork,2*b);

          printf("Grad Up compact A After is:\n");
          print_matrix(Awork,2*b,b);


          lda_cpy(b,b,2*b,m,EL(Awork,m,0,0),EL(updated_A,m,mb*myr,0));
          lda_cpy(b,b,2*b,m,EL(Awork,m,b,0),EL(updated_A,m,mb*comm_pe,0));

          printf("Grad Up A After is:\n");
          print_matrix(updated_A,m,b);
        }



        //Multiply the Q's

        //          cdgemm('N','N',mb,b,b,1.0, EL(Q,m,mb*myr,mb*myr),m,swork,2*b,0.0,EL(Qwork,m,mb*myr,mb*myr),m);
        //          cdgemm('N','N',mb,b,b,1.0, EL(Q,m,mb*comm_pe,mb*comm_pe),m,swork+b+2*b*b,2*b,0.0,EL(Qwork,m,mb*comm_pe,mb*comm_pe),m);
        //
                  //could be done in place ?
        //          cdgemm('N','N',mb,b,b,1.0, EL(Q,m,mb*myr,mb*myr),m,swork+2*b*b,2*b,0.0,EL(Q,m,mb*myr,mb*comm_pe),m);
        //          cdgemm('N','N',mb,b,b,1.0, EL(Q,m,mb*comm_pe,mb*comm_pe),m,swork+b,2*b,0.0,EL(Q,m,mb*comm_pe,mb*myr),m);
        //
        //
        //          //now copy back to Q
        //          lda_cpy(mb,b,m,m,EL(Qwork,m,mb*myr,mb*myr),EL(Q,m,mb*myr,mb*myr));
        //          lda_cpy(mb,b,m,m,EL(Qwork,m,mb*comm_pe,mb*comm_pe),EL(Q,m,mb*comm_pe,mb*comm_pe));



        //form the sparse Q instead of compact
        std::fill(Qwork,Qwork+m*m,0.0);
        for(i=0;i<m;i++)
          Qwork[i+i*m] =1;

        lda_cpy(b,b,2*b,m,EL(swork,2*b,0,0),EL(Qwork,m,mb*myr,mb*myr));
        lda_cpy(b,b,2*b,m,EL(swork,2*b,b,b),EL(Qwork,m,mb*comm_pe,mb*comm_pe));
        lda_cpy(b,b,2*b,m,EL(swork,2*b,0,b),EL(Qwork,m,mb*myr,mb*comm_pe));
        lda_cpy(b,b,2*b,m,EL(swork,2*b,b,0),EL(Qwork,m,mb*comm_pe,mb*myr));


        if(verbose){
          //try to apply the update
          cdgemm('T','N',m,b,m,1.0,Q,m,whole_A,m,0.0,Awork,m);
          cdgemm('T','N',m,b,m,1.0,Qwork,m,whole_A,m,0.0,Awork,m);
          printf("(1) A After is:\n");
          print_matrix(Awork,m,b);
        }

        //compute new Q
        cdgemm('N','N',m,m,m,1.0,Q,m,Qwork,m,0.0,Awork,m);
        lda_cpy(m,m,m,m,Awork,Q);
        if(verbose){
          //Apply it again
          cdgemm('T','N',m,b,m,1.0,Q,m,whole_A,m,0.0,Awork,m);
          printf("(2) A After is:\n");
          print_matrix(Awork,m,b);
        }
      }
      if(verbose){
        printf("-------------------------------\n");
      }
    }
    if(verbose){
      printf("*******************************\n");
    }
  }

  if(verbose){
    printf("A before is:\n");
    print_matrix(whole_A,m,b);
    cdgemm('T','N',m,b,m,1.0,Q,m,whole_A,m,0.0,Qwork,m);
    printf("A After is:\n");
    print_matrix(Qwork,m,b);

    free(updated_A);
  }
  free(Awork);

}




#endif

/**
 * \brief Test TSQR 
 *
 * \param[in] m number of rows in A
 * \param[in] b number of columns in A
 * \param[in] root is the root process that owns A
 * \param[in] myRank rank in communicator column
 * \param[in] numPes number of processes in column
 * \param[in] req_id request id to use for send/recv
 * \param[in] comm MPI communicator for column
 **/

#ifdef READ_FILE
void par_tsqr_unit_test(const char * filename, 
                        int const root, 
                        int const myRank, 
                        int const numPes, 
                        int const req_id, 
                        CommData_t  cdt){
#else
void par_tsqr_unit_test(int const m,
                        int const b, 
                        int const root, 
                        int const myRank, 
                        int const numPes, 
                        int const req_id, 
                        CommData_t  cdt){
#endif
  if (myRank == 0)  printf("unit testing parallel TSQR...\n");
  double *A,*R,*whole_A,*swork,*tau,*R_ans,*tree_tau;
  double *whole_YR,*collect_YR,*Q,*Qwork;
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
          m*b*sizeof(double))));
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
  assert(0==(posix_memalign((void**)&R_ans,
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

#ifdef ERROR_CHECK
  if (myRank == root){
    assert(0==(posix_memalign((void**)&whole_YR,
            ALIGN_BYTES,
            m*b*sizeof(double))));
    assert(0==(posix_memalign((void**)&collect_YR,
            ALIGN_BYTES,
            m*b*sizeof(double))));
    assert(0==(posix_memalign((void**)&Q,
            ALIGN_BYTES,
            m*m*sizeof(double))));
    assert(0==(posix_memalign((void**)&Qwork,
            ALIGN_BYTES,
            m*m*sizeof(double))));
  }
#endif



  //get tournament R
  bitree_tsqr(A,mb,R,tree_tau,m,b,myRank,numPes,root,req_id,cdt,1);


#ifdef ERROR_CHECK
  //Gather pieces of A to form whole_Y
  MPI_Gather(A, bmb*b, MPI_DOUBLE, collect_YR, bmb*b, MPI_DOUBLE, root, cdt.cm);
#endif




  if (myRank == root){
    pass = 1;
#ifdef ERROR_CHECK
    for (i=0; i<numPes; i++){
      mbi = (m+root*b)/numPes;
      if (i < root){
        ioffset = (numPes-root)*mbi+i*(mbi-b);
        mbi-=b;
      } else
        ioffset = (i-root)*mbi;

      lda_cpy(mbi, b, mbi, m, collect_YR+bmb*b*i, whole_YR+ioffset);
    }

    if(m<=10){
      printf("whole_YR is:\n");
      print_matrix(whole_YR,m,b);
    }
    printf("-------------------------------\n");

    get_tsqr_Q(Q, Qwork, whole_YR,tau,m,b,swork,root,0, numPes,whole_A,0);
/*    memcpy(Q, whole_YR, m*b*sizeof(double));
    lda_cpy(m,b,m,m,whole_YR,Q);
    cdorgqr( m, m, b, Q, m, tree_tau, swork, m*b, &info );*/

    if(m<=10){
        printf("TSQR Q is:\n");
        print_matrix(Q,m,m);
        printf("-------------------------------\n");
        printf("TSQR R is:\n");
        print_matrix(R,b,b);
        printf("-------------------------------\n");
    }

    //compute QQ^T - I
    std::fill(Qwork,Qwork+m*m,0.0);
    for(i=0;i<m;i++)
      Qwork[i+i*m] =-1.0;

    cdgemm('N','T',m,m,m,1.0,Q,m, Q,m,1.0,Qwork,m);
    //get the norm
    tau[0] = cdlange('F',m,m, Qwork,m,NULL);
    if(tau[0] <= 1.E-6)
      pass=1;
    else
      pass=0;
    
    printf("||QQ^T - I|| = %.6E\n",tau[0]);

    //compute QR - A
    std::fill(Qwork,Qwork+m*m,0.0);
    lda_cpy(m,b,m,m,whole_A,Qwork);
    cdgemm('N','N',m,b,b,1.0,Q,m, R,b,-1.0,Qwork,m);
    //get the norm
    tau[0] = cdlange('F',m,b, Qwork,m,NULL);
    if(tau[0] <= 1.E-6 && pass)
      pass=1;
    else
      pass=0;

    printf("TSQR ||QR - A|| = %.6E\n",tau[0]);
#endif

    lda_cpy(m,b,m,m,whole_A,Qwork);

    cdgeqrf(m,b,whole_A,m,tau,swork,m*b,&info);
    if(m<=10){
      printf("-------------------------------\n");
      printf("Correct whole YR\n");
      print_matrix(whole_A, m, b);
      printf("-------------------------------\n");
    }
    assert(info == 0);
    //compute QR - A
    std::fill(R_ans, R_ans+b*b, 0.0);
    copy_upper(whole_A, R_ans, b, m, b, 0);
    if(m<=10){
      printf("Correct R is:\n");
      print_matrix(R_ans,b,b);
      printf("-------------------------------\n");
    }
    //get Q of LAPACK
    lda_cpy(m,b,m,m,whole_A,Q);
    cdorgqr( m, m, b, Q, m, tau, swork, m*b, &info );
    cdgemm('N','N',m,b,b,1.0,Q,m, R_ans,b,-1.0,Qwork,m);

    //get the norm
    tau[0] = cdlange('F',m,b, Qwork,m,NULL);
    if(tau[0] <= 1.E-6 && pass)
      pass=1;
    else
      pass=0;
    printf("LAPACK ||QR - A|| = %.6E\n",tau[0]);

    if (pass == 0){
      printf("TEST FAILED!!!\n");
    } else {
      printf("Test successful.\n");
    }


    //    print_matrix(tau, 1, b);
    //    printf("\n\n");
    //    print_matrix(whole_A, m, b);
/*    printf("\n\n");
//    print_matrix(R, b, b);
//    printf("\n\n");
//    print_matrix(R_ans, b, b);

    pass = 1;
    for (i=0; i<b*b; i++){
      if (fabs(fabs(R_ans[i])-fabs(R[i])) >= 1.E-6 ||
          fabs(fabs(R_ans[i])-fabs(R[i]))/fabs(R_ans[i]) >= 1.E-6){
//        printf("ERROR: R_ans[%d,%d] = %.2E, R_TSQR[%d,%d] = %.2E\n",
//            i%b,i/b,R_ans[i],i%b,i/b,R[i]);
        pass = 0;
      }
    }
    if (pass == 0){
      printf("TEST FAILED!!!\n");
    } else {
      printf("Test successful.\n");
    }*/



  }
  MPI_Barrier(cdt.cm);

  free(A);
  free(R);
  free(R_ans);
  free(whole_A);
#ifdef ERROR_CHECK
  if (myRank == root){
    free(whole_YR);
    free(collect_YR);
    free(Q);
    free(Qwork);
  }
#endif
  free(tau);
  free(swork);
}

int main(int argc, char **argv) {
  int myRank, numPes, m, b, root;

  CommData_t cdt_glb;
  INIT_COMM(numPes, myRank, 1, cdt_glb);


#ifdef PROFILE
  TAU_PROFILE_SET_CONTEXT(MPI_COMM_WORLD);
#endif
#ifdef READ_FILE
  string filename;
  if (argc != 2 ) 
    printf("Usage: mpirun -np <num procs> ./exe <matrix file>\n");

  if (argc == 2) {
    filename.append(argv[1]);
  }
  root =0;
  par_tsqr_unit_test(filename.c_str(), 0, myRank, numPes, 0, cdt_glb);
#else
  if (argc == 2 && argc > 5) 
    printf("Usage: mpirun -np <num procs> ./exe <number of rows> <number of columns>\n");

  if (argc == 1) {
    b = 21;
    m = 42*numPes;
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
    assert((m+root*b) % numPes == 0);
    assert((m+root*b) / numPes >= b);
  }
  par_tsqr_unit_test(m, b, root, myRank, numPes, 0, cdt_glb);
#endif


  COMM_EXIT;

  return 0;
}
