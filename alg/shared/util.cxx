/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
#include <stdio.h>
#include <vector>
#include <string>
#include <ios>
#include <fstream>
#include <sstream>


#include "string.h"
#include "assert.h"
#include "util.h"
#include "stdlib.h"
long_int total_flop_count = 0;

void flops_add(long_int n){
  total_flop_count+=n;
}

long_int get_flops(){
  return total_flop_count;
}



/**
 * \brief prints matrix in 2D
 * \param[in] M matrix
 * \param[in] n number of rows
 * \param[in] m number of columns
 */
void print_matrix(double const * M, int n, int m){
  int i,j;
  for (i = 0; i < n; i++){
    for (j = 0; j < m; j++){
      printf("%+.4lf ", M[i+j*n]);
    }
    printf("\n");
  }
}

void print_matrix(double const *M, int n, int m, int lda){
  int i,j;
  for (i = 0; i < n; i++){
    for (j = 0; j < m; j++){
      printf("%+.4lf ", M[i+j*lda]);
    }
    printf("\n");
  }
}

/* abomination */
double util_dabs(double x){
  if (x >= 0.0) return x;
  return -x;
}

/**
 * \brief computes the size of a tensor in NOT HOLLOW packed symmetric layout
 * \param[in] ndim tensor dimension
 * \param[in] len tensor edge _elngths
 * \param[in] sym tensor symmetries
 * \return size of tensor in packed layout
 */
long_int sy_packed_size(int ndim, int* len, int* sym){
  int i, k, mp;
  long_int size, tmp;

  if (ndim == 0) return 1;

  k = 1;
  tmp = 1;
  size = 1;
  if (ndim > 0)
    mp = len[0];
  else
    mp = 1;
  for (i = 0;i < ndim;i++){
    tmp = (tmp * mp) / k;
    k++;
    mp ++;
    
    if (sym[i] == 0){
      size *= tmp;
      k = 1;
      tmp = 1;
      if (i < ndim - 1) mp = len[i + 1];
    }
  }
  size *= tmp;

  return size;
}




/**
 * \brief computes the size of a tensor in packed symmetric layout
 * \param[in] ndim tensor dimension
 * \param[in] len tensor edge _elngths
 * \param[in] sym tensor symmetries
 * \return size of tensor in packed layout
 */
long_int packed_size(int ndim, int* len, int* sym){

  int i, k, mp;
  long_int size, tmp;

  if (ndim == 0) return 1;

  k = 1;
  tmp = 1;
  size = 1;
  if (ndim > 0)
    mp = len[0];
  else
    mp = 1;
  for (i = 0;i < ndim;i++){
    tmp = (tmp * mp) / k;
    k++;
    if (sym[i] != 1)
      mp--;
    else
      mp ++;
    
    if (sym[i] == 0){
      size *= tmp;
      k = 1;
      tmp = 1;
      if (i < ndim - 1) mp = len[i + 1];
    }
  }
  size *= tmp;

  return size;
}

/**
 * \brief computes the size of a tensor in packed symmetric layout
 * \param[in] n a positive number
 * \param[out] nfactor number of factors in n
 * \param[out] factor array of length nfactor, corresponding to factorization of n
 */
void factorize(int n, int *nfactor, int **factor){
  int tmp, nf, i;
  int * ff;
  tmp = n;
  nf = 0;
  while (tmp > 1){
    for (i=2; i<=n; i++){
      if (tmp % i == 0){
        nf++;
        tmp = tmp/i;
        break;
      }
    }
  }
  if (nf == 0){
    *nfactor = nf;
  } else {
    ff  = (int*)malloc(sizeof(int)*nf);
    tmp = n;
    nf = 0;
    while (tmp > 1){
      for (i=2; i<=n; i++){
        if (tmp % i == 0){
          ff[nf] = i;
          nf++;
          tmp = tmp/i;
          break;
        }
      }
    }
    *factor = ff;
    *nfactor = nf;
  }
}

unsigned int FileRead( std::istream & is, std::vector <char> & buff ) {
  is.read( &buff[0], buff.size() );
  return is.gcount();
}

unsigned int CountLines( const std::vector <char> & buff, int sz ) {
  int newlines = 0;
  const char * p = &buff[0];
  for ( int i = 0; i < sz; i++ ) {
    if ( p[i] == '\n' ) {
      newlines++;
    }
  }
  return newlines;
}

void read_matrix(const char * filename, int myRank, int numPes, double * & whole_A, int & m, int &b, double * & A, int & mb){
  const int SZ = 1024 * 1024;
  std::vector <char> buff( SZ );
  std::ifstream file(filename);
  double cur_val;
  
  mb = m = b = 0;

  //now parse the file to get the number of rows
  while( int cc = FileRead( file, buff ) ) {
    m += CountLines( buff, cc );
  }
  assert(m>=numPes);
  
  file.clear() ;
  file.seekg(0, std::ios::beg) ;
  std::string line;
  //first parse the first line to get the number of column
  getline(file,line);
  {
    std::stringstream ss(line);
    while( ss >> cur_val ){ b++; }
  }


  mb = m / numPes;

    assert(m > 0);
    assert(b > 0);
    assert(m % numPes == 0);
    assert(m / numPes >= b);

  file.clear() ;
  file.seekg(0, std::ios::beg) ;

  //allocate 
  assert(0==(posix_memalign((void**)&whole_A,
          ALIGN_BYTES,
          m*b*sizeof(double))));
    
  for(int row =0;row<m;row++){
    for(int col =0;col<b;col++){
      file >> cur_val;
      whole_A[row + col*m] = cur_val; 
    }
  }
  //close the file
  file.close();

  assert(0==(posix_memalign((void**)&A,
          ALIGN_BYTES,
          mb*b*sizeof(double))));
  lda_cpy(mb,b,m,mb,whole_A + myRank*mb, A);

}


/**
 * \brief copy upper-triangular matrix from one rectangular buffer to another
 *
 * \param[in] R_in buffer that contains initial upper-triangular matrix
 * \param[in] R_out buffer to which the upper-triangular matrix is copied
 * \param[in] b dimension of R
 * \param[in] lda_in number of rows in the buffer R_in
 * \param[in] lda_out number of rows in the buffer R_out
 * \param[in] zero_square whether to zero out the rest of the square buffer
 **/
void copy_upper(double const *  R_in,
                double *        R_out,
                int const       b,
                int const       lda_in,
                int const       lda_out,
                int const       zero_square){
  int i;

  for (i=0; i<b; i++){
    // copy into upper triangle
    memcpy(R_out + lda_out*i, R_in + lda_in*i, (i+1)*sizeof(double));
    // zero out lower triangle if wanted
    if (zero_square){
      std::fill(R_out + lda_out*i + i+1, R_out + lda_out*i + b, 0.0);
    }
  }
}

/**
 * \brief copy lower-triangular matrix from one rectangular buffer to another
 *
 * \param[in] R_in buffer that contains initial lower-triangular matrix
 * \param[in] R_out buffer to which the lower-triangular matrix is copied
 * \param[in] b number of columns of R
 * \param[in] r number of rows of R
 * \param[in] lda_in number of rows in the buffer R_in
 * \param[in] lda_out number of rows in the buffer R_out
 * \param[in] zero_square whether to zero out the rest of the square buffer
 **/
void copy_lower(double const *  R_in,
                double *        R_out,
                int const       b,
                int const       r,
                int const       lda_in,
                int const       lda_out,
                int const       zero_square){
  int i;

  for (i=0; i<b; i++){
    // copy into upper triangle
    memcpy(R_out + lda_out*i +i+1, R_in + lda_in*i +i+1, (r-i-1)*sizeof(double));
    // zero out lower triangle if wanted
    if (zero_square){
//      std::fill(R_out + lda_out*i + i+1, R_out + lda_out*i + b, 0.0);
      std::fill( R_out + lda_out*i, R_out + lda_out*i + i, 0.0);
    }
  }
}

/**
 * \brief pack lower-triangular matrix from one rectangular buffer to 
 *        a triangular one
 *
 * \param[in] R_in buffer that contains initial lower-triangular matrix
 * \param[out] R_out buffer to which the lower-triangular matrix is copied
 * \param[in] b number of columns of R
 * \param[in] lda_in number of rows in the buffer R_in
 * \param[in] has_diag if 1 then diagonal explicitly stored
 **/
void pack_lower(double const *  R_in,
                double *        R_out,
                int const       b,
                int const       lda_in,
                int const       has_diag){
  int i, r_ctr;

  r_ctr = 0;
  for (i=0; i<b; i++){
    // pack lower triangle
    memcpy(R_out + r_ctr, R_in + lda_in*i +i+1-has_diag, (b-i-1+has_diag)*sizeof(double));
    r_ctr = r_ctr+b-i-1+has_diag;
  }
}

/**
 * \brief pack upper-triangular matrix from one rectangular buffer to 
 *        a triangular one
 *
 * \param[in] R_in buffer that contains initial upper-triangular matrix
 * \param[out] R_out buffer to which the upper-triangular matrix is copied
 * \param[in] b number of columns of R
 * \param[in] lda_in number of rows in the buffer R_in
 **/
void pack_upper(double const *  R_in,
                double *        R_out,
                int const       b,
                int const       lda_in){
  int i, r_ctr;

  r_ctr = 0;
  for (i=0; i<b; i++){
    // pack upper triangle
    memcpy(R_out + r_ctr, R_in + lda_in*i, (i+1)*sizeof(double));
    r_ctr = r_ctr+i+1;
  }
}

/**
 * \brief unpack lower-triangular matrix from one rectangular buffer to 
 *        a triangular one
 *
 * \param[in] R_in buffer that contains initial lower-triangular matrix
 * \param[out] R_out buffer to which the lower-triangular matrix is copied
 * \param[in] b number of columns of R
 * \param[in] lda_out number of rows in the buffer R_out
 * \param[in] has_diag if 1 then diagonal explicitly stored
 **/
void unpack_lower(double const *  R_in,
                  double *        R_out,
                  int const       b,
                  int const       lda_out,
                  int const       has_diag){
  int i, r_ctr;

  r_ctr = 0;
  for (i=0; i<b; i++){
    // unpack into lower triangle
    memcpy(R_out + lda_out*i+i+(1-has_diag), R_in + r_ctr, (b-i-1+has_diag)*sizeof(double));
    r_ctr = r_ctr+b-i-1+has_diag;
  }
}

/**
 * \brief unpack upper-triangular matrix from one rectangular buffer to 
 *        a triangular one
 *
 * \param[in] R_in buffer that contains initial upper-triangular matrix
 * \param[out] R_out buffer to which the upper-triangular matrix is copied
 * \param[in] b number of columns of R
 * \param[in] lda_out number of rows in the buffer R_out
 **/
void unpack_upper(double const *  R_in,
                  double *        R_out,
                  int const       b,
                  int const       lda_out){
  int i, r_ctr;

  r_ctr = 0;
  for (i=0; i<b; i++){
    // unpack into upper triangle
    memcpy(R_out + lda_out*i, R_in + r_ctr, (i+1)*sizeof(double));
    r_ctr = r_ctr+i+1;
  }
}

/**
 * \brief sets a submatrix to zero
 *
 * \param[in] nrow number of rows to set to zero
 * \param[in] ncol number of columns to set to zero
 * \param[in] lda leading dimension length of matrix
 * \param[in] A matrix 
 */
void lda_zero(const int nrow,  const int ncol, 
              const int lda,   double * A){
  int i,j;
  for (i=0; i<ncol; i++){
    for (j=0; j<nrow; j++){
      A[i*lda+j] = 0.0;
    }
  }
}

/**
 * \brief reconstructs TAU values based on Householder vectors
 *
 * \param[in] strucuture: 'R' - rect, 'L' - lower-triangular, 
 *                        'U' - upper_triangular
 * \param[in] ncol number of Householder vectors
 * \param[in] nrow length of longest HH vector
 * \param[in] lda of matrix contaaining HH vectors
 * \param[in] Y matrix contaaining HH vectors
 * \param[out] tau nrow TAU values (preallocated)
 */
void tau_recon(const char     structure,
               const int      ncol,
               const int      nrow,
               const int      lda,
               double const * Y,
               double *       tau){
  int i;
  if (structure == 'R'){
    for (i=0; i<ncol; i++){
      tau[i] = cdlange('F',nrow,1,Y+lda*i,lda,NULL);
      tau[i] = 2.0 / ((tau[i]*tau[i])+1.0);
      tau[i] = (tau[i]==2.0)?0.0:tau[i];
    }
  }
  if (structure == 'L'){
    for (i=0; i<ncol; i++){
      tau[i] = cdlange('F',nrow-i-1,1,Y+lda*i+i+1,lda,NULL);
      tau[i] = 2.0 / ((tau[i]*tau[i])+1.0);
      tau[i] = (tau[i]==2.0)?0.0:tau[i];
    }
  }
  if (structure == 'U'){
    for (i=0; i<ncol; i++){
      tau[i] = cdlange('F',i+1,1,Y+lda*i,lda,NULL);
      tau[i] = 2.0 / ((tau[i]*tau[i])+1.0);
      tau[i] = (tau[i]==2.0)?0.0:tau[i];
    }
  }
}


void init_dist_sym_matrix(int n, 
                          int ipr,
                          int pr,
                          int ipc,
                          int pc,
                          int b,
                          double * full_A, 
                          double * loc_A){
  int loc_off, i ,j;
  loc_off = 0;
  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      if (i<=j){
        full_A[i*n+j] = drand48();
      } else { 
        full_A[i*n+j] = full_A[j*n+i];
      }
      if ((i/b)%pc == ipc && (j/b)%pr == ipr){
        loc_A[loc_off] = full_A[i*n+j];
        loc_off++;
      }
    }
  }
}



void print_dist_mat(int      nrow,
                    int      ncol, 
                    int      b,
                    int      ipr,
                    int      pr,
                    int      rpr,
                    int      ipc,
                    int      pc,
                    int      rpc,
                    MPI_Comm comm,
                    double * A,
                    int64_t  lda_A){
  int li, lj, i ,j;
  double * full_A = (double*)malloc(nrow*ncol*sizeof(double));
  std::fill(full_A, full_A+nrow*ncol, 0.0);
  li=0;
  for (i=0; i<nrow; i++){
    lj=0;
    if ((i/b+pr+rpr)%pr == ipr){
      for (j=0; j<ncol; j++){
        if ((j/b+pc+rpc)%pc == ipc){ 
          printf("[%d,%d] printing A[%d,%d] pr=%d rpr=%d ipr=%d\n",ipr,ipc,i,j,pr,rpr,ipr);
          full_A[j*nrow+i] = A[lj*lda_A+li];
          lj++;
        }
      }
      li++;
    }
  }
  if (ipr+ipc != 0){
    MPI_Reduce(full_A, NULL, nrow*ncol, MPI_DOUBLE, MPI_SUM, 0, comm);
  } else {
    MPI_Reduce(MPI_IN_PLACE, full_A, nrow*ncol, MPI_DOUBLE, MPI_SUM, 0, comm);
    print_matrix(full_A,nrow,ncol);
  }
  free(full_A);
}
