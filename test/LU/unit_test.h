#ifndef __UNIT_TEST_H__
#define __UNIT_TEST_H__

/* confirms LU factorization of a matrix
 * with arbitrary pivoting
 * assumes the matrix was creating with 
 * A[row,col] = rand48() with seed48(seed+col*dim + row) */
void pvt_con_lu(int const nrows,
    int const ncols,
    int const seed,
    double const* LU,
    int const*  P);

/* confirms LU factorization of a matrix
 * by computing the backward error norm
 * assumes the matrix was creating with 
 * A[i,j] = rand48() with seed48(seed+i*dim + j) */
void backerr_lu(int const nrows,
    int const ncols,
    int const seed,
    double const* LU,
    int const*  P,
    double*   frb_norm,
    double*   max_norm);

void backerr_lu(int const nrows,
    int const ncols,
    int const seed,
    double const* LU,
    int const*  P,
    double*   frb_norm,
    double*   max_norm,
    int const row_st,
    int const col_st,
    int const num_row_chk,
    int const num_col_chk);

/* test sequetnial tournament pivoting
 * b is the size of the panel */
void seq_tnmt_unit_test(int b);

/* test parallel tournament pivoting
 * b is the size of the panel */
void par_tnmt_unit_test(int b, int myRank, int numPes, int req_id, CommData cdt);

/* test parallel tournament pivoting parallel swap function */
void par_pivot_unit_test(int  b_sm, 
       int  mat_dim,
       int  myRank, 
       int  numPes, 
       int  req_id, 
       CommData cdt);

/* test parallel tournament pivoting
 * n is the test matrix dimension 
 * b_sm is the small block dimension 
 * b_lrg is the large block dimension */
void lu_25d_unit_test(int const     n,
                      int const   b_sm, 
                      int const   b_lrg, 
                      int const   myRank, 
                      int const   numPes, 
                      int const   c_rep,
                      CommData const  cdt);

void lu_25d_pvt_unit_test( int const    n,
                           int const    b_sm, 
                           int const    b_lrg, 
                           int const    myRank, 
                           int const    numPes, 
                           int const    c_rep,
                           CommData const   cdt);
  


#endif //__UNIT_TEST_H__

