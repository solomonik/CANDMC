#include "lapack.h"


#if !FTN_UNDERSCORE
#define UTIL_ZGEMM      zgemm
#define UTIL_DGEMM      dgemm
#define UTIL_DAXPY      daxpy
#define UTIL_ZAXPY      zaxpy
#define UTIL_DCOPY      dcopy
#define UTIL_ZCOPY      zcopy
#define UTIL_DSCAL      dscal
#define UTIL_ZSCAL      zscal
#define UTIL_DDOT       ddot
#define UTIL_DGETRF     dgetrf
#define UTIL_DGEQRF   dgeqrf
#define UTIL_ZGEQRF   zgeqrf
#define UTIL_DTRSM  dtrsm
#define UTIL_DSYRK  dsyrk
#define UTIL_DORMQR   dormqr
#define UTIL_DGER   dger
#define UTIL_DLANGE   dlange
#define UTIL_DORGQR   dorgqr
#define UTIL_DTPQRT  dtpqrt
#define UTIL_DTPMQRT  dtpmqrt
#define UTIL_DTPRFB  dtprfb
#define UTIL_DLARFB  dlarfb
#define UTIL_DLARFT  dlarft
#define UTIL_DLACPY dlacpy
#define UTIL_DTRTRI    dtrtri
#define UTIL_DGETRI dgetri
#define UTIL_DSYEVX    dsyevx
#define UTIL_DLATRD    dlatrd
#define UTIL_DSYTRD    dsytrd
#define UTIL_PDSYEVX   pdsyevx
#define UTIL_PDLATRD   pdlatrd
#define UTIL_PDSYTRD   pdsytrd
#else
#define UTIL_ZGEMM      zgemm_
#define UTIL_DGEMM      dgemm_
#define UTIL_DAXPY      daxpy_
#define UTIL_ZAXPY      zaxpy_
#define UTIL_DCOPY      dcopy_
#define UTIL_ZCOPY      zcopy_
#define UTIL_DSCAL      dscal_
#define UTIL_ZSCAL      zscal_
#define UTIL_DDOT       ddot_
#define UTIL_DGETRF     dgetrf_
#define UTIL_DTRSM      dtrsm_
#define UTIL_DSYRK  dsyrk_
#define UTIL_DGEQRF   dgeqrf_
#define UTIL_ZGEQRF   zgeqrf_
#define UTIL_DORMQR   dormqr_
#define UTIL_DGER   dger_
#define UTIL_DLANGE   dlange_
#define UTIL_DORGQR   dorgqr_
#define UTIL_DTPQRT  dtpqrt_
#define UTIL_DTPMQRT  dtpmqrt_
#define UTIL_DTPRFB  dtprfb_
#define UTIL_DLARFB  dlarfb_
#define UTIL_DLARFT  dlarft_
#define UTIL_DLACPY  dlacpy_
#define UTIL_DTRTRI    dtrtri_
#define UTIL_DGETRI dgetri_
#define UTIL_DSYEVX    dsyevx_
#define UTIL_DLATRD    dlatrd_
#define UTIL_DSYTRD    dsytrd_
#define UTIL_PDSYEVX   pdsyevx_
#define UTIL_PDLATRD   pdlatrd_
#define UTIL_PDSYTRD   pdsytrd_
#endif

#ifdef USE_JAG
extern "C"
void jag_zgemm( char *, char *,
                int *,  int *,
                int *,  double *,
                double *,       int *,
                double *,       int *,
                double *,       double *,
                                int *);

#endif

extern "C"
void UTIL_DSYRK(char const *        UPLO,
                char const *        TRANS,
                int const *         M,
                int const *         K,
                double const *      ALPHA,
                double const *      A,
                int const *         LDA,
                double const *      BETA,
                double *            C,
                int const *         LDC);



extern "C"
void UTIL_DGEMM(const char *,   const char *,
                int *,    int *,
                int *,    const double *,
                const double *, int *,
                const double *, int *,
                const double *, double *,
                                int *);

extern "C"
void UTIL_ZGEMM(const char *,   const char *,
                int *,    int *,
                int *,    const std::complex<double> *,
                const std::complex<double> *,   int *,
                const std::complex<double> *,   int *,
                const std::complex<double> *,   std::complex<double> *,
                                int *);


extern "C"
void UTIL_DAXPY(int * n,          double * dA,
                const double * dX,      int * incX,
                double * dY,            int * incY);

extern "C"
void UTIL_ZAXPY(int * n,                          std::complex<double> * dA,
                const std::complex<double> * dX,        int * incX,
                std::complex<double> * dY,              int * incY);

extern "C"
void UTIL_DCOPY(int * n,
                const double * dX,      int * incX,
                double * dY,            int * incY);

extern "C"
void UTIL_ZCOPY(int * n,
                const std::complex<double> * dX,        int * incX,
                std::complex<double> * dY,              int * incY);


extern "C"
void UTIL_DSCAL(int *n,           double *dA,
                double * dX,      int *incX);

extern "C"
void UTIL_ZSCAL(int *n,           std::complex<double> *dA,
                std::complex<double> * dX,      int *incX);

extern "C"
double UTIL_DDOT(int * n,         const double * dX,      
                 int * incX,      const double * dY,      
                 int * incY);

extern "C" 
void UTIL_DTRSM( char *,	  char *,
                 char *,	  char *,
                 int *,	    int *,
                 double *,	const double *,
                 int *,	    double *,
                 int *);

extern "C"
void UTIL_DGETRF(int *    M,    int * N,
		             double * A,		int * lda,
		             int * IPIV,		int * info);

extern "C"
void UTIL_DGEQRF(int const *  M,
                 int const *  N,
                 double * A,
                 int const *  LDA,
                 double * TAU2,
                 double * WORK,
                 int const *  LWORK,
                 int  * INFO);

extern "C"
void UTIL_ZGEQRF(int const *  M,
                 int const *  N,
                 std::complex<double> * A,
                 int const *  LDA,
                 std::complex<double> * TAU2,
                 std::complex<double> * WORK,
                 int const *  LWORK,
                 int  * INFO);


extern "C"
void UTIL_DORMQR(char const * SIDE,
                 char const * TRANS,
                 int const *  M,
                 int const *  N,
                 int const *  K,
                 double const * A,
                 int const *  LDA,
                 double const * TAU2,
                 double * C,
                 int const *  LDC,
                 double * WORK,
                 int const *  LWORK,
                 int  * INFO);

extern "C"
void UTIL_DORGQR(int const *  M,
                 int const *  N,
                 int const *  K,
                 double * A,
                 int const *  LDA,
                 double const * TAU2,
                 double * WORK,
                 int const *  LWORK,
                 int  * INFO);


extern "C"
void UTIL_DGER(int const *  M,
               int const *  N,
               double const * alpha,
               const double *   X,
               const int *  incX,
               const double *   Y,  
               const int *  incY,
               double *   A,
               const int *  lda_A);

extern "C"
double UTIL_DLANGE(char const *   NORM,
                   int const  *   M,
                   int const  *   N,
                   double const * A,
                   int const  *   LDA,
                   double *       WORK);



#if (LAPACKHASTSQR==1)
extern "C"
void UTIL_DTPQRT( int const *       m,
                  int const *       n,
                  int const *       l,
                  int const *       np,
                  double *          A,
                  int const *       lda_A,
                  double *          B,
                  int const *       lda_B,
                  double *          T,
                  int const *       lda_T,
                  double *          work,
                  int *             info);
extern "C"
void UTIL_DTPMQRT( char const *  side,
                   char const *  trans,
                   int const *   m,
                   int const *   n,
                   int const *   k,
                   int const *   l,
                   int const *   NB,
                   double *      V,
                   int const *   lda_V,
                   double *      T,
                   int const *   lda_T,
                   double *      A,
                   int const *   lda_A,
                   double *      B,
                   int const *   lda_B,
                   double *      work,
                   int *         info);
                    
extern "C"
void UTIL_DTPRFB(
             char const *side,
             char const *trans,
             char const *DIRECT,
             char const *STOREV,
             int const * m,
             int const * n,
             int const * k,
             int const * l,
             double *    V,
             int const * lda_V,
             double *    T,
             int const * lda_T,
             double *    A,
             int const * lda_A,
             double *    B,
             int const * lda_B,
             double *    work,
             int *       ldwork);
#endif

extern "C"
void UTIL_DLARFT(char         * f, 
                 char         * c, 
                 int          * m, 
                 int          * b, 
                 double const * Y, 
                 int          * lda_Y, 
                 double const * tau,
                 double       * T, 
                 int          * lda_T);


extern "C"
void UTIL_DLARFB(char         * l, 
                 char         * t, 
                 char         * f, 
                 char         * c, 
                 int          * m, 
                 int          * k, 
                 int          * b, 
                 double const * Y, 
                 int          * lda_Y, 
                 double const * T, 
                 int          * lda_T,
                 double       * B, 
                 int          * lda_B, 
                 double       * buffer, 
                 int          * buf_sz);
extern "C"
void UTIL_DLACPY(char         * uplo,
                 int          * m,
                 int          * n,
                 double const * A,
                 int          * lda_A,
                 double       * B,
                 int          * lda_B);

extern "C"
void UTIL_DSYEVX( char *   JOBZ, 
               char *   RANGE, 
               char *   UPLO, 
               int *    N, 
               double * A, 
               int *    LDA, 
               double * VL, 
               double * VU, 
               int *    IL,
               int *    IU, 
               double * ABSTOL, 
               int *    M, 
               double * W, 
               double * Z, 
               int *    LDZ, 
               double * WORK, 
               int *    LWORK, 
               int *    IWORK, 
               int *    IFAIL, 
               int *    INFO);

extern "C"
void UTIL_DLATRD( char *    UPLO,
             int *     N,
             int *     NB,
             double *  A,
             int *     LDA,
             double *  E,
             double *  TAU2,
             double *  W,
             int *     LDW);

extern "C"
void UTIL_DSYTRD( char *   UPLO,
             int *    N,
             double * A,
             int *    LDA,
             double * D,
             double * E,
             double * TAU2,
             double * WORK,
             int *    LWORK,
             int *    INFO);

extern "C"
void UTIL_DTRTRI(char *    side,
                 char *    transp,
                 int *     m,
                 double *  T,
                 int *     lda_T,
                 int *     info);

extern "C"
void UTIL_DGETRI(int *    N,
                 double * A,
                 int *    LDA,
                 int *    IPIV,
                 double * WORK,
                 int *    lwork,
                 int  *   info);



void cdlacpy(char uplo, int m, int n, double const * A, int lda_A, double * B, int lda_B){
  UTIL_DLACPY(&uplo, &m, &n, A, &lda_A, B, &lda_B);
}



void cdsyrk(char const          UPLO,
            char const          TRANS,
            int const           M,
            int const           K,
            double const        ALPHA,
            double const *      A,
            int const           LDA,
            double const        BETA,
            double *            C,
            int const           LDC){
  UTIL_DSYRK(&UPLO, &TRANS, &M, &K, &ALPHA, A, &LDA, &BETA, C, &LDC);
}


void cdgetrf(int      M,    int N,
		         double * A,		int lda,
		         int * IPIV,		int * info){
  UTIL_DGETRF(&M, &N, A, &lda, IPIV, info);
}
   

void cdtrsm( char SIDE,	char UPLO,
             char TRANSA,	char DIAG,
             int M,		int N,
             double alpha,	const double * A,
             int lda,	double * B,
					   int ldb){
  UTIL_DTRSM(&SIDE, &UPLO, &TRANSA, &DIAG, &M, &N, &alpha, A, &lda, B, &ldb);
}
	    
void cdgemm(char transa,  char transb,
            int m,        int n,
            int k,        const double a,
            const double * A,   int lda,
            const double * B,   int ldb,
            double b,     double * C,
                                int ldc){
  UTIL_DGEMM(&transa, &transb, &m, &n, &k, &a, A,
             &lda, B, &ldb, &b, C, &ldc);
}

void czgemm(char transa,  char transb,
            int m,        int n,
            int k,        const std::complex<double> a,
            const std::complex<double> * A,     int lda,
            const std::complex<double> * B,     int ldb,
            const std::complex<double> b,       std::complex<double> * C,
                                int ldc){
#ifdef USE_JAG
  jag_zgemm((char*)&transa, (char*)&transb, (int*)&m, (int*)&n, (int*)&k, (double*)&a, (double*)A,
             (int*)&lda, (double*)B, (int*)&ldb, (double*)&b, (double*)C, (int*)&ldc);
#else
  UTIL_ZGEMM(&transa, &transb, &m, &n, &k, &a, A,
             &lda, B, &ldb, &b, C, &ldc);
#endif
}

void czaxpy(int n,        
            std::complex<double> dA,
            const std::complex<double> * dX,
            int incX,
            std::complex<double> * dY,
            int incY){
  UTIL_ZAXPY(&n, &dA, dX, &incX, dY, &incY);
}


void cdaxpy(int n,        double dA,
            const double * dX,  int incX,
            double * dY,        int incY){
  UTIL_DAXPY(&n, &dA, dX, &incX, dY, &incY);
}

void czcopy(int n,
            const std::complex<double> * dX,
            int incX,
            std::complex<double> * dY,
            int incY){
  UTIL_ZCOPY(&n, dX, &incX, dY, &incY);
}


void cdcopy(int n,
            const double * dX,  int incX,
            double * dY,        int incY){
  UTIL_DCOPY(&n, dX, &incX, dY, &incY);
}

void cdscal(int n,        double dA,
            double * dX,  int incX){
  UTIL_DSCAL(&n, &dA, dX, &incX);
}

void czscal(int n,        std::complex<double> dA,
            std::complex<double> * dX,  int incX){
  UTIL_ZSCAL(&n, &dA, dX, &incX);
}


double cddot(int n,       const double *dX,
             int incX,    const double *dY,
             int incY){
  return UTIL_DDOT(&n, dX, &incX, dY, &incY);
}



void cdgeqrf(int const    M,
             int const    N,
             double   *   A,
             int const    LDA,
             double *     TAU2,
             double *     WORK,
             int const    LWORK,
             int  *       INFO){
  UTIL_DGEQRF(&M, &N, A, &LDA, TAU2, WORK, &LWORK, INFO);
}

template<> 
void cxgeqrf<double>(int const    M,
             int const    N,
             double   *   A,
             int const    LDA,
             double *     TAU2,
             double *     WORK,
             int const    LWORK,
             int  *       INFO){
  UTIL_DGEQRF(&M, &N, A, &LDA, TAU2, WORK, &LWORK, INFO);
}

template<> 
void cxgeqrf< std::complex<double> >(int const    M,
             int const    N,
             std::complex<double>   *   A,
             int const    LDA,
             std::complex<double> *     TAU2,
             std::complex<double> *     WORK,
             int const    LWORK,
             int  *       INFO){
  UTIL_ZGEQRF(&M, &N, A, &LDA, TAU2, WORK, &LWORK, INFO);
}




void cdormqr(char const   SIDE,
             char const   TRANS,
             int const    M,
             int const    N,
             int const    K,
             double const * A,
             int const    LDA,
             double const * TAU2,
             double   * C,
             int const    LDC,
             double * WORK,
             int const    LWORK,
             int  * INFO){
  UTIL_DORMQR(&SIDE, &TRANS, &M, &N, &K, A, &LDA, TAU2, C, &LDC,
        WORK, &LWORK, INFO);
}

void cdorgqr(int const    M,
             int const    N,
             int const    K,
             double   * A,
             int const    LDA,
             double const * TAU2,
             double * WORK,
             int const    LWORK,
             int  * INFO){
  UTIL_DORGQR(&M, &N, &K, A, &LDA, TAU2, WORK, &LWORK, INFO);
}


void cdger(int const    M,
           int const    N,
           double const   alpha,
           const double *   X,
           const int    incX,
           const double *   Y,  
           const int    incY,
           double *   A,
           const int    lda_A){
  UTIL_DGER(&M, &N, &alpha, X, &incX, Y, &incY, A, &lda_A);
}

double cdlange(char const     NORM,
               int const      M,
               int const      N,
               double const * A,
               int const      LDA,
               double *       WORK){
  return UTIL_DLANGE(&NORM, &M, &N, A, &LDA, WORK);
}

 
template <> 
void cxgemm<double>(const char transa,  const char transb,
            const int m,        const int n,
            const int k,        const double a,
            const double * A,   const int lda,
            const double * B,   const int ldb,
            const double b,     double * C,
                                const int ldc){
  cdgemm(transa, transb, m, n, k, a, A, lda, B, ldb, b, C, ldc);
}

template <> 
void cxgemm< std::complex<double> >(const char transa,  const char transb,
            const int m,        const int n,
            const int k,        const std::complex<double> a,
            const std::complex<double> * A,     const int lda,
            const std::complex<double> * B,     const int ldb,
            const std::complex<double> b,       std::complex<double> * C,
                                const int ldc){
  czgemm(transa, transb, m, n, k, a, A, lda, B, ldb, b, C, ldc);
}

template <> 
void cxaxpy<double>(const int n,        double dA,
                    const double * dX,  const int incX,
                    double * dY,        const int incY){
  cdaxpy(n, dA, dX, incX, dY, incY);
}

template <> 
void cxaxpy< std::complex<double> >
                    (const int n,
                     std::complex<double> dA,
                     const std::complex<double> * dX,
                     const int incX,
                     std::complex<double> * dY,
                     const int incY){
  czaxpy(n, dA, dX, incX, dY, incY);
}

template <> 
void cxscal<double>(const int n,        double dA,
                    double * dX,  const int incX){
  cdscal(n, dA, dX, incX);
}

template <> 
void cxscal< std::complex<double> >
                    (const int n,
                     std::complex<double> dA,
                     std::complex<double> * dX,
                     const int incX){
  czscal(n, dA, dX, incX);
}

template <> 
void cxcopy<double>(const int n,
                    const double * dX,  const int incX,
                    double * dY,        const int incY){
  cdcopy(n, dX, incX, dY, incY);
}

template <> 
void cxcopy< std::complex<double> >
                    (const int n,
                     const std::complex<double> * dX,
                     const int incX,
                     std::complex<double> * dY,
                     const int incY){
  czcopy(n, dX, incX, dY, incY);
}

void cdgetri(int      N,
             double * A,
             int      LDA,
             int *    IPIV,
             double * WORK,
             int      lwork,
             int  *   info){
  UTIL_DGETRI(&N, A, &LDA, IPIV, WORK, &lwork, info);
}



#if (LAPACKHASTSQR==1)
void cdtpqrt( int const         m,
              int const         n,
              int const         l,
              int const         nb,
              double *          A,
              int const         lda_A,
              double *          B,
              int const         lda_B,
              double *          T,
              int const         lda_T,
              double *          work,
              int *             info){
  UTIL_DTPQRT(&m, &n, &l, &nb, A, &lda_A, B, &lda_B, T, &lda_T, work, info);
}

void cdtprfb(char const  side,
             char const  trans,
             char const  DIRECT,
             char const  STOREV,
             int const   m,
             int const   n,
             int const   k,
             int const   l,
             double *    V,
             int const   lda_V,
             double *    T,
             int const   lda_T,
             double *    A,
             int const   lda_A,
             double *    B,
             int const   lda_B,
             double *    work,
             int         ldwork){
  UTIL_DTPRFB(&side, &trans, &DIRECT, &STOREV, &m, &n, &k, &l, V, &lda_V, T, &lda_T, A, &lda_A, B, &lda_B, work, &ldwork);
}
void cdtpmqrt( char const  side,
               char const  trans,
               int const   m,
               int const   n,
               int const   k,
               int const   l,
               int const   NB,
               double *    V,
               int const   lda_V,
               double *    T,
               int const   lda_T,
               double *    A,
               int const   lda_A,
               double *    B,
               int const   lda_B,
               double *    work,
               int *       info){
                
  UTIL_DTPMQRT(&side, &trans, &m, &n, &k, &l, &NB, V, &lda_V, T, &lda_T, A, &lda_A, B, &lda_B, work, info);
}
             
#endif
 
void cdlarft(char           f, 
             char           c, 
             int            m, 
             int            b, 
             double const * Y, 
             int            lda_Y, 
             double const * tau,
             double *       T, 
             int            lda_T){
  UTIL_DLARFT(&f,&c,&m,&b,Y,&lda_Y,tau,T,&lda_T);
}

void cdlarfb(char           l, 
             char           t, 
             char           f, 
             char           c, 
             int            m, 
             int            k, 
             int            b, 
             double const * Y, 
             int            lda_Y, 
             double const * T, 
             int            lda_T,
             double *       B, 
             int            lda_B, 
             double *       buffer, 
             int            buf_sz){
  UTIL_DLARFB(&l,&t,&f,&c,&m,&k,&b,Y,&lda_Y,T,&lda_T,
              B,&lda_B,buffer,&buf_sz);
}

void cdlatrd(char      UPLO,
             int       N,
             int       NB,
             double *  A,
             int       LDA,
             double *  E,
             double *  TAU2,
             double *  W,
             int       LDW){
  UTIL_DLATRD(&UPLO, &N, &NB, A, &LDA, E, TAU2, W, &LDW);
}

void cdsytrd(char     UPLO,
             int      N,
             double * A,
             int      LDA,
             double * D,
             double * E,
             double * TAU2,
             double * WORK,
             int      LWORK,
             int *    INFO){
  UTIL_DSYTRD(&UPLO, &N, A, &LDA, D, E, TAU2, WORK, &LWORK, INFO);
}

void cdsyevx(char     JOBZ, 
             char     RANGE, 
             char     UPLO, 
             int      N, 
             double * A, 
             int      LDA, 
             double   VL, 
             double   VU, 
             int      IL,
             int      IU, 
             double   ABSTOL, 
             int *    M, 
             double * W, 
             double * Z, 
             int      LDZ, 
             double * WORK, 
             int      LWORK, 
             int *    IWORK, 
             int *    IFAIL, 
             int *    INFO){
  UTIL_DSYEVX(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU, &ABSTOL,
         M, W, Z, &LDZ, WORK, &LWORK, IWORK, IFAIL, INFO);
}

   
void cdtrtri(char     UPLO,
            char     DIAG,
            int      N,
            double * A,
            int      LDA,
            int *    INFO){
  UTIL_DTRTRI(&UPLO, &DIAG, &N, A, &LDA, INFO);
}


#ifdef USE_SCALAPACK
#if (defined BGP || defined BGQ)
#define DESCINIT  descinit
#define PDGEMM    pdgemm
#define PDSYRK    pdsyrk
#define PDTRSM    pdtrsm
#define PDSYEVX   pdsyevx
#define PDLATRD   pdlatrd
#define PDSYTRD   pdsytrd
#define PDGEQRF   pdgeqrf
#define PZGEQRF   pzgeqrf
#define PDORGQR   pdorgqr
#define PZORGQR   pzungqr
#else
#define DESCINIT  descinit_
#define PDGEMM    pdgemm_
#define PDSYRK    pdsyrk_
#define PDTRSM    pdtrsm_
#define PDSYEVX   pdsyevx_
#define PDLATRD   pdlatrd_
#define PDSYTRD   pdsytrd_
#define PDGEQRF   pdgeqrf_
#define PZGEQRF   pzgeqrf_
#define PDORGQR   pdorgqr_
#define PZORGQR   pzungqr_
#endif

extern "C"{
  void DESCINIT(int *,       const int *,
                const int *, const int *,
                const int *, const int *,
                const int *, const int *,
                const int *, int *);
 
  void PDGEMM( char *,     char *,
             int *,      int *,
             int *,      double *,
             double *,   int *,
             int *,      int *,
             double *,   int *,
             int *,      int *,
             double *,  double *,
             int *,      int *,
                                 int *);

  void PDSYRK(char * UPLO,
              char * TRANS,
              int *   N,
              int *   K,
              double *  ALPHA,
              double const *  A,
              int *   IA,
              int *   JA,
              int *   DESCA,
              double *  BETA,
              double *  C,
              int *   IC,
              int *   JC,
              int *   DESCC);

  
  void PDTRSM(char *  SIDE,
              char *  UPLO,
              char *  TRANS,
              char *  DIAG,
              int *   M,
              int *   N,
              double *  ALPHA,
              double const *  A,
              int *   IA,
              int *   JA,
              int *   DESCA,
              double *  B,
              int *   IB,
              int *   JB,
              int *   DESCB); 
  
  void PDGEQRF(int  *  M,
               int  *  N,
               double *     A,
               int  *  IA,
               int  *  JA,
               int const *  DESCA,
               double *     TAU2,
               double *     WORK,
               int  *  LWORK,
               int *        INFO);
   
  void PDORGQR(int  *  M,
               int  *  N,
               int  *  K,
               double *     A,
               int  *  IA,
               int  *  JA,
               int const *  DESCA,
               double *     TAU2,
               double *     WORK,
               int  *  LWORK,
               int *        INFO);
   
  void PZGEQRF(int  *  M,
               int  *  N,
               std::complex<double> *     A,
               int  *  IA,
               int  *  JA,
               int const *  DESCA,
               std::complex<double> *     TAU2,
               std::complex<double> *     WORK,
               int  *  LWORK,
               int *        INFO);
   
  void PZORGQR(int  *  M,
               int  *  N,
               int  *  K,
               std::complex<double> *     A,
               int  *  IA,
               int  *  JA,
               int const *  DESCA,
               std::complex<double> *     TAU2,
               std::complex<double> *     WORK,
               int  *  LWORK,
               int *        INFO);
  


  void UTIL_PDLATRD(char *    UPLO,
               int *     N,
               int *     NB,
               double *  A,
               int *     IA,
               int *     JA,
               int *     DESCA,
               double *  D,
               double *  E,
               double *  TAU2,
               double *  W,
               int *     IW,
               int *     JW,
               int *     DESCW,
               double *  WORK);
  
  void UTIL_PDSYTRD(char *   UPLO,
               int *    N,
               double * A,
               int *    IA,
               int *    JA,
               int *    DESCA,
               double * D,
               double * E,
               double * TAU2,
               double * WORK,
               int *    LWORK,
               int *    INFO);


  void UTIL_PDSYEVX(char *   JOBZ, 
               char *   RANGE, 
               char *   UPLO, 
               int *    N, 
               double * A, 
               int *    IA, 
               int *    JA, 
               int *    DESCA, 
               double * VL, 
               double * VU, 
               int *    IL,
               int *    IU, 
               double * ABSTOL, 
               int *    M, 
               int *    NZ, 
               double * W, 
               double * ORFAC, 
               double * Z, 
               int *    IZ, 
               int *    JZ, 
               int *    DESCZ,
               double * WORK, 
               int *    LWORK, 
               int *    IWORK, 
               int *    LIWORK, 
               int *    IFAIL, 
               int *    ICLUSTR, 
               double * GAP,
               int *    INFO);

}
void cdescinit(int * desc, 
               const int m,	    const int n,
               const int mb, 	  const int nb,
               const int irsrc,	const int icsrc,
               const int ictxt,	const int LLD,
               int * info){
  DESCINIT(desc,&m,&n,&mb,&nb,&irsrc,&icsrc,
           &ictxt, &LLD, info);
}

void cpdgemm( char n1,    char n2,
                    int sz1,     int sz2,
                    int sz3,     double ALPHA,
                    double * A,  int ia,
                    int ja,      int * desca,
                    double * B,  int ib,
                    int jb,      int * descb,
                    double BETA,        double * C,
                    int ic,      int jc,
                                         int * descc){
  PDGEMM(&n1, &n2, &sz1, 
         &sz2, &sz3, &ALPHA, 
         A, &ia, &ja, desca, 
         B, &ib, &jb, descb, &BETA,
         C, &ic, &jc, descc);
}

void cpdsyrk(char  UPLO,
             char  TRANS,
             int    N,
             int    K,
             double  ALPHA,
             double const *  A,
             int  IA,
             int  JA,
             int *   DESCA,
             double  BETA,
             double *  C,
             int   IC,
             int   JC,
             int *   DESCC){
  PDSYRK(&UPLO, &TRANS, &N, &K, &ALPHA, A, 
         &IA, &JA, DESCA, &BETA, C, &IC, &JC, DESCC);
}

void cpdtrsm(char   SIDE,
             char   UPLO,
             char   TRANS,
             char   DIAG,
             int    M,
             int    N,
             double   ALPHA,
             double const *  A,
             int    IA,
             int    JA,
             int *   DESCA,
             double *  B,
             int    IB,
             int    JB,
             int *   DESCB){
  PDTRSM(&SIDE, &UPLO, &TRANS, &DIAG, &M, &N, &ALPHA, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB);
}

void cpdgeqrf(int  M,
              int  N,
              double *  A,
              int  IA,
              int  JA,
              int const *     DESCA,
              double *  TAU2,
              double *  WORK,
              int  LWORK,
              int *     INFO){
  PDGEQRF(&M,&N,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
}

template <> 
void cpxgeqrf<double>(int  M,
              int  N,
              double *  A,
              int  IA,
              int  JA,
              int const *     DESCA,
              double *  TAU2,
              double *  WORK,
              int  LWORK,
              int *     INFO){
  PDGEQRF(&M,&N,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
}

template <typename dtype> 
void cpxgeqrf(int  M,
              int  N,
              dtype *  A,
              int  IA,
              int  JA,
              int const *     DESCA,
              dtype *  TAU2,
              dtype *  WORK,
              int  LWORK,
              int *     INFO){
  assert(0);
}

template <> 
void cpxgeqrf< std::complex<double> >(int  M,
              int  N,
              std::complex<double> *  A,
              int  IA,
              int  JA,
              int const *     DESCA,
              std::complex<double> *  TAU2,
              std::complex<double> *  WORK,
              int  LWORK,
              int *     INFO){
  PZGEQRF(&M,&N,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
}
extern "C"
void UTIL_DGEQRF(int const *  M,
                 int const *  N,
                 double * A,
                 int const *  LDA,
                 double * TAU2,
                 double * WORK,
                 int const *  LWORK,
                 int  * INFO);



void cpdorgqr(int  M,
              int  N,
              int  K,
              double *  A,
              int  IA,
              int  JA,
              int const *     DESCA,
              double *  TAU2,
              double *  WORK,
              int  LWORK,
              int *     INFO){
  PDORGQR(&M,&N,&K,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
}

template <>
void cpxorgqr<double>(int  M,
              int  N,
              int  K,
              double *  A,
              int  IA,
              int  JA,
              int const *     DESCA,
              double *  TAU2,
              double *  WORK,
              int  LWORK,
              int *     INFO){
  PDORGQR(&M,&N,&K,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
}

template <>
void cpxorgqr< std::complex<double> >(int  M,
              int  N,
              int  K,
               std::complex<double>  *  A,
              int  IA,
              int  JA,
              int const *     DESCA,
               std::complex<double>  *  TAU2,
               std::complex<double>  *  WORK,
              int  LWORK,
              int *     INFO){
  PZORGQR(&M,&N,&K,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
}


  
void cpdlatrd( char      UPLO,
               int       N,
               int       NB,
               double *  A,
               int       IA,
               int       JA,
               int *     DESCA,
               double *  D,
               double *  E,
               double *  TAU2,
               double *  W,
               int       IW,
               int       JW,
               int *     DESCW,
               double *  WORK){
  UTIL_PDLATRD(&UPLO, &N, &NB, A, &IA, &JA, DESCA, D, E, TAU2, W, &IW, &JW, DESCW, WORK);
}
 


void cpdsytrd( char     UPLO,
               int      N,
               double * A,
               int      IA,
               int      JA,
               int *    DESCA,
               double * D,
               double * E,
               double * TAU2,
               double * WORK,
               int      LWORK,
               int *    INFO){
  UTIL_PDSYTRD(&UPLO, &N, A, &IA, &JA, DESCA, D, E, TAU2, WORK, &LWORK, INFO);
}

void cpdsyevx( char     JOBZ, 
               char     RANGE, 
               char     UPLO, 
               int      N, 
               double * A, 
               int      IA, 
               int      JA, 
               int *    DESCA, 
               double   VL, 
               double   VU, 
               int      IL,
               int      IU, 
               double   ABSTOL, 
               int *    M, 
               int *    NZ, 
               double * W, 
               double   ORFAC, 
               double * Z, 
               int      IZ, 
               int      JZ, 
               int *    DESCZ,
               double * WORK, 
               int      LWORK, 
               int *    IWORK, 
               int      LIWORK, 
               int *    IFAIL, 
               int *    ICLUSTR, 
               double * GAP,
               int *    INFO){
  UTIL_PDSYEVX(&JOBZ, &RANGE, &UPLO, &N, A, &IA, &JA, DESCA, &VL, &VU, &IL, &IU,
          &ABSTOL, M, NZ, W, &ORFAC, Z, &IZ, &JZ, DESCZ, WORK, &LWORK, 
          IWORK, &LIWORK, IFAIL, ICLUSTR, GAP, INFO);
}

#endif


