#include <memory>
#include <time.h>
#include <algorithm>
#include <float.h>
#include <limits.h>
#include <string.h>
#include "./GST_def.h"
#include "./Object.hpp"
#include "BLAS_t.hpp"
//#include "GST_util.hpp"

/*
	经测试 似乎-o2有优于-o3
*/
#ifdef MKL_LIB
	#include "mkl.h"
	#include "mkl_types.h"
	#include "mkl_spblas.h"
	#include "mkl_blas.h"
	#include "mkl_lapack.h"
	#include "mkl_lapacke.h"
	

	static MKL_Complex16 Z1_mkl={1.0,0.0},Z0_mkl={0.0,0.0};

#endif
static double D1_=1.0,D0_=0.0;
static const int inc_1=1;

template<> bool IS_NAN<COMPLEXd>( const COMPLEXd &a ){	return	std::isnan(a.real()) || std::isnan(a.imag());		}
template<> bool IS_NAN<COMPLEXf>( const COMPLEXf &a ){	return  std::isnan(a.real()) || std::isnan(a.imag());		}

template<> double ABS_1<float>( const float &a )			{	return fabs(a);	}
template<> double ABS_1<double>( const double &a )			{	return fabs(a);	}
template<> double ABS_1<COMPLEXd>( const COMPLEXd &a )		{	return  fabs(a.real())+ fabs(a.imag());	}
template<> double ABS_1<COMPLEXf>( const COMPLEXf &a )		{	return  fabs(a.real())+ fabs(a.imag());	}

template<> double TxTc<COMPLEXd>( const COMPLEXd &a )		{	double re=a.real(),im=a.imag();	return  re*re+im*im;	}
template<> double TxTc<COMPLEXf>( const COMPLEXf &a )		{	double re=a.real(),im=a.imag();	return  re*re+im*im;	}

template <> bool isReal<COMPLEXd>( COMPLEXd val )	{	return val.imag()==0.0; }

static char trans[]={'N','T','C'};
//template <typename T> T NRM2( const int dim,T *X )
template<> double NRM2<double>( const int dim,const double *X )	{
	return DNRM2( &dim,X, &inc_1);
}
template<> double NRM2<float>( const int dim,const float *X )	{
	return SNRM2( &dim,X, &inc_1);
}
template<> double NRM2<COMPLEXd>( const int dim,const COMPLEXd *X )	{
	return DZNRM2( &dim,(MKL_Complex16*)X, &inc_1);
}
template<> double NRM2<COMPLEXf>( const int dim,const COMPLEXf *X )	{
	return SCNRM2( &dim,(MKL_Complex8*)X, &inc_1);
}
//template <typename T> void SCAL( const int dim,const double alpha,T *X )		
template<> void SCALd<double>( const int dim,const double alpha,double *X )	{
	DSCAL( &dim,&alpha, X, &inc_1);
}
template<> void SCALd<float>( const int dim,const double alpha,float *X )	
{	float a=alpha;	SSCAL( &dim,&a, X, &inc_1);		}
template<> void SCALd<COMPLEXd>( const int dim,const double alpha,COMPLEXd *X )	
{	ZDSCAL( &dim,&alpha, (MKL_Complex16*)X, &inc_1);	}
template<> void SCALd<COMPLEXf>( const int dim,const double alpha_,COMPLEXf *X )	{	
	float alpha=alpha_;
	CSSCAL( &dim,&alpha, (MKL_Complex8*)X, &inc_1);	
}

template<> void SCAL<COMPLEXd>( const int dim,const COMPLEXd alpha,COMPLEXd *X )	{
	ZSCAL( &dim,(MKL_Complex16*)&alpha, (MKL_Complex16*)X, &inc_1);
}

//template <typename T> T DOT( const int dim,const T *X,const T *Y )	
template <>
void DOT<double>( double& dot,const int dim,const double *X,const double *Y )	{
	dot = DDOT( &dim, X, &inc_1, Y, &inc_1);
}
template <>
void DOT<float>( float& dot,const int dim,const float *X,const float *Y )	{
	dot = SDOT( &dim, X, &inc_1, Y, &inc_1);
}
template <>
void DOT<COMPLEXd>( COMPLEXd& dot,const int dim,const COMPLEXd *X,const COMPLEXd *Y )	{
	ZDOTC((MKL_Complex16 *)(&dot), &(dim), (const MKL_Complex16 *)X, &inc_1, (const MKL_Complex16 *)Y, &inc_1);
}

template<> void COPY<float>( const int dim,const float *X,float *Y )	
{	SCOPY( &dim,X,&inc_1,Y,&inc_1 );}
template<> void COPY<double>( const int dim,const double *X,double *Y )	
{	DCOPY( &dim,X,&inc_1,Y,&inc_1 );}
template<> void COPY<COMPLEXd>( const int dim,const COMPLEXd *X,COMPLEXd *Y )	
{	ZCOPY( &dim,(const MKL_Complex16 *)X,&inc_1,(MKL_Complex16 *)Y,&inc_1 );}

template <> void IMAT_T<COMPLEXf>( const char ordering, size_t rows, size_t cols,COMPLEXf * AB){
	MKL_Complex8 alpha={1.0};
	if(ordering ==charR)		//the ordering is row-major.
		mkl_cimatcopy (ordering, charT, rows, cols, alpha, (MKL_Complex8*)AB, cols, rows);
	else                       //the ordering is column-major.
		mkl_cimatcopy (ordering, charT, rows, cols, alpha, (MKL_Complex8*)AB, rows, cols);
}

template<> void SWAP<COMPLEXf>( const int dim,COMPLEXf *X,int incx,COMPLEXf *Y,int incy ){
	CSWAP( &dim, (MKL_Complex8 *)X, &incx, (MKL_Complex8 *)Y, &incy);
}

//Y := a*X + Y
template <> void AXPY<double>( const int dim,const double alpha,const double *X,double *Y )	{
	DAXPY( &dim,&alpha, X, &inc_1, Y, &inc_1);
//	for( int k = 0; k < dim; k++ )	Y[k]+=alpha*X[k];
}
template <> void AXPY<float>( const int dim,const float alpha,const float *X,float *Y )	{
	SAXPY( &dim,&alpha, X, &inc_1, Y, &inc_1);
}
template<> void AXPY<COMPLEXd>( const int dim,const COMPLEXd alpha,const COMPLEXd *X,COMPLEXd *Y )	{
	/*for( int i=0;i<dim;i++ ){
		Y[i] += alpha*X[i];
	}*/
	assert( sizeof(COMPLEXd)==sizeof(MKL_Complex16) );
	ZAXPY( &dim,(const MKL_Complex16 *)(void *)(&alpha), (const MKL_Complex16 *)(void *)X, &inc_1, (MKL_Complex16 *)(void *)Y, &inc_1);
}
template<> void AXPY<COMPLEXf>( const int dim,const COMPLEXf alpha,const COMPLEXf *X,COMPLEXf *Y )	{
	CAXPY( &dim,(const MKL_Complex8 *)(void *)(&alpha), (const MKL_Complex8 *)(void *)X, &inc_1, (MKL_Complex8 *)(void *)Y, &inc_1);
}

//Y := a*X + bY
template <> void AXPBY<double>( const int dim,const double alpha,const double *X,const double beta,double *Y )	{
	DAXPBY( &dim,&alpha, X, &inc_1, &beta,Y, &inc_1);
}
template <> void AXPBY<float>( const int dim,const float alpha,const float *X,const float beta,float *Y )	{
	SAXPBY( &dim,&alpha, X, &inc_1, &beta,Y, &inc_1);
}

template<> 
void GER_11<double>( const int M,const int N, const double alpha, const double *x, const double *y, double *A, const int lda )	{
//	A[0] += x[0]*y[0];
//	DGER( &M, &N, &alpha, x, &incx, y, &incy, A, &lda );
	int i,j;
	double *pa,x_i;
	for( i = 0; i < M; i++ ) {
		pa = A+i*lda;		x_i = x[i];
		for( j = 0; j < N; j++ ) {
			pa[j] += x_i*y[j];
		}
	}
}

template<> 
void GEMV<double>( char transa,const int M,const int N, const double alpha, const double *A, const int lda, const double *X, const int incx, const double beta, double *Y, const int incy )	{
	DGEMV(&transa,&M,&N,&alpha,A,&lda,X,&incx,&beta,Y,&incy);
}
template<> 
void GEMV<C>( char transa,const int M,const int N, const C alpha, const C *A, const int lda, const C *X, const int incx, const C beta, C *Y, const int incy )	{
	CGEMV(&transa,&M,&N,(MKL_Complex8*)(&alpha),(MKL_Complex8*)A,&lda,(MKL_Complex8*)X,&incx,(MKL_Complex8*)(&beta),(MKL_Complex8*)Y,&incy);
}

template<> 
void GEMV<Z>( char transa,const int M,const int N, const Z alpha, const Z *A, const int lda, const Z *X, const int incx, const Z beta, Z *Y, const int incy )	{
	ZGEMV(&transa,&M,&N,(MKL_Complex16*)(&alpha),(MKL_Complex16*)A,&lda,(MKL_Complex16*)X,&incx,(MKL_Complex16*)(&beta),(MKL_Complex16*)Y,&incy);
}

template<> 
void GEAV<Z>( char transa,const int M,const int N, const Z *A, const int lda, const Z *X, Z *Y  )	{	//Y=AX
	ZGEMV(&transa,&M,&N,&Z1_mkl,(MKL_Complex16*)A,&lda,(MKL_Complex16*)X,&inc_1,&Z0_mkl,(MKL_Complex16*)Y,&inc_1);
}
template<> void GEAV<C>( char transa,const int M,const int N, const C *A, const int lda, const C *X, C *Y  )	{	//Y=AX
	MKL_Complex8 one={1.0f,0.0},zero={0.0,0.0};
	CGEMV(&transa,&M,&N,&one,(MKL_Complex8*)A,&lda,(MKL_Complex8*)X,&inc_1,&zero,(MKL_Complex8*)Y,&inc_1);
}
template<> void GEAV<double>( char transa,const int M,const int N, const double *A, const int lda, const double *X, double *Y  )	{	//Y=AX
	double one(1.0),zero(0.0);
	DGEMV(&transa,&M,&N,&one,A,&lda,X,&inc_1,&zero,Y,&inc_1);
}

template<> 
void GEMM<COMPLEXd>( char transa, char transb, int m, int n, int k, const COMPLEXd *alpha, COMPLEXd *a, int lda, COMPLEXd *b, int ldb, const COMPLEXd *beta, COMPLEXd *c, int ldc )	{
	ZGEMM(&transa, &transb, &m, &n, &k, (MKL_Complex16*)alpha, (MKL_Complex16*)a, &lda, (MKL_Complex16*)b, &ldb, (MKL_Complex16*)beta, (MKL_Complex16*)c, &ldc);
}
template<> 
void GEMM<COMPLEXf>( char transa, char transb, int m, int n, int k, const COMPLEXf *alpha, COMPLEXf *a, int lda, COMPLEXf *b, int ldb, const COMPLEXf *beta, COMPLEXf *c, int ldc )	{
	CGEMM(&transa, &transb, &m, &n, &k, (MKL_Complex8*)alpha, (MKL_Complex8*)a, &lda, (MKL_Complex8*)b, &ldb, (MKL_Complex8*)beta, (MKL_Complex8*)c, &ldc);
}
template<> 
void C_GEMM<COMPLEXf>( char transa, char transb, int m, int n, int k, const COMPLEXf *alpha, COMPLEXf *a, int lda, COMPLEXf *b, int ldb, const COMPLEXf *beta, COMPLEXf *c, int ldc )	{
	CBLAS_TRANSPOSE ta=CblasNoTrans,tb=CblasNoTrans;
	assert( ldc>=MAX(1,n) );
	cblas_cgemm(CblasRowMajor,ta, tb, m, n, k, (MKL_Complex8*)alpha, (MKL_Complex8*)a, lda, (MKL_Complex8*)b, ldb, (MKL_Complex8*)beta, (MKL_Complex8*)c, ldc);
}

template<> 
void GEMM<double>( char transa, char transb, int m, int n, int k, const double *alpha,  double *a, int lda, double *b, int ldb, const double *beta, double *c, int ldc )	{
	DGEMM(&transa, &transb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc);
}
template<>  void GEMM<float>( char transa, char transb, int m, int n, int k, const float *alpha, float *a, int lda, float *b, int ldb, const float *beta, float *c, int ldc )	{
	SGEMM(&transa, &transb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc);
}

template<>  void HERK_s<COMPLEXf>( char uplo, char trans, int m, int k, COMPLEXf *a, int lda, COMPLEXf *c, int ldc,int flag ){
	float alpha=flag==1 ? 1:-1,beta=0;
	CHERK(&uplo, &trans,&m, &k, &alpha, (MKL_Complex8*)a, &lda, &beta, (MKL_Complex8*)c, &ldc);
}
template<>  void HERK_s<COMPLEXd>( char uplo, char trans, int m, int k, COMPLEXd *a, int lda, COMPLEXd *c, int ldc,int flag ){
	double alpha=flag==1 ? 1:-1,beta=0;
	ZHERK(&uplo, &trans,&m, &k, &alpha, (MKL_Complex16*)a, &lda, &beta, (MKL_Complex16*)c, &ldc);
}

template<> 
void TRSV<COMPLEXd>( char uplo,char transa, char diag, int m, COMPLEXd *a, int lda, COMPLEXd *b, int inc_ )	{
	ZTRSV(&uplo,&transa, &diag, &m, (MKL_Complex16*)a, &lda, (MKL_Complex16*)b, &inc_);
}
template<> 
void TRSV<COMPLEXf>( char uplo,char transa, char diag, int m, COMPLEXf *a, int lda, COMPLEXf *b, int inc_ )	{
	CTRSV(&uplo,&transa, &diag, &m, (MKL_Complex8*)a, &lda, (MKL_Complex8*)b, &inc_);
}
template<> 
void TRSV<double>( char uplo,char transa, char diag, int m, double *a, int lda, double *b, int inc_ )	{
	DTRSV(&uplo,&transa, &diag, &m, a, &lda, b, &inc_ );
}

template<> 
void TRSM<COMPLEXd>( char side,char uplo,char transa, char diag, int m, int n, const COMPLEXd *alpha, COMPLEXd *a, int lda, COMPLEXd *b, int ldb )	{
	ZTRSM(&side,&uplo,&transa, &diag, &m, &n, (MKL_Complex16*)alpha, (MKL_Complex16*)a, &lda, (MKL_Complex16*)b, &ldb);
}
template<> 
void TRSM<COMPLEXf>( char side,char uplo,char transa, char diag, int m, int n, const COMPLEXf *alpha, COMPLEXf*a, int lda, COMPLEXf *b, int ldb )	{
	CTRSM(&side,&uplo,&transa, &diag, &m, &n, (MKL_Complex8*)alpha, (MKL_Complex8*)a, &lda, (MKL_Complex8*)b, &ldb);
}
template<> 
void TRSM<double>( char side,char uplo,char transa, char diag, int m, int n, const double *alpha, double *a, int lda, double *b, int ldb )	{
	DTRSM(&side,&uplo,&transa, &diag, &m, &n, alpha, a, &lda, b, &ldb );
}

template<>
void GER<float>( int m, int n, float *alpha, float *vX, int incx, float *vY, int incy, float *beta, float *A, int lda ){
	SGER(&m, &n, alpha, vX, &incx, vY, &incy, A, &lda);
}

template<> int GELS<double>(char trans, int m, int n,int nrhs, double* a, int lda,double* b, int ldb ){
	int ret = LAPACKE_dgels( LAPACK_COL_MAJOR, trans, m, n,nrhs,  a, lda, b, ldb );
	return ret;
}
template<> int GELS<COMPLEXf>(char trans, int m, int n,int nrhs, COMPLEXf* a, int lda,COMPLEXf* b, int ldb ){
	int ret = LAPACKE_cgels( LAPACK_COL_MAJOR, trans, m, n,nrhs,  (MKL_Complex8*)a, lda, (MKL_Complex8*)b, ldb );
	return ret;
}
template<> int GELS<COMPLEXd>(char trans, int m, int n,int nrhs, COMPLEXd* a, int lda,COMPLEXd* b, int ldb ){
	int ret = LAPACKE_zgels( LAPACK_COL_MAJOR, trans, m, n,nrhs,  (MKL_Complex16*)a, lda, (MKL_Complex16*)b, ldb );
	return ret;
}

/*	为了避免格式转化的混淆，固定transa为'N'，固定为one-based indexing('f')	*/
template<> 
void COO_MM<double>( int m, int n, int k, double *alpha, double *a, int *rowA,int*colA, int nnz,double *b, int ldb, double *beta, double *c, int ldc )	{
	char matdescra[6]={'g','l','n','f'},transa=charN;
//	for( int i = 0; i < n; i++ ) mkl_dcoomv( &transa, &m, &k, alpha, matdescra, a, rowA, colA, &nnz, b+ldb*i,beta, c+ldc*i );
	mkl_dcoomm( &transa, &m, &n, &k, alpha, matdescra, a, rowA, colA, &nnz, b, &ldb,beta, c, &ldc);	
}
template<> 
void COO_MM<float>( int m, int n, int k, float *alpha, float *a, int *rowA,int*colA, int nnz,float *b, int ldb, float *beta, float *c, int ldc )	{
	char matdescra[6]={'g','l','n','f'},transa=charN;
//	for( int i = 0; i < n; i++ ) 
//		mkl_scoomv( &transa, &m, &k, alpha, matdescra, a, rowA, colA, &nnz, b+ldb*i,beta, c+ldc*i );
	mkl_scoomm( &transa, &m, &n, &k, alpha, matdescra, a, rowA, colA, &nnz, b, &ldb,beta, c, &ldc);		
}
/*
template<> 
void COO_MM_t<double>( char transa,  int m, int n, int k, double *alpha, double *a, int *rowA,int*colA, int nnz,double *b, int ldb, double *beta, double *c, int ldc )	{
	char matdescra[6]={'g','l','n','f'};
	double *temp=nullptr;
	if( 1 )	{
		assert( ldc==m );
		temp=new double[m*n];	memcpy( temp,c,sizeof(double)*m*n );
		if( transa==charT )
			mkl_dcoomm( &transa, &k, &n, &m, alpha, matdescra, a, rowA, colA, &nnz, b, &ldb,beta, temp, &ldc);
		else
			mkl_dcoomm( &transa, &m, &n, &k, alpha, matdescra, a, rowA, colA, &nnz, b, &ldb,beta, temp, &ldc);
	}	
		for( int i = 0; i < n; i++ )	{
			mkl_dcoomv( &transa, &m, &k, alpha, matdescra, a, rowA, colA, &nnz, b+ldb*i,beta, c+ldc*i );
		}
	//	mkl_dcoomm( &transa, &m, &n, &k, alpha, matdescra, a, rowA, colA, &nnz, b, &ldb,beta, c, &ldc);
	if( 1 )	{
		double sum=0,d,nrmc=NRM2(m*n,c);
		for( int i = m*n-1; i--; i>=0 )		
		{d=temp[i]-c[i];	sum+=d*d;			}
		assert( sqrt(sum)<1.0e-10*max(nrmc,1.0) );
		delete[] temp;
	}
}
void TEST_OOMM( int m,int n,int k ){
	int nnz=m;
	double		*values=new double[nnz]();
	MKL_INT		*columns=new int[nnz](),*rows=new int[nnz]();
	for( int i = 0; i < m; i++ )	{
		values[i]=1.0;		columns[i]=i+1;			rows[i]=i+1;
	}
	double		alpha = 1.0, beta = 0.0,*b=new double[k*n](),*c=new double[m*n]();
	char transa= 't',matdescra[6]={'g','l','n','f'};;
	mkl_dcoomm(&transa, &m, &n, &k, &alpha, matdescra, values, rows, columns, &nnz, b, &k,  &beta, c,  &m);
}*/

//dist <1: uniform (0,1)	2: uniform (-1,1)	3: normal (0,1)>
template<> void LARNV_seed<double>( const int dim,double *X,int dist,int seed ){
	int iseed[4] = { 1, 3, 5, 1 };		//iseed: the array elements must be between 0 and 4095, and iseed(4) must be odd.
	for( int i = 0; i < 3; i++ )	iseed[i]=abs(dim*i+seed)%4095;
	DLARNV( &dist,iseed,&dim,X );
}
template<> void LARNV_seed<float>( const int dim,float *X,int dist,int seed ){
	int iseed[4] = { 1, 3, 5, 1 };
	for( int i = 0; i < 3; i++ )	iseed[i]=abs(dim*i+seed)%4095;
	SLARNV( &dist,iseed,&dim,X );	
}

template<> void LARNV<double>( const int dim,double *X,int dist ){
//idist <1: uniform (0,1)	2: uniform (-1,1)	3: normal (0,1)>
//iseed: the array elements must be between 0 and 4095, and iseed(4) must be odd.
	int iseed[4] = { 1, 3, 5, 1 }, idist = dist;
	for( int i = 0; i < 3; i++ )	iseed[i]=abs(dim*i+1)%4095;
	DLARNV( &idist,iseed,&dim,X );
}
template<> void LARNV<float>( const int dim,float *X,int dist ){
	int iseed[4] = { 1, 3, 5, 1 }, idist = dist;
	for( int i = 0; i < 3; i++ )	iseed[i]=abs(dim*i+1)%4095;
	SLARNV( &idist,iseed,&dim,X );
	if( 0 )	{
		double *dv=new double[dim];
		for( int i = 0; i < 3; i++ )	iseed[i]=(dim*i+1)%4095;		iseed[3]=1;
		DLARNV( &idist,iseed,&dim,dv );
		delete[] dv;
	}
}
template<> void LARNV<COMPLEXd>( const int dim,COMPLEXd *X,int seed ){
//2: real and imaginary parts each uniform (-1,1)
//4: uniformly distributed on the disc abs(z) < 1		5: uniformly distributed on the circle abs(z) = 1
	int iseed[4] = { 1, seed, 5, 7 }, idist = 2;	
	ZLARNV( &idist,iseed,&dim,(MKL_Complex16*)X );
}
template<> void LARNV<COMPLEXf>( const int dim,COMPLEXf *X,int seed ){
//2: real and imaginary parts each uniform (-1,1)
//4: uniformly distributed on the disc abs(z) < 1		5: uniformly distributed on the circle abs(z) = 1
	int iseed[4] = { 1, seed, 5, 7 }, idist = 2;	
	CLARNV( &idist,iseed,&dim,(MKL_Complex8*)X );
}

/*
template<> void GaussRNV<float>( const int dim,float *X,float mean,float sigma,int flag ){
	VSLStreamStatePtr stream;
	int seed=777,brng=VSL_BRNG_MCG31,errcode;
	brng=VSL_BRNG_MT19937;
	errcode=vslNewStream( &stream,brng,seed );
	CheckVslError( errcode );
	errcode=vsRngGaussian( VSL_METHOD_SGAUSSIAN_BOXMULLER,stream,dim,X,mean,sigma );
	CheckVslError( errcode );
}
template<> void GaussRNV<double>( const int dim,double *X,double mean,double sigma,int flag ){
	VSLStreamStatePtr stream;
	int seed=777,brng=VSL_BRNG_MCG31,errcode;
	brng=VSL_BRNG_MT19937;
	errcode=vslNewStream( &stream,brng,seed );
	CheckVslError( errcode );
	errcode=vdRngGaussian( VSL_METHOD_SGAUSSIAN_BOXMULLER,stream,dim,X,mean,sigma );
	CheckVslError( errcode );
}
template<> void GaussRNV<COMPLEXd>( const int dim,COMPLEXd *X,COMPLEXd mean,COMPLEXd sigma,int flag ){
	//3	real and imaginary parts each normal (0,1);
	assert(mean==0.0 && sigma==1.0);
	int iseed[4] = { 1, 3, 5, 7 }, idist = 3;	
	ZLARNV( &idist,iseed,&dim,(MKL_Complex16*)X );
}

template<> bool isUnitary<COMPLEXd>( int n, const hGMAT &mU,COMPLEXd *tau )	{
	if( GST_util::verify<GST_util::VERIRY_UNITARY_MAT )
		return true;
	int m=mU->RowNum( );
	COMPLEXd *tU=TO<COMPLEXd>(mU);
	ZGEMM(trans+2, trans, &n, &n, &m, &Z1_mkl, (MKL_Complex16*)tU, &m, (MKL_Complex16*)tU, &m, &Z0_mkl, (MKL_Complex16*)tau, &n);
	int i,j;
	double a,thrsh=1.0e-10,rel,img;
	for( i = 0; i < n; i++ )	{
	for( j = 0; j < n; j++ )	{
		rel = tau[i*n+j].real(),		img = tau[i*n+j].imag();
		a = i==j ?1.0- sqrt(rel*rel+img*img) : sqrt(rel*rel+img*img);
		if( fabs(a)<thrsh )	{
		}
		else		{
			return false;
		}
	}
	}
	return true;
}
template<> bool isIMat<COMPLEXd>( int nCol, const hGMAT &mU)	{
	if( GST_util::verify<GST_util::VERIRY_I_MAT )
		return true;

	int i,j,m=mU->RowNum( );
	COMPLEXd *tU=TO<COMPLEXd>(mU),one(1.0);
	double off,thrsh=1.0e-10;
	
	for( i = 0; i < nCol; i++ )	{
	for( j = 0; j < m; j++ )	{
		off = ( i==j ) ? fabs( tU[i*m+j]-one) : abs( tU[i*m+j]);
		if( off<thrsh )	{
		}
		else		{
			PRINT( "isIMat<%d> failed at (%d,%d,%g)\r\n",nCol,j,i,off );
			return false;
		}
	}
	}
	return true;
}

template<> bool isTriangular<COMPLEXd>( int nCol, const hGMAT &mU,bool isUp )	{
	if( GST_util::verify<GST_util::VERIRY_TRI_MAT )
		return true;
	int i,j,m=mU->RowNum( );
	assert(m>=nCol);
	COMPLEXd *tU=TO<COMPLEXd>(mU);
	double a,thrsh=1.0e-10;
	bool isTri=true;
	
	for( i = 0; i < nCol; i++ )	{
	for( j = 0; j < nCol; j++ )	{
		a = fabs( tU[i*m+j]);
		if( isUp )	{
			if( j>i && a>thrsh )
				isTri = false;
		}else	{
			if( j<i && a>thrsh )
				isTri = false;
		}
		if( !isTri )	{
			PRINT( "isTriangular<%d> failed at (%d,%d,%g)\r\n",nCol,j,i,a );
			return false;
		}
	}
	}
	return true;
}

template <>  void GEQR<COMPLEXd>( int m,int n,hGMAT &mB, const hGMAT &mA,COMPLEXd sift, int flag){	//A=QR
	assert( m<=mA->RowNum( ) && n<=mA->ColNum( ) && m<=mB->RowNum( ) && n<=mB->ColNum( ) );
	assert( mA->Count()==mB->Count() );
	int ldA=mA->RowNum(),ldB=mB->RowNum(),i;
	COMPLEXd *tQ=TO<COMPLEXd>(mB),*tA=TO<COMPLEXd>(mA);
	MKL_Complex16 *tau=new MKL_Complex16[n*n];
	memcpy( tQ,tA,sizeof(COMPLEXd)*mA->Count() );
	if( sift!=0.0 )	{
		for( i = min(m,n)-1; i >= 0 ; i-- )
			tQ[i*ldB+i] -= sift;
	}
	if( m<ldB )	{
		for( int i = 0; i < n; i++ )	{
			for( int j = m; j < ldB; j++ )	{
				tQ[i*ldB+j]=0;
			}
		}
	}
	LAPACKE_zgeqrf( LAPACK_COL_MAJOR,m,n, (MKL_Complex16*)tQ, ldB, tau );
	LAPACKE_zungqr( LAPACK_COL_MAJOR, m, n, n, (MKL_Complex16*)tQ, ldB, tau );
	if( 1 )	{	//verify
		assert( isUnitary( n,mB,(COMPLEXd *)tau ) );	
	}
	delete[] tau;
}*/

template<> int GEQR_p<COMPLEXd>(int m, int n, COMPLEXd *A, int ldA, COMPLEXd *Q, int *jpvt, int flag) {
	MKL_Complex16 *tau=new MKL_Complex16[MAX(m,n)];
	int matrix_order = flag==0 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR,i,no;
	//lapack_int info = LAPACKE_zgeqp3( matrix_order,m,n,(MKL_Complex16*)A,ldA,jpvt,tau );
	lapack_int info = LAPACKE_zgeqpf( matrix_order,m,n,(MKL_Complex16*)A,ldA,jpvt,tau );
	if (info == 0) {
		COMPLEXd pivot,pivot_0=DBL_MAX;
		for (int i = 0; i < n; i++) {
			pivot = A[i*m + i];		assert(std::abs(pivot) <= std::abs(pivot_0));
			pivot_0 = pivot;
		}
	}	else {
		assert(0);
		return info;
	}

	if (Q != nullptr) {
	}
	delete[] tau;
	return info;
}

template<> int GEQR_p<COMPLEXf>(int m, int n, COMPLEXf *A, int ldA, COMPLEXf *Q, int *jpvt, int flag) {
	MKL_Complex8 *tau=new MKL_Complex8[MAX(m,n)];
	int matrix_order = flag==0 ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR,i,no;
	//lapack_int info = LAPACKE_zgeqp3( matrix_order,m,n,(MKL_Complex16*)A,ldA,jpvt,tau );
	lapack_int info = LAPACKE_cgeqpf( matrix_order,m,n,(MKL_Complex8*)A,ldA,jpvt,tau );
	if (info == 0) {
		COMPLEXf pivot,pivot_0=DBL_MAX;
		for (int i = 0; i < n; i++) {
			pivot = A[i*m + i];		assert(std::abs(pivot) <= std::abs(pivot_0));
			pivot_0 = pivot;
		}
	}	else {
		assert(0);
		return info;
	}

	if (Q != nullptr) {
	}
	delete[] tau;
	return info;
}

/*
	A=UTU'
	注意 
		返回时A被替换为T!!!

template<>  void SchurDecomp<COMPLEXd>( int rank,hGMAT &mA,hGMAT &mU,hGVEC &vW,COMPLEXd info,int flag )	{
	int n=rank,lda=mA->RowNum( ),sdim,i,no=-1,ldu=mU->RowNum();
	COMPLEXd *ta=TO<COMPLEXd>(mA),*tu=TO<COMPLEXd>(mU),*w=TO<COMPLEXd>(vW);
	no = LAPACKE_zgees(LAPACK_COL_MAJOR,charV,charN, nullptr,n, (MKL_Complex16*)ta, lda, &sdim,(MKL_Complex16*)w,(MKL_Complex16*)tu, ldu );
	double a,aMax=0.0;
	for( i = 0; i < n; i++ )	{
		a = abs(w[i]);
		if( a>aMax )	
		{	aMax=a;		no=i;	}
	}
	if( no!=0 )	{
		LAPACKE_ztrexc( LAPACK_COL_MAJOR, charV, n,(MKL_Complex16*)ta,lda, (MKL_Complex16*)tu,ldu,no+1,1 );
		swap( w[no],w[0] );
	}
}
*/
template <>  double vCOS<double>( const int dim,const double *a,const double *b,int flag )	{
	double na=NRM2(dim,a),nb=NRM2(dim,b),dot;
	assert( na>0 && nb>0 );
	DOT( dot,dim,a,b );
	double xita=dot/na/nb;
	return xita;
}
/*
template <>  void vMUL<double>( const int dim,double *a,double *b,double *y )	{
#ifdef MKL_LIB
	vdmul( &dim, a, b, y );
#else
	for( int i = 0; i < dim; i++ )	{
		y[i] = a[i]*b[i];
	}
#endif
}
template <>  void vMUL<float>( const int dim,float *a,float *b,float *y )	{
#ifdef MKL_LIB
	vsmul( &dim, a, b, y );
#else
	for( int i = 0; i < dim; i++ )	{
		y[i] = a[i]*b[i];
	}
#endif
}*/

template <>  void vEXP<double>( const int dim,double *Z )	{
#ifdef MKL_LIB
		vdExp( dim, Z, Z );		
#else
		for( int i = 0; i < dim; i++ )	{	Z[i] = exp(Z[i]);	}
#endif
}
template <>  void vEXP<float>( const int dim,float *Z )	{
#ifdef MKL_LIB
		vsExp( dim, Z, Z );		
#else
		for( int i = 0; i < dim; i++ )	{	Z[i] = exp(Z[i]);	}
#endif
}

template <>  void GESVD<double>( const int m,const int n,double *a,double *sval,double *u,double *v,int lda,int ldu,int ldv )	{
	double *superb=new double[MIN(m,n)];
	if( lda==0 )	lda=m;
	if( ldu==0 )	ldu=m;
	if( ldv==0 )	ldv=n;
	int iRet = LAPACKE_dgesvd(LAPACK_COL_MAJOR, charA, charA, m,n,a,lda,sval,u,ldu,v,ldv,superb );
	delete[] superb;
	if( iRet!=0 )	throw  ("GESVD failed!!!");;
}

template<> void GESVD<COMPLEXf>( const int m,const int n,COMPLEXf *a,COMPLEXf *sval_,COMPLEXf *u,COMPLEXf *v,int lda,int ldu,int ldv ){
	float *superb=new float[MIN(m,n)];
	float *sval=(float*)sval_;
	if( lda==0 )	lda=m;
	if( ldu==0 )	ldu=m;
	if( ldv==0 )	ldv=n;
	int iRet = LAPACKE_cgesvd(LAPACK_COL_MAJOR, charA, charA, m,n,(MKL_Complex8*)a,lda,
		sval,(MKL_Complex8*)u,ldu,(MKL_Complex8*)v,ldv,superb );
	delete[] superb;
	if( iRet!=0 )	throw  ("GESVD failed!!!");;
}


/*
void TEST_OOMM( ){
#define M 5
#define NNZ 13
#define N 2
		MKL_INT		m = M, nnz = NNZ,n = N;
        double		values[NNZ]	  = {1.0, -1.0, -3.0, -2.0, 5.0, 4.0, 6.0, 4.0, -4.0, 2.0, 7.0, 8.0, -5.0};
		MKL_INT		columns[NNZ]  = {0, 1, 2, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
		MKL_INT		rows[NNZ]	  = {0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4};
	 double		alpha = 1.0, beta = 0.0;
//	 double		sol[M][N]	= {1.0, 5.0, 1.0, 4.0, 1.0, 3.0, 1.0, 2.0, 1.0,1.0};
//	double		rhs[M][N]	= {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	 double *sol=new double[M*N](),*rhs=new double[M*N]();
		char transa= 't',matdescra[6]={'g','l','n','c'};;
        mkl_dcoomm(&transa, &m, &n, &m, &alpha, matdescra, values, rows, columns, &nnz, sol, &n,  &beta, rhs,  &n);
}*/

template<> void GESOP<float>( int dim,int n,float *X,double* dSop,int flag ){
	int i,inc=1;
	float one=1,*vec,*sop=new float[dim*dim]();
	for( i=0;i<n;i++ ){
		vec=X+dim*i;
		GER(dim, dim,&one, vec, inc, vec, inc, &one,sop, dim);
	}
	for( i=dim*dim-1;i>=0;i-- )dSop[i]=sop[i];
	delete[] sop;
}

template<> 
int GEINV<double>( int dim,double *X,int flag ){
	int matrix_layout=LAPACK_COL_MAJOR,i,j;
	matrix_layout=LAPACK_ROW_MAJOR;
	double *old=nullptr;
#ifdef _DEBUG
	//old=new double[dim*dim*2];
#endif
	if( old!=nullptr )		memcpy( old,X,sizeof(double)*dim*dim );
	lapack_int* ipiv=new lapack_int[dim];
	lapack_int iRet=LAPACKE_dgetrf( matrix_layout, dim, dim, X, dim,ipiv );			
	if( iRet!=0x0 )		//If info = i, uii is 0.
		return iRet;
	iRet=LAPACKE_dgetri( matrix_layout, dim, X, dim, ipiv );						
	if( iRet!=0x0 )		//If info = i, uii is 0.
		return iRet;
	if( old!=nullptr ){	//仅用于校验
		double *A=old+dim*dim,one=1,zero=0,*row=A,len=NRM2(dim*dim,old);
		GEMM( charN,charN,dim,dim,dim,&one,X,dim,old,dim,&zero,A,dim );
		for( i=0;i<dim;i++,row+=dim ){
			for( j=0;j<dim;j++ ){
				if( j==i )		
					assert( fabs(row[j]-1.0)<FLT_EPSILON*dim );
				else
					assert( fabs(row[j])<FLT_EPSILON*dim );
			}
		}
	}
	delete[] ipiv;
	if( old!=nullptr )		delete[] old;
	return iRet;
}

//float不适合求逆，统一调用GEINV<double>版本
template<> int GEINV<float>( int dim,float *X,int flag ){
	double *dX=new double[dim*dim];
	for( int i=dim*dim-1;i>=0;i-- )	dX[i]=X[i];
	int iRet = GEINV<double>( dim,dX,flag );
	for( int i=dim*dim-1;i>=0;i-- )	X[i]=dX[i];
	delete[] dX;
	return iRet;
}

template<> int POTR_FS<float>( int dim,float *X,int ldX,int nRhs,float *B,int ldB,int flag ){
	int iRet = -1,i;
	if( nRhs==0 ){
		iRet = LAPACKE_spotrf( LAPACK_ROW_MAJOR,charU,dim,X,ldX );
	}else{
		//iRet = LAPACKE_spotrs( LAPACK_ROW_MAJOR,charU,dim,nRhs,X,dim,B,1 );
		for( i=0;i<nRhs;i++ ){
			iRet = LAPACKE_spotrs( LAPACK_ROW_MAJOR,charU,dim,1,X,ldX,B+dim*i,ldB );
		}
	}
	return iRet;
}
template<> int POTR_FS<double>( int dim,double *X,int ldX,int nRhs,double *B,int ldB,int flag ){
	int iRet = -1;
	if( nRhs==0 ){
		iRet = LAPACKE_dpotrf( LAPACK_ROW_MAJOR,charU,dim,X,ldX );
	}else{
		iRet = LAPACKE_dpotrs( LAPACK_ROW_MAJOR,charU,dim,nRhs,X,ldX,B,ldB );
	}
	return iRet;
}
template<> int POTR_FS<COMPLEXd>( int dim,COMPLEXd *X,int ldX,int nRhs,COMPLEXd *B,int ldB,int flag ){
	int iRet = -1;
	if( nRhs==0 ){
		iRet = LAPACKE_zpotrf( LAPACK_ROW_MAJOR,charU,dim,(MKL_Complex16*)X,ldX );
	}else{
		iRet = LAPACKE_zpotrs( LAPACK_ROW_MAJOR,charU,dim,nRhs,(MKL_Complex16*)X,ldX,(MKL_Complex16*)B,ldB );
	}
	return iRet;
}
template<> int POTR_FS<COMPLEXf>( int dim,COMPLEXf *X,int ldX,int nRhs,COMPLEXf *B,int ldB,int flag ){
	int iRet = -1;
	if( nRhs==0 ){
		iRet = LAPACKE_cpotrf( LAPACK_ROW_MAJOR,charU,dim,(MKL_Complex8*)X,ldX );
	}else{
		iRet = LAPACKE_cpotrs( LAPACK_ROW_MAJOR,charU,dim,nRhs,(MKL_Complex8*)X,ldX,(MKL_Complex8*)B,ldB );
	}
	return iRet;
}

template<> int POTRF<COMPLEXd>( const char uplo,int dim,COMPLEXd *X,int ldX ){
	int info=0;
	ZPOTRF( &uplo, &dim, (MKL_Complex16*)X, &ldX, &info );
	return info;
}
template<> int POTRF<COMPLEXf>( const char uplo,int dim,COMPLEXf *X,int ldX ){
	int info=0;
	CPOTRF( &uplo, &dim, (MKL_Complex8*)X, &ldX, &info );
	return info;
}

template<> int GESV<COMPLEXd>(int dim, COMPLEXd *A, int nRhs, COMPLEXd *B, COMPLEXd *X, int flag) {
	lapack_int *ipiv = new int[dim];
	if (X != B) {
		memcpy(X, B, sizeof(COMPLEXd)*dim*nRhs);
	}
	int iRet= LAPACKE_zgesv(LAPACK_COL_MAJOR,dim, nRhs,(MKL_Complex16*)A,dim, ipiv, (MKL_Complex16*)X, dim );
	//assert( ipiv[0]==0 && ipiv[1]==1 && ipiv[2]==2 );
	delete[] ipiv;
	return iRet;
}
template<> int GESV<double>(int dim, double *A, int nRhs, double *B, double *X, int flag) {
	lapack_int *ipiv = new int[dim];
	if (X != B) {
		memcpy(X, B, sizeof(double)*dim*nRhs);
	}
	int iRet= LAPACKE_dgesv(LAPACK_COL_MAJOR,dim, nRhs,A,dim, ipiv, X, dim );
	//assert( ipiv[0]==0 && ipiv[1]==1 && ipiv[2]==2 );
	delete[] ipiv;
	return iRet;
}


template<> int GETRF( int dim,COMPLEXf*mA,int lda,int *ipiv,int flag ){
	assert( ipiv!=nullptr && mA!=nullptr );
	/*	[MKL]		A = P*L*U,
		L is lower triangular with unit diagonal elements U is upper triangular The routine uses partial pivoting, with row interchanges.
		[GSS]		A =  L*U,		ROW-MAJOR
		L is lower triangular;	U is upper triangular with unit diagonal 
	*/
	lapack_int iRet;
	cgetrf(  &dim, &dim, (MKL_Complex8*)mA, &lda, ipiv,&iRet );
	return iRet;
}
template<> int GETRS( int dim,COMPLEXf*mA,int lda,int *ipiv,int nRhs,COMPLEXf*b,int ldb,int flag )	{	
	lapack_int iRet;
	cgetrs( &charN, &dim, &nRhs, (MKL_Complex8*)mA, &lda, ipiv,(MKL_Complex8*)b,&ldb,&iRet );
	return iRet;
}

template<> int GETRF_r( int dim,COMPLEXf*mA,int lda,int *ipiv,int flag ){
	assert( ipiv!=nullptr && mA!=nullptr );
	/*	[MKL]		A = P*L*U,
		L is lower triangular with unit diagonal elements U is upper triangular The routine uses partial pivoting, with row interchanges.
		[GSS]		A =  L*U,		ROW-MAJOR
		L is lower triangular;	U is upper triangular with unit diagonal 
	*/
	lapack_int iRet = LAPACKE_cgetrf( LAPACK_ROW_MAJOR, dim, dim, (MKL_Complex8*)mA, lda, ipiv );
//	lapack_int iRet = LAPACKE_cgetrf( LAPACK_ROW_MAJOR, dim, dim, (MKL_Complex8*)mA, lda, ipiv );
	return iRet;
}
template<> int GETRS_r( int dim,COMPLEXf*mA,int lda,int *ipiv,int nRhs,COMPLEXf*b,int ldb,int flag ){	
	assert( ipiv!=nullptr && mA!=nullptr );
	lapack_int iRet = LAPACKE_cgetrs( LAPACK_ROW_MAJOR, charN, dim, nRhs, (MKL_Complex8*)mA, lda, ipiv, (MKL_Complex8*)b,ldb );
	return iRet;
}
template<> int GETRF_r( int dim,COMPLEXd*mA,int lda,int *ipiv,int flag ){	
	assert( ipiv!=nullptr && mA!=nullptr );
	lapack_int iRet = LAPACKE_zgetrf( LAPACK_ROW_MAJOR, dim, dim, (MKL_Complex16*)mA, lda, ipiv );
//	lapack_int iRet = LAPACKE_zgetrf( LAPACK_ROW_MAJOR, dim, dim, (MKL_Complex16*)mA, lda, ipiv );
	return iRet;
}
template<> int GETRS_r( int dim,COMPLEXd*mA,int lda,int *ipiv,int nRhs,COMPLEXd*b,int ldb,int flag ){	
	assert( ipiv!=nullptr && mA!=nullptr );
	lapack_int iRet = LAPACKE_zgetrs( LAPACK_ROW_MAJOR, charN, dim, nRhs, (MKL_Complex16*)mA, lda, ipiv, (MKL_Complex16*)b,ldb );
	return iRet;
}

template<> int SYTRF_r( int dim,COMPLEXf*a,int lda,int *ipiv,int nRhs,COMPLEXf*b,int ldb,int flag ){
	lapack_int iRet = -1;
	char uplo=charL;
	if( nRhs<=0 ){
		iRet =  LAPACKE_csytrf( LAPACK_ROW_MAJOR,uplo,dim,(MKL_Complex8*)a,lda, ipiv );
	}else{
		iRet =  LAPACKE_csytrs( LAPACK_ROW_MAJOR,uplo,dim,nRhs,(MKL_Complex8*)a,lda, ipiv,(MKL_Complex8*)b,ldb );
	}
	return iRet;
}
template<> int SYTRF_r( int dim,COMPLEXd*a,int lda,int *ipiv,int nRhs,COMPLEXd*b,int ldb,int flag ){
	lapack_int iRet = -1;
	char uplo=charL;
	if( nRhs<=0 ){
		iRet =  LAPACKE_zsytrf( LAPACK_ROW_MAJOR,uplo,dim,(MKL_Complex16*)a,lda, ipiv );
	}else{
		iRet =  LAPACKE_zsytrs( LAPACK_ROW_MAJOR,uplo,dim,nRhs,(MKL_Complex16*)a,lda, ipiv,(MKL_Complex16*)b,ldb );
	}
	return iRet;
}
