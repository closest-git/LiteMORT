#pragma once

/*
	GrusST 大型稀疏矩阵模版库 设计原则
	1 算法与容器分开
		容器包括vector,matrix,pattern等
	2 高效率
		2.1	避免大矩阵的直接复制
			private:
				GeMAT(const GeMAT&);
				GeMAT& operator=(const GeMAT&);

	3 支持多种数据类型
		采用模版

	4 简洁：
		尽量使用，返回std::shared_ptr
		4.1	避免临时变量及代码
		例如 |Ax-b|的计算通常要引入临时变量y
		|Ax-b|：NRM2(A*x-b)
		4.2 如何支持连缀操作？	

	5 支持广义矩阵 采用模版
		4.1 多个矩阵合并存储。如稀疏LU分解之后，L+U。
		4.2 矩阵隐式存在，如arnoldi迭代中的Av
		4.3 矩阵的谱变换
		



	6 灵活的构造函数

*/
#include <cassert>
#include <typeinfo>

#ifdef XMU_COMPLEX

#else
#include <complex>
	typedef std::complex<double> COMPLEXd;
	typedef std::complex<float>  COMPLEXf;

	typedef std::complex<double>	Z;
	typedef std::complex<float>		C;
	typedef float					S;
	typedef double					D;
#endif // !XMU_COMPLEX


const char charA='A',charN='N',charT='T',charC='C',charE='E',charV='V',charB='B',charL='L',charR='R',charU='U',charS='S';

/* Yes, this is exceedingly ugly.  Blame Microsoft, which hopelessly */
/* violates the IEEE 754 floating-point standard in a bizarre way. */
/* If you're using an IEEE 754-compliant compiler, then x != x is true */
/* iff x is NaN.  For Microsoft, (x < x) is true iff x is NaN. */
/* So either way, this macro safely detects a NaN. */
//#define IS_NAN(x)	(((x) != (x)) || (((x) < (x))))
template <typename T> bool IS_NAN( const T&a )		{	return std::isnan(a);		}
template<> bool IS_NAN<COMPLEXd>( const COMPLEXd &a );
template<> bool IS_NAN<COMPLEXf>( const COMPLEXf &a );

//a*a'= |a|*|a|
template <typename T> double TxTc( const T&a )		{	return a*a;	}
template<> double TxTc<COMPLEXd>( const COMPLEXd &a );
template<> double TxTc<COMPLEXf>( const COMPLEXf &a );


template <typename T> void SCALd( const int dim,const double alpha,T *X )		{	assert(0);		return;	}
template<> void SCALd<float>( const int dim,const double alpha,float *X );
template<> void SCALd<double>( const int dim,const double alpha,double *X );
template<> void SCALd<COMPLEXd>( const int dim,const double alpha,COMPLEXd *X );
template<> void SCALd<COMPLEXf>( const int dim,const double alpha,COMPLEXf *X );

template <typename T> void SCAL( const int dim,const T alpha,T *X )		{	assert(0);		return;	}
template<> void SCAL<COMPLEXd>( const int dim,const COMPLEXd alpha,COMPLEXd *X );

template <typename T> double NRM2(const int dim, const T *X) {
	double nrm = 0.0;		for (int i = 0; i<dim; i++)	nrm += X[i] * X[i];	return sqrt(nrm);
}
template<> double NRM2<double>(const int dim, const double *X);
template<> double NRM2<float>(const int dim, const float *X);
template<> double NRM2<COMPLEXd>(const int dim, const COMPLEXd *X);
template<> double NRM2<COMPLEXf>(const int dim, const COMPLEXf *X);

template <typename T> void DOT( T &dot,const int dim,const T *X,const T *Y )		{	assert(0);		return 0.0;	}
template<> void DOT<double>( double &dot,const int dim,const double *X,const double *Y );
template<> void DOT<float>( float &dot,const int dim,const float *X,const float *Y );
template<> void DOT<COMPLEXd>( COMPLEXd &dot,const int dim,const COMPLEXd *X,const COMPLEXd *Y );

template <typename T> void COPY( const int dim,const T *X,T *Y )		{	assert(0);		return;	}
template<> void COPY<double>( const int dim,const double *X,double *Y );
template<> void COPY<float>( const int dim,const float *X,float *Y );
template<> void COPY<COMPLEXd>( const int dim,const COMPLEXd *X,COMPLEXd *Y );

//很耗时间!!! in-place transposition/copying of matrices
template <typename T> void IMAT_T( const char ordering, size_t rows, size_t cols,T * AB)	{	throw "IMATCOPY is ...";		}
template <> void IMAT_T<COMPLEXf>( const char ordering, size_t rows, size_t cols,COMPLEXf * AB);

template <typename T> void SWAP( const int dim,T *X,int,T *Y,int )		{	assert(0);		return;	}
template<> void SWAP<COMPLEXf>( const int dim,COMPLEXf *X,int incx,COMPLEXf *Y,int incy );
//Y := a*X + Y
template <typename T> void AXPY( const int dim,const T alpha,const T *X,T *Y )		{	assert(0);		return;	}
template<> void AXPY<float>( const int dim,const float alpha,const float *X,float *Y );
template<> void AXPY<double>( const int dim,const double alpha,const double *X,double *Y );
template<> void AXPY<COMPLEXd>( const int dim,const COMPLEXd alpha,const COMPLEXd *X,COMPLEXd *Y );
template<> void AXPY<COMPLEXf>( const int dim,const COMPLEXf alpha,const COMPLEXf *X,COMPLEXf *Y );

//Y := a*X + b*Y
template <typename T> void AXPBY( const int dim,const T alpha,const T *X,const T beta,T *Y )		{	assert(0);		return;	}
template<> void AXPBY<float>( const int dim,const float alpha,const float *X,const float beta,float *Y );
template<> void AXPBY<double>( const int dim,const double alpha,const double *X,const double beta,double *Y );

template <typename T> void GER_11( const int M,const int N, const T alpha, const T *x, const T *y, T *a, const int lda )		{	assert(0);		return;	}
template<> void GER_11<double>( const int M,const int N, const double alpha, const double *x, const double *y, double *a, const int lda );

//参见GVMAT_t.cpp::Set<T>
template <typename T>  void vSET( const int dim,T *a,T b=0.0 ){	for( int i=0;i<dim; i++)	a[i]=b;	}

template <typename T>  double vCOS( const int dim,const T *a,const T *b,int flag=0x0 ){	assert(0);		return 0;	}
template <>  double vCOS<double>( const int dim,const double *a,const double *b,int flag );

template <typename T>  void vMUL( const int dim,T *a,T *b,T*y ){	assert(0);		return;	}
template <>  void vMUL<double>( const int dim,double *a,double *b,double *y );
template <>  void vMUL<float>( const int dim,float *a,float *b,float *y );

template <typename T>  void vEXP( const int dim,T *Z ){	assert(0);		return;	}
template <>  void vEXP<double>( const int dim,double *Z );
template <>  void vEXP<float>( const int dim,float *Z );

template <typename T>  void vSIGMOD( const int dim,T *Z ){	assert(0);		return;	}
template <>  void vSIGMOD<double>( const int dim,double *Z );
template <>  void vSIGMOD<float>( const int dim,float *Z );

template <typename T>  void vSOFTMAX( const int dim,int ld,T *Z ){	assert(0);		return;	}
template <>  void vSOFTMAX<float>( const int dim,int ld,float *Z );
template <>  void vSOFTMAX<double>( const int dim,int ld,double *Z );
template <typename T>  void vSOFTMAX_trunc( const int dim,int ld,T *Z,float thrsh ){	assert(0);		return;	}
template <>  void vSOFTMAX_trunc<double>( const int dim,int ld,double *Z,float thrsh );

template <typename T> void GEMV( char transa,const int M,const int N, const T alpha, const T *A, const int lda, const T *X, const int incx, const T beta, T *Y, const int incy )	{	assert(0);		return;	}
template<> void GEMV<double>( char transa,const int M,const int N, const double alpha, const double *A, const int lda, const double *X, const int incx, const double beta, double *Y, const int incy );
template<> void GEMV<C>( char transa,const int M,const int N, const C alpha, const C *A, const int lda, const C *X, const int incx, const C beta, C *Y, const int incy );
template<> void GEMV<Z>( char transa,const int M,const int N, const Z alpha, const Z *A, const int lda, const Z *X, const int incx, const Z beta, Z *Y, const int incy );

template <typename T> void GEAV( char transa,const int M,const int N, const T *A, const int lda, const T *X, T *Y )	{	assert(0);		return;	}
template <> void GEAV<double>( char transa,const int M,const int N, const double *A, const int lda, const double *X, double *Y );
template<> void GEAV<Z>( char transa,const int M,const int N, const Z *A, const int lda, const Z *X, Z *Y );
template<> void GEAV<C>( char transa,const int M,const int N, const C *A, const int lda, const C *X, C *Y );

//A += alpha*x*y'	rank-1 update
template<typename T> void GER( int m, int n, T *alpha, T *vX, int incx, T *vY, int incy, T *beta, T *A, int lda );	
template<>
void GER<float>( int m, int n, float *alpha, float *vX, int ldx, float *vY, int ldy, float *beta, float *A, int lda );

template<typename T> void AB2C( char transa, char transb, int m, int n, int k,T *a, int lda, T *b, int ldb,T *c, int ldc ){
	T one(1.0),zero(0.0);
	GEMM( transa,transb,m,n,k,&one,a,lda,b,ldb,&zero,c,ldc );
}

template<typename T> void COO_MM( int m, int n, int k, T *alpha, T *a, int *rowA,int*colA, int nnz,T *b, int ldb, T *beta, T *c, int ldc );
template<> void COO_MM<double>( int m, int n, int k, double *alpha, double *a, int *rowA,int*colA, int nnz,double *b, int ldb, double *beta, double *c, int ldc );
template<> void COO_MM<float>( int m, int n, int k, float *alpha, float *a, int *rowA,int*colA, int nnz,float *b, int ldb, float *beta, float *c, int ldc );

template<typename T> void C_GEMM( char transa, char transb, int m, int n, int k, const T *alpha, T *a, int lda, T *b, int ldb, const T *beta, T *c, int ldc )
{	assert(0);		return;		}	
template<> 
void C_GEMM<COMPLEXf>( char transa, char transb, int m, int n, int k, const COMPLEXf *alpha, COMPLEXf *a, int lda, COMPLEXf *b, int ldb, const COMPLEXf *beta, COMPLEXf *c, int ldc );

template<typename T> void GEMM( char transa, char transb, int m, int n, int k, const T *alpha, T *a, int lda, T *b, int ldb, const T *beta, T *c, int ldc )
{	assert(0);		return;		}	
template<> 
void GEMM<COMPLEXd>( char transa, char transb, int m, int n, int k, const COMPLEXd *alpha, COMPLEXd *a, int lda, COMPLEXd *b, int ldb, const COMPLEXd *beta, COMPLEXd *c, int ldc );
template<> 
void GEMM<COMPLEXf>( char transa, char transb, int m, int n, int k, const COMPLEXf *alpha, COMPLEXf *a, int lda, COMPLEXf *b, int ldb, const COMPLEXf *beta, COMPLEXf *c, int ldc );
template<>
void GEMM<double>( char transa, char transb, int m, int n, int k, const double *alpha, double *a, int lda, double *b, int ldb, const double *beta, double *c, int ldc );
template<>
void GEMM<float>( char transa, char transb, int m, int n, int k, const float *alpha, float *a, int lda, float *b, int ldb, const float *beta, float *c, int ldc );

template<typename T> void HERK_s( char uplo, char trans, int m, int k, T *a, int lda, T *c, int ldc,int flag );	
template<> 
void HERK_s<COMPLEXd>( char uplo, char trans, int m, int k, COMPLEXd *a, int lda, COMPLEXd *c, int ldc, int flag );
template<> 
void HERK_s<COMPLEXf>( char uplo, char trans, int m, int k, COMPLEXf *a, int lda,  COMPLEXf *c, int ldc,int flag );

template<typename T> void TRSV( char uplo,char transa, char diag, int m, T *a, int lda, T *b, int inc_ );	
template<> 
void TRSV<COMPLEXd>( char uplo,char transa, char diag, int m, COMPLEXd *a, int lda, COMPLEXd *b, int inc_ );
template<> 
void TRSV<COMPLEXf>( char uplo,char transa, char diag, int m, COMPLEXf *a, int lda, COMPLEXf *b, int inc_ );

template<>
void TRSV<double>( char uplo,char transa, char diag, int m, double *a, int lda, double *b, int inc_ );

template<typename T> void TRSM( char side,char uplo,char transa, char diag, int m, int n, const T *alpha, T *a, int lda, T *b, int ldb );	
template<> 
void TRSM<COMPLEXd>( char side,char uplo,char transa, char diag, int m, int n, const COMPLEXd *alpha, COMPLEXd *a, int lda, COMPLEXd *b, int ldb );
template<> 
void TRSM<COMPLEXf>( char side,char uplo,char transa, char diag, int m, int n, const COMPLEXf *alpha, COMPLEXf *a, int lda, COMPLEXf *b, int ldb );
template<>
void TRSM<double>( char side,char uplo,char transa, char diag, int m, int n, const double *alpha, double *a, int lda, double *b, int ldb );

template<typename T> int GELS(char trans, int m, int n,int nrhs, T* a, int lda,T* b, int ldb );
template<> int GELS<double>(char trans, int m, int n,int nrhs, double* a, int lda,double* b, int ldb );
template<> int GELS<COMPLEXf>(char trans, int m, int n,int nrhs, COMPLEXf* a, int lda,COMPLEXf* b, int ldb );
template<> int GELS<COMPLEXd>(char trans, int m, int n,int nrhs, COMPLEXd* a, int lda,COMPLEXd* b, int ldb );

template<typename T> void GELSQ(char trans, int m, int n,int nrhs, T* a, int lda,T* b, int ldb,T*Q );
template<> void GELSQ<double>(char trans, int m, int n,int nrhs, double* a, int lda,double* b, int ldb,double *Q );

template <typename T> void LARNV_seed( const int dim,T *X,int dist,int seed )		{	assert(0);		return;	}
template<> void LARNV_seed<double>( const int dim,double *X,int dist,int seed );
template<> void LARNV_seed<float>( const int dim,float *X,int dist,int seed );

template <typename T> void LARNV( const int dim,T *X,int dist=3 )		{	assert(0);		return;	}
template<> void LARNV<double>( const int dim,double *X,int dist );
template<> void LARNV<float>( const int dim,float *X,int dist );
template<> void LARNV<COMPLEXd>( const int dim,COMPLEXd *X,int dist ); 
template<> void LARNV<COMPLEXf>( const int dim,COMPLEXf *X,int dist ); 

//normal distributed random numbers
template <typename T> void GaussRNV( const int dim,T *X,T mean,T sigma,int flag )		{	assert(0);		return;	}
template<> void GaussRNV<float>( const int dim,float *X,float mean,float sigma,int flag );
template<> void GaussRNV<double>( const int dim,double *X,double mean,double sigma,int flag );
template<> void GaussRNV<COMPLEXd>( const int dim,COMPLEXd *X,COMPLEXd mean,COMPLEXd sigma,int flag );

template  <typename T> bool isReal( T val )	{	return true; }
template <> bool isReal<COMPLEXd>( COMPLEXd val )	;

//LU factorization of A[dim:dim] A is in row-major
template  <typename T> int GETRF_r( int dim,T*a,int lda,int *ipiv,int flag )			{	assert(0);		return -1;	}
template<> int GETRF_r( int dim,COMPLEXf*a,int lda,int *ipiv,int flag );
template<> int GETRF_r( int dim,COMPLEXd*a,int lda,int *ipiv,int flag );
template  <typename T> int GETRS_r( int dim,T*a,int lda,int *ipiv,int nRhs,T*b,int ldb,int flag )			{	assert(0);		return -1;	}
template<> int GETRS_r( int dim,COMPLEXf*a,int lda,int *ipiv,int nRhs,COMPLEXf*b,int ldb,int flag );
template<> int GETRS_r( int dim,COMPLEXd*a,int lda,int *ipiv,int nRhs,COMPLEXd*b,int ldb,int flag );

//LU factorization of A[dim:dim] A is in column-major
template  <typename T> int GETRF( int dim,T*a,int lda,int *ipiv,int flag )			{	assert(0);		return -1;	}
template<> int GETRF( int dim,COMPLEXf*a,int lda,int *ipiv,int flag );
template  <typename T> int GETRS( int dim,T*a,int lda,int *ipiv,int nRhs,T*b,int ldb,int flag )			{	assert(0);		return -1;	}
template<> int GETRS( int dim,COMPLEXf*a,int lda,int *ipiv,int nRhs,COMPLEXf*b,int ldb,int flag );

template  <typename T> int SYTRF_r( int dim,T*a,int lda,int *ipiv,int nRhs,T*b,int ldb,int flag )			{	assert(0);		return -1;	}
template<> int SYTRF_r( int dim,COMPLEXf*a,int lda,int *ipiv,int nRhs,COMPLEXf*b,int ldb,int flag );
template<> int SYTRF_r( int dim,COMPLEXd*a,int lda,int *ipiv,int nRhs,COMPLEXd*b,int ldb,int flag );

/*template  <typename T> bool isUnitary( int n, const hGMAT &mU,T *tau );
template<> bool isUnitary<COMPLEXd>( int n, const hGMAT &mU,COMPLEXd *tau );
template  <typename T> bool isIMat( int n, const hGMAT &mU );
template<> bool isIMat<COMPLEXd>( int n, const hGMAT &mU);
template  <typename T> bool isTriangular( int n, const hGMAT &mU,bool isUp );
template<> bool isTriangular<COMPLEXd>( int n, const hGMAT &mU,bool isUp );
template  <typename T> bool isHMat( int n, const hGMAT &mU,T *tau ){	return false;	}

template <typename T>  void GEQR(int m,int n,hGMAT &mB, const hGMAT &mA,T info, int flag=0){	assert(0);		return;	}
template<>  void GEQR<COMPLEXd>(int m,int n,hGMAT &mB, const hGMAT &mA,COMPLEXd info, int flag);
template <typename T> void SchurDecomp( int rank,hGMAT &mA,hGMAT &hU,hGVEC &vW,T info,int flag=0 ){	assert(0);		return;	}
template<> void SchurDecomp<COMPLEXd>( int rank,hGMAT &mA,hGMAT &hU,hGVEC &vW,COMPLEXd info,int flag );
*/

//AR: the upper triangle(trapezoidal) is overwritten by R.
template <typename T> int GEQR_p(int m, int n, T *AR, int ldA,T *Q,int *jpvt, int flag = 0x0);
template<> int GEQR_p<COMPLEXd>(int m, int n, COMPLEXd *A, int ldA,COMPLEXd *R,int *jpvt, int flag );
template<> int GEQR_p<COMPLEXf>(int m, int n, COMPLEXf *A, int ldA,COMPLEXf *R,int *jpvt, int flag );
//Interpolative Decomposition --  Y=Y(:, J)X		return the numeric rank
//expressing Y as linear combination of selected columns of Y		epsi-compression tolerance


template <typename T>  void GESVD( const int m,const int n,T *a,T *sval,T *u,T *v,int lda=0,int ldu=0,int ldv=0 );
template<> void GESVD<double>( const int m,const int n,double *a,double *sval,double *u,double *v,int lda,int ldu,int ldv );
template<> void GESVD<COMPLEXf>( const int m,const int n,COMPLEXf *a,COMPLEXf *sval,COMPLEXf *u,COMPLEXf *v,int lda,int ldu,int ldv );

//inversion
template <typename T> int GEINV( int dim,T *X,int flag );
template<> int GEINV<float>( int dim,float *X,int flag );
template<> int GEINV<double>( int dim,double *X,int flag );

template <typename T> int POTRF( const char uplo,int dim,T *X,int ldX )	{		assert(0);	return;		}
template<> int POTRF<COMPLEXd>( const char uplo,int dim,COMPLEXd *X,int ldX );
template<> int POTRF<COMPLEXf>( const char uplo,int dim,COMPLEXf *X,int ldX );

//Cholesky factorization of a symmetric(Hermitian) positive-definite matrix.
template <typename T> int POTR_FS( int dim,T *X,int ldX,int nRhs,T *B,int ldB,int flag );
template<> int POTR_FS<float>( int dim,float *X,int ldX,int nRhs,float *B,int ldB,int flag );
template<> int POTR_FS<double>( int dim,double *X,int ldX,int nRhs,double *B,int ldB,int flag );
template<> int POTR_FS<COMPLEXd>( int dim,COMPLEXd *X,int ldX,int nRhs,COMPLEXd *B,int ldB,int flag );
template<> int POTR_FS<COMPLEXf>( int dim,COMPLEXf *X,int ldX,int nRhs,COMPLEXf *B,int ldB,int flag );

//A会改变!!! Overwritten by the factors L and U from the factorization of A =P*L*U; the unit diagonal elements of L are not stored.
template <typename T> int GESV(int dim, T *A, int nRhs, T *B,T *X, int flag);
//A会改变!!! Overwritten by the factors L and U from the factorization of A =P*L*U; the unit diagonal elements of L are not stored.
template<> int GESV<double>(int dim, double *A, int nRhs, double *B, double *X,int flag);
template<> int GESV<COMPLEXd>(int dim, COMPLEXd *A, int nRhs, COMPLEXd *B, COMPLEXd *X,int flag);

//inversion by SVD
template <typename T> void GEINV_SVD( int dim,T *X,double thrsh,int flag );
template<> void GEINV_SVD<float>( int dim,float *X,double thrsh,int flag );
template<> void GEINV_SVD<double>( int dim,double *X,double thrsh,int flag );

//sums over all outer products
template <typename T> void GESOP( int dim,int n,T *X,double* sop,int flag );
template<> void GESOP<float>( int dim,int n,float *X,double* sop,int flag );



template <typename T> 
void OrthoDeflate( int k,T *y,T *Q,int flag=0x0 )	{	//sorensen的公式(Lemma 4.1)里居然有BUG，真是让人吃惊，真有意思！
	MEM_CLEAR( Q,sizeof(T)*k*k );
	COPY( k,y,Q );
	int i,j;
	T *q,gama,dot=0,c;
	double tao_0=abs(y[0]),sigma=tao_0*tao_0,a,tao;
	a = NRM2(k,y);		assert( a==1.0 );
	for( i = 1; i < k; i++ )	{
		q = Q+i*k;
		a = abs(y[i]);		sigma+=a*a;
		tao=sqrt(sigma);
		if( tao_0!=0 )	{
			gama = y[i]/tao/tao_0;		c=-T(gama.real(),-gama.imag());
			AXPY( i,c,y,q );
			q[i] = tao_0/tao;
		} else		{
			q[i-1] = 1.0;
		}
		tao_0 = tao;
		if( 1 )	{
			dot=0;
			for( j = 0; j <= i; j++ )	{	
				c= T(q[j].real(),-q[j].imag())*y[j];
				dot += c;
			}
			assert( abs(dot)<1.0e-15 );
		}
	}
}