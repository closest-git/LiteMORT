#pragma once
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;
#ifdef WIN32
    #include <tchar.h>
	#include <assert.h>
#else    
	#define assert(cond)
#endif
#include "./GST_def.h"

namespace Grusoft {
	//暂时借用，是否合适?
	class DATA_SHAPE;
	typedef shared_ptr<DATA_SHAPE> hSHAPE;
	template<typename T>
	T* TO(const hSHAPE hS ) 	{
		T* ts=dynamic_cast<T*>(hS.get( ) );
		if( ts==nullptr )
			throw "TO hSHAPE hS is XXX";	//("TO hSHAPE"); 
		return ts; 
	}

	//数据的一些通道
	typedef enum {
		GST_PIXEL_3=0x100,
	}GST_DATA_MAPs;

	class DATA_SHAPE{
	public:
		unsigned int x;
	//In general the leading dimension lda is equal to the number of elements in the major dimension. It is also
	//equal to the distance in elements between two neighboring elements in a line in the minor dimension.
		typedef enum{
			UNKNOWN,SAMPLE_MAJOR=100,MAP_MAJOR=110,
			NMWH=200,	MNWH,
		}LAYOUT;
		LAYOUT layout;
		DATA_SHAPE( )
		{	x=0;	layout=UNKNOWN;	}
	//elements in tensor	(all maps,all frames...)
		virtual void Init( const string &sParam )	{	throw "DATA_SHAPE::Init( const string &sParam ) unimplemented!!!";	}
		virtual unsigned int Count( )	const					
		{	throw "DATA_SHAPE::Count unimplemented!!!";	}
	};

	/*
		Dataset和Matrix既有类似，也有区别，如何合并？		 10/19/2014		cys
		DataSample适用于机器学习，即每个样本都有标记(tag或outY）
	*/
	struct DataSample;
	typedef DataSample* hDATASET;

	struct DataSample{
		static std::string sDumpFolder,sLastPicPath;
		static const string sBackSrc;

		string nam;
		enum{
			TAG_ZERO = 0x10, 
			CHECK_DEVIA = 0x20,
			PREC_BASIC = 0x100,
			TRANS=0x200,
			DIM_MOST=0x10000,		//个数为nMost，tag无效
		};
		enum{
			MARK_0=0x0,
			MARK_REFER=0x100,
		};
		typedef enum{
			CLASIFY,REGRESS
		}TASK;

		hSHAPE shape;
		TASK task;
		int nMost,ldX,ldY,*tag,*info,avrage;
		double *outY;		//task必须为REGRESS

		double x_0, x_1, mean, devia, rOK;		//Accuracy-the proportion of correctly classified
		double *eMean;	//mean for each element

		DataSample() : nMost(0),ldX(0),ldY(0),outY(nullptr),shape(nullptr),task(CLASIFY),mean(0),devia(0),x_0(-DBL_MAX),x_1(DBL_MAX),rOK(0.0),tag(nullptr),info(nullptr),eMean(nullptr),avrage(1){}
		DataSample(int n,int ld) : nMost(n), ldX(ld),shape(nullptr),ldY(0),outY(nullptr),task(CLASIFY), mean(0), devia(0), x_0(-DBL_MAX), x_1(DBL_MAX), rOK(0.0),
			info(nullptr),tag(nullptr),eMean(nullptr),avrage(1)	{
		}
		DataSample(int n,int ldX,int ldY,int flag) : nMost(n), ldX(ldX),shape(nullptr),ldY(ldY),outY(nullptr),task(REGRESS), mean(0), devia(0), x_0(-DBL_MAX), x_1(DBL_MAX), rOK(0.0),
			info(nullptr),tag(nullptr),eMean(nullptr),avrage(1)	{
			GST_VERIFY( ldX>0 && ldY>=0,"DataSample_ldX_ldY" );
			if( ldY==0 ){
				task=CLASIFY;
			}else
				outY=new double[nMost*ldY];
		}
		DataSample(int n, hSHAPE shp) : nMost(n), ldX(shp->Count()),shape(nullptr),ldY(0),outY(nullptr),task(CLASIFY), mean(0), devia(0), x_0(-DBL_MAX), x_1(DBL_MAX), rOK(0.0),
			info(nullptr),tag(nullptr),eMean(nullptr),avrage(1)	{
		}
	
		virtual ~DataSample()	{	
			if(outY!=nullptr)		delete[] outY; 
		}
		virtual void Empty( int flag ){
			if( tag!=nullptr )	
			{	for( int i=0; i<nMost;	i++ )	tag[i]=-1;	}
			mean=0, devia=0, x_0=-DBL_MAX, x_1=DBL_MAX, rOK=0.0;
		}

		virtual bool isValidShape( )	{	return shape!=nullptr;	}

		//virtual void Stat(string sTitle, int flag = 0x0)	{ throw exception("DataSample::Stat unimplemented!!!"); }
		virtual void Decrimini(string sTitle, int flag = 0x0)	{ throw exception("DataSample::Decrimini unimplemented!!!"); }

		virtual int Shrink(int nTo, int flag)		{ throw exception("DataSample::Shrink unimplemented!!!"); }
		int SelectTT(hDATASET train, hDATASET test, int alg, int flag);
		int SelectTo(hDATASET train, vector<int>& nEach,bool isMove=false, int flag=0x0);
		int nSample() const	{
			if (nMost <= 0)	return 0;
			int i;
			for (i = 0; i < nMost && tag[i] >= 0; i++);
			return i;
		}

		virtual bool ImportFile( size_t no, const wchar_t *sPath,int tg,int flag=0x0 )	{ throw exception("DataSample::ImportFile unimplemented!!!"); }
		virtual void *tX() const	{ throw exception("DataSample::X unimplemented!!!"); }
		virtual void *Sample(size_t k) const	{ throw exception("DataSample::Sample unimplemented!!!"); }
		virtual double *Y(int k) const		{	return outY+ldY*k; }
		virtual void Copy(size_t i, const hDATASET hS, size_t no, int nSample = 1, int flag = 0x0)	{ throw exception("DataSample::Sample unimplemented!!!"); }
		virtual	void CopyInfo(size_t to, const hDATASET hS, size_t from, int nSample = 1, int flag = 0x0);
		virtual void TransTo(size_t from, const hDATASET hTarget, size_t no, int nSample = 1, int flag = 0x0) { throw exception("DataSample::TransTo unimplemented!!!"); }

		virtual int load_fp(FILE *fp, int flag)				{ throw exception("DataSample::load_fp unimplemented!!!"); }
		virtual int save_fp(FILE *fp, int flag)				{ throw exception("DataSample::save_fp unimplemented!!!"); }
		virtual void Dump(string sTitle, int flag);
		virtual int Load(const std::string sPath, int flag);
		virtual int Load(const std::wstring sPath, int flag);
		virtual int Save(const std::wstring sPath, int flag);
		virtual int Save(const std::string sPath, int flag);
		virtual int ToBmp(int epoch, int _x = 0, int flag = 0x0)	{ throw exception("DataSample::ToBmp unimplemented!!!"); }
	//	virtual int ToFolder(string sRoot, int type, int flag = 0x0);

		enum{
			NORMAL_FILTERBANK = 100,
		};
		virtual double Normal(int type, int flag)	{ throw exception("DataSample::Normal unimplemented!!!"); }
	};	

	template<typename T> 
	int STA_distribute(size_t dim, T *X_0, double&mean, double&devia,int flag = 0x0)	{
		mean = 0.0, devia = 0.0;
		if (dim == 0)
			return 0;

		double a,x_0=DBL_MAX,x_1=-DBL_MAX;;
		T *x=X_0,a2=0.0;
		size_t i;
		//DOT(a2,dim,X_0,X_0);
		for( i=0; i<dim; i++,x++ )	{
			a2 += (*x)*(*x);
			x_0 = MIN(x_0, *x);		x_1 = MAX(x_1, *x);
			mean += *x;		
		}
		if( x_0==x_1 ){
			mean=x_0;		devia=0.0;
		}else{
			mean /= dim;
			a = a2-dim*mean*mean;	assert(a>=0);
			devia = sqrt(a / dim);
		}
		return 0x0;
	}

	template<typename T>
	class DataSample_T : public DataSample {
	public:
		T *X;

		virtual void *tX() const	{ return X; }
		virtual void *Sample(size_t k) const	{ assert(k >= 0 && k < nMost);	return X + (size_t)(ldX*k); }

		DataSample_T() : DataSample(),X(nullptr){}
		DataSample_T( hDATASET src,int flag=0x0 ): DataSample(src->nMost,src->ldX) { 
			tag = new int[nMost]();				info=new int[nMost]();
			for (int i = 0; i < nMost; i++)	tag[i] = -1;
			size_t sz=(size_t)(nMost)*ldX;
			X = new T[sz]();
			if( src->nSample()>0 )		
				Copy( 0,src,0,src->nSample(),flag );
		}
		//task=CLASIFY
		DataSample_T(int n, int ld, T *x = nullptr,int flag=0x0 ) : DataSample(n, ld), X(x){
		try{		//		nMost=n,	ldX=ld,	type=tp,tag=nullptr,	X=x,	rOK=0.0;
			assert(nMost > 0 && nMost<INT_MAX/8 );
			tag = new int[nMost]();				info=new int[nMost]();
			if (!BIT_TEST(flag, TAG_ZERO))	{
				for (int i = 0; i < nMost; i++)	tag[i] = -1;
			}
			if (x == nullptr)	{
				size_t sz=(size_t)(nMost)*ldX;
				X = sz>0 ? new T[sz]() : nullptr;		//
			} else		{
				X = x;
			}
		}catch( ... )	{
			tag=nullptr;		X=nullptr;
			throw "DataSample_T is XXX";
		}
		}
		//task=REGRESS
		DataSample_T(int n, int ldX,int ldY,int flag=0x0 ) : DataSample(n,ldX,ldY,flag), X(nullptr){
		try{		
			assert(nMost > 0 && nMost<INT_MAX/8 );
			tag = new int[nMost]();				info=new int[nMost]();
			if (!BIT_TEST(flag, TAG_ZERO))	{
				for (int i = 0; i < nMost; i++)	tag[i] = -1;
			}
			size_t sz=(size_t)(nMost)*ldX;
			X = new T[sz]();		
		}catch( ... )	
		{		throw "DataSample_T_REGRESS is XXX";	}
		}

		DataSample_T( const string &sPath,int flag=0x0) : DataSample(),X(nullptr){
			SERIAL ar( sPath,true );
			Serial( ar,flag );
		}
		virtual ~DataSample_T()	{ Clear(); }
		virtual void Clear(){
			delete[] tag;		delete[](X);
			if( info!=nullptr )			delete[] info;
			if( eMean!=nullptr )		
				delete[] eMean;
			tag = nullptr;		X = nullptr;
			ldX = 0;				nMost = 0;
		}
		virtual void Empty( int type ){
			DataSample::Empty( type );
			if(X!=nullptr)	 memset( X,0x0,sizeof(T)*ldX*nMost );
		}

		virtual int load_fp(FILE *fp, int flag)	{
			if (fread(&nMost, sizeof(int), 1, fp) != 1)		{	return -10;		}
			if (fread(&ldX, sizeof(int), 1, fp) != 1)		{	return -11;		}
			if( tag!=nullptr )		{	delete[] tag;		tag=nullptr;	}
			tag = new int[nMost];
			if (fread(tag, sizeof(int), nMost, fp) != nMost)
			{			return -12;		}
			if( X!=nullptr )		{	delete[] X;		X=nullptr;	}
			size_t sz=nMost*(size_t)(ldX);
			X = new T[sz];
			if (fread(X, sizeof(T), sz, fp) != sz)
			{			return -13;		}
			if( info!=nullptr )		{	delete[] info;		info=nullptr;	}	// 8/19/2015
			info = new int[nMost];
			if (fread(info, sizeof(int), nMost, fp) != nMost)
			{			return -14;		}
			return 0x0;
		}
		virtual int save_fp(FILE *fp, int flag)	{
			int dim = nSample(),i;//
			if( BIT_TEST(flag,DIM_MOST) )
				dim = nMost;
			if (fwrite(&dim, sizeof(int), 1, fp) != 1)
			{			return -10;		}
			if (fwrite(&ldX, sizeof(int), 1, fp) != 1)
			{			return -11;		}
			if (fwrite(tag, sizeof(int), dim, fp) != dim)
			{			return -12;		}
			size_t sz,sz_1=(size_t)(dim)*ldX,ldSec=100000000,nSec=(int)ceil(sz_1*1.0/ldSec ),pos=0;
			for(i = 0; i < nSec; i++,pos+=ldSec )	{
				sz=min(sz_1,pos+ldSec)-pos;		assert(sz>0);
				if (fwrite(X+pos, sizeof(T), sz, fp) != sz )			{
					return -13;
				}
			}
			if (fwrite(info, sizeof(int), dim, fp) != dim)
			{			return -14;		}
			return 0x0;
		}
		
		virtual int Shrink(int nTo, int flag)	{
			int nFrom = nSample(), grid = nFrom / nTo, i, pos;
			if (grid <= 1)
				return nFrom;
			for (i = 0; i < nTo; i++)	{
				pos = i*grid;
				tag[i] = tag[pos];
				memcpy(X + ldX*i, X + ldX*pos, ldX*sizeof(T));
			}
			for (i = nTo; i < nFrom; i++)		tag[i] = -1;
			return nTo;
		}			

		void _mean(T *x, int flag)	{
			int i;
			T a0 = FLT_MAX, a1 = -FLT_MAX, mean = 0.0;
			for (i = 0; i < ldX; i++)	{
				a0 = min(a0, x[i]);		a1 = max(a1, x[i]);
				mean += x[i];
			}
			mean /= ldX;
			for (i = 0; i < ldX; i++)	x[i] -= mean;
		}
		//３倍方差内的数据缩放到[0.1-0.9]
		virtual void Sigma_Scal(int alg, int flag)	{
			int i, nz = ldX*nSample();
			T *x = X;
	//		T a0 = (T)1.0e100, a1 = -(T)1.0e100;
			T a0 = x[0],		a1 = x[0];
			mean = 0.0, devia = 0.0;
			for (i = 0; i < nz; i++)	{
				a0 = min(a0, x[i]);		a1 = max(a1, x[i]);
				mean += x[i];
			}
			x_0 = a0;		x_1 = a1;
			mean /= nz;
			for (i = 0; i < nz; i++)	{
				x[i] -= mean;		devia += x[i] * x[i];
			}
			devia = sqrt(devia / nz);
			double d3 = 3.0*devia, a;
			x_0 = DBL_MAX;		x_1 = -DBL_MAX;
			mean = 0.0;
			for (i = 0; i < nz; i++)	{
				x[i] = MIN(d3, x[i]);		x[i] = MAX(-d3, x[i]);
				a = x[i] / d3;
				x[i] = (a + 1.0)*0.4 + 0.1;
				assert(x[i] >= 0.1 && x[i] <= 0.9);
				x_0 = MIN(x_0, x[i]);		x_1 = MAX(x_1, x[i]);
				mean += x[i];
			}
			mean /= nz;
		}
		//似乎应由copy替代
		virtual void TransTo(size_t from, const hDATASET hTarget, size_t no, int nSample=1, int flag=0x0)	{	
			DataSample_T<float> *hFloat=dynamic_cast<DataSample_T<float> *>(hTarget);
			T *fromX = X + from*ldX;
			int i,nz=nSample*ldX;
			memcpy(hTarget->tag + no, tag + from, sizeof(int)*nSample);
			if( hFloat!=nullptr ){	//DATASET_cr::TransTo与保持一致
				float *toX=(float *)(hFloat->X)+no*ldX;
				for (i = 0; i < nz; i++)	{
					toX[i] = fromX[i];			//3/11/2016		不再/255.0,莫名其妙
				//	toX[i] = fromX[i]/255.0;
				}
			}else
				hTarget->Copy(no, this, from, nSample, flag);	
			if( outY!=nullptr ){
				memcpy( hTarget->outY+no*ldY, outY+from*ldY, sizeof(double)*ldY*nSample );
			}
		}

		//核心函数
		virtual void Copy(size_t to, const hDATASET hS, size_t from, int nSample = 1, int flag = 0x0)	{
			assert(to >= 0 && to < nMost && from >= 0 && from < hS->nSample());
			//DATASET_cr *hCR=dynamic_cast<DATASET_cr*>(hS);
			void *hCR = nullptr;
			size_t pS=ldX*(size_t)(to),pF=ldX*(size_t)(from);
			assert( pS>=0 && pF>=0 );
			if( hCR!=nullptr ){
				BIT_8 *pixel=(BIT_8 *)(hS->tX())+pF;
				T *X_1 = X+pS;//to*ldX;
				int k,nz=nSample*ldX;
				for( k=0; k<nz; k++ )	{
					X_1[k]=pixel[k];
				}
			}else{
				assert(ldX == hS->ldX);
				T *fromX = (T*)hS->tX();
				memcpy(X + pS, fromX + pF, ldX*sizeof(T)*nSample);
			}
			memcpy(tag + to, hS->tag + from, sizeof(int)*nSample);
			if( info!=nullptr && hS->info!=nullptr )
				memcpy(info + to, hS->info + from, sizeof(int)*nSample);
			if( outY!=nullptr ){
				memcpy(outY+to*ldY, hS->outY+from*ldY, sizeof(double)*ldY*nSample);
			}
			return;
		}

	
	};

	/*
	struct FeatData : public DataSample_T<float> {
		enum {
			ERR_OOB = -1
		};
		vector<int> permut;
		bool qOrder;
		void *hBase=nullptr;
		int *mark = nullptr;
		int ldF=-1;		//必须由Reshape初始化
		float *feats, err, *distri;
		void Reshape(int flag = 0x0);
		float *Feat(int no) {
			assert(no >= 0 && no < nFeat);
			return feats + no*ldF;
		}
		
	public:
		static FeatData* read_json(const string &sPath, int flag);

		int nFeat, nCls;
		vector<int>nEach;
		FeatData(const string &sPath, int flag);
		FeatData(int n, int ldX, int ldY, int nC, int flag) : DataSample_T<float>(n, ldX, ldY, flag),
			hBase(nullptr), feats(nullptr), mark(nullptr), distri(nullptr), qOrder(false) {
			nFeat = ldX;		nCls = nC;
			mark = new int[n];
			if (nCls > 0) {		//仅适用于multiclass
				for (int i = 0; i < nC; i++)	nEach.push_back(0);
				distri = new float[n*nCls]();
			}
		}
		FeatData(void *hB, int n, int ld, int nC, int flag) : DataSample_T<float>(n, ld, nullptr),
			hBase(hB), feats(nullptr), mark(nullptr), distri(nullptr), qOrder(false) {
			nFeat = ld;		nCls = nC;
			mark = new int[n];
			if (nCls > 0) {		//仅适用于multiclass
				for (int i = 0; i < nC; i++)	nEach.push_back(0);
				distri = new float[n*nCls]();
			}
		}
		virtual ~FeatData() {
			if (feats != nullptr)		delete[] feats;
			if (mark != nullptr)		delete[] mark;
			if (distri != nullptr)		delete[] distri;
			permut.clear();
		}
		virtual void Distri2Tag(int flag = 0x0);
		virtual void Shuffle(int flag = 0);
		//virtual void Mark2Bmp(BMPP* bmp, int type, int flag = 0);
		//virtual void ToBmp( SHAPE_IMAGE si,int type,int flag=0x0 );
		//virtual int Serial( SERIAL&ar,int flag=0x0 );
	};*/
}

