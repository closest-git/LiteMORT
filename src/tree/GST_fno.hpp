#pragma once

#include <vector>
template<typename T>
struct FLOA_NO{
	T val;	int pos; 
	union{
		int x_1;
		int c;
		float f_1;
	};
	union{
		int x_2;
		int r;
		float f_2;
	};
	static bool isBig( const FLOA_NO<T> &l,const FLOA_NO<T> &r)			{	return l.val>r.val;	}
	static bool isSmall( const FLOA_NO<T> &l,const FLOA_NO<T> &r)			{	return l.val<r.val;	}
	static bool isPSmall( const FLOA_NO<T> *l,const FLOA_NO<T> *r)			{	return l->val<r->val;	}
	static bool isPBig( const FLOA_NO<T> *l,const FLOA_NO<T> *r)			{	return l->val>r->val;	}
	FLOA_NO( ) : val(0),pos(0),x_1(-1),x_2(-2){}
	FLOA_NO( T v,int n ) : val(v),pos(n),x_1(-1),x_2(-2){}
	FLOA_NO( T v,int n,int x1 ) : val(v),pos(n),x_1(x1),x_2(-2){}
	FLOA_NO( T v,int n,int x1,int x2 ) : val(v),pos(n),x_1(x1),x_2(x2){}
	FLOA_NO( const FLOA_NO<T>& fn) : val(fn.val),pos(fn.pos),x_1(fn.x_1),x_2(fn.x_2)	{;}

	static void ARR2FNOs( std::vector<FLOA_NO<T>>& fnos,int dim,T* arr,bool isOrder,int flag=0 )	{
		fnos.clear( );		fnos.reserve(dim);
		for( int i=0;i < dim; i++)	fnos.push_back( FLOA_NO<T>(arr[i],i) );
		if( isOrder )
			sort( fnos.begin(),fnos.end( ),FLOA_NO<T>::isBig );
	}
	/*bool isIn( std::vector<FLOA_NO<T>>& fnos,int type )	{	//不好的设计		1/22/2016
		for each( FLOA_NO<T> fn in fnos )	
		{	if( fn.pos==pos )	return true;		}
		return false;
	}*/
};
typedef FLOA_NO<float>F4NO;
typedef FLOA_NO<float>* pF4NO;
typedef FLOA_NO<double>F8NO;
typedef std::vector<F4NO> arrFNO;
typedef std::vector<F4NO*> arrPFNO;	
typedef std::vector<F8NO> arrDNO;

//暂时借用，是否合适?
typedef arrFNO POLYGON;


