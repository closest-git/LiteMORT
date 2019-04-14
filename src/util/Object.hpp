#pragma once

#include <memory>		//for shared_ptr
#include <string>
#include <vector>
#include <typeinfo>
#include <algorithm>
#include <complex>
#include <limits.h>
#include <cstring>
#include <stdio.h>  
#include <unordered_map>
//https://github.com/Tessil/hopscotch-map
#ifdef HOPSCOTCH_MAP_LIB
	#include "./Lib/hopscotch_map.h"
#endif
#include <map>
#include <math.h>
#include <float.h>
//#include "./Lib/pugiconfig.hpp"
//#include "./Lib/pugixml.hpp"
//typedef pugi::xml_document	XML_DOC;

#include <assert.h>
#include "./GST_def.h"
//#include "./Util/Dump.h"

using namespace std;


#define XMU_COMPLEX
typedef std::complex<double> COMPLEXd;
typedef std::complex<float>  COMPLEXf;
typedef std::complex<double>	Z;
typedef std::complex<float>		C;
typedef float					S;
typedef double					D;
typedef COMPLEXf	TypeK;
//typedef COMPLEXd	TypeK;


// 稀疏的荷载向量
typedef pair<int, COMPLEXd> POS_Z;
typedef vector<POS_Z> vPOS_Z;

#define ref
#define internal
#define override		virtual
#define readonly

//无需释放的资源
#define point_to

//远程资源（跨平台，分布式）
#define remote_

//必须释放的资源
#define must_free

//#define G_STR(x)		(std::to_string(x))

class GObject{	
public:
	string name;	//每个object都有name
	virtual string ToString(int format=0x0){	return name;	}
};

bool G_isFolder( const string& sFolder );

template <typename T> double ABS_1( const T&a )		{	assert(0);		return 0;	}
template<> double ABS_1<float>( const float &a );
template<> double ABS_1<double>( const double &a );
template<> double ABS_1<COMPLEXd>( const COMPLEXd &a );
template<> double ABS_1<COMPLEXf>( const COMPLEXf &a );


template <typename T> T G_SUM( const int dim,const T *X )		{		T sum=0.0;		for(int i=0;i<dim;i++ )	sum+=X[i];	return sum;	}

//stringstream通常是用来做数据转换的,相比c库的转换，它更加安全，自动和直接。
template<typename T>
std::string G_STR(const T& x){
	std::stringstream ss;
	ss<<x;				
	return ss.str( );
}

template<typename T>
T G_S2T_(const string& s,T init){
	std::stringstream ss(s);
	T x=init;
	ss>>x;				
	return x;
}

template<typename T>
void G_S2TTT_(const string& str_,vector<T>&nums,const char* seps=" ,:;{}()\t=",int flag = 0x0) {
    nums.clear();
	string str = str_;
	char *token = strtok((char*)str.c_str(), seps);
	while (token != NULL) {
		if( typeid(T)==typeid(int) ){
			int nInt;
			if( sscanf( token,"%d",&nInt)==1 )	{
				nums.push_back(nInt);
			}
		}else if( typeid(T)==typeid(double) || typeid(T)==typeid(float) ){
			double a;
			if( sscanf( token,"%lf",&a)==1 )	{
				nums.push_back(a);
			}
		}else	{
			throw "Str2Numbers is ...";
		}
		token = strtok(NULL, seps);
	}
}

template <typename T>
void G_SomeXat_( vector<int>& pos,const vector<T>& someX,const vector<T>& allX,int flag=0x0 ){
	for( T legend : someX ){
		for( int i=0;i<allX.size();i++) {
			if( legend!=allX[i] )
				continue;
			//arrV4X.push_back( make_pair(allX[i],i) );
			pos.push_back( i );
		}
	}
}

template <typename T>
int G_ClosesTick( const T& x,const vector<T>&ticks,int flag=0x0 ){
	int i,nTick=ticks.size( ),tick=-1;
	double dis=0,dis_0=DBL_MAX;
	for( i=0; i<nTick; i++ ){
		dis = abs(ticks[i]-x);
		if( dis<dis_0 ){
			dis_0=dis;		tick=i;
		}
	}
	return tick;
}


/*	
http://stackoverflow.com/questions/15735035/why-dont-vectors-need-to-be-manually-freed
	vector's destructor simply destroys every contained element, then deallocates its internal array.
	But Destroying a plain pointer doesn't destroy what it points to!!!
*/
template<typename T>
void FREE_hVECT(  vector<T*>& vecs ){
	for( T *t : vecs)
		delete t;
	vecs.clear( );
}

template<typename T>
must_free T *NEW_( size_t len,T a0 ){
	T *arr=new T[len];
	for( int i=0;i<len;i++ )	arr[i]=a0;
	return arr;
}

//0-|arr|		1-|arr.real|
template<typename T>
void MIN_MAX( size_t len,T *arr,double&a_0,double&a_1,int flag=0x0 ){
	size_t i;
	double a;
	if( flag==1 ){
		for( a_0=DBL_MAX,a_1=-DBL_MAX,i=0;i<len;i++ ){
			COMPLEXd z=arr[i];
			a = z.real( );
			a_0=MIN(a_0,a);		a_1=MAX(a_1,a);
		}
	}else{
		for( a_0=DBL_MAX,a_1=-DBL_MAX,i=0;i<len;i++ ){
			a = ABS_1(arr[i]);
			a_0=MIN(a_0,a);		a_1=MAX(a_1,a);
		}
	}
}

//delete[] array
template<typename T>
void FREE_a( T* &ptr ){
	if( ptr!=nullptr )	{
		delete[] ptr;
		ptr=nullptr;
	}
}

#ifdef HOPSCOTCH_MAP_LIB
template<typename K,typename V>
//V *HT_FIND( unordered_map<K,V*>&HT,K key){
V *HT_FIND( tsl::hopscotch_map<K,V*>&HT,K key){
	if( HT.find(key)==HT.end( ) )
		return nullptr;
	else
		return HT[key];
}
#else
template<typename K,typename V>
V *HT_FIND( unordered_map<K,V*>&HT,K key){
	if( HT.find(key)==HT.end( ) )
		return nullptr;
	else
		return HT[key];
}
#endif


