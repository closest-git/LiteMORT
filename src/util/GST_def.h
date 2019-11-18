#ifndef _GST_DEF_H_
#define _GST_DEF_H_

#include <stdint.h>
#include <math.h>
#include <vector>
#include <string>

typedef uint8_t BIT_8;
typedef uint64_t BIT_64;
typedef uint32_t BIT_32;
typedef int32_t INT_32;
typedef uint64_t UINT_64;
typedef int64_t G_INT_64;

//描述下降的方向，似乎float也可以，以节省内存
//typedef double tpDOWN;
typedef float tpDOWN;
typedef uint8_t tpFOLD;
//
typedef double tpY;
//typedef float tpY;
typedef enum { is_XXX,	is_int, is_float, is_char,is_double } IS_TYPE;

#if (defined _WINDOWS) || (defined WIN32)
typedef int (*ON_TRAVEL_wpath_)(void* user_data,const wchar_t *sPath,int flag);

#define GST_THROW(s) {													\
	char sInfo[1024]="exception";										\
	sprintf_s<1024>(sInfo, "\nEXCEPTION@%s(%d): %s\n",__FILE__,__LINE__,s);		\
	printf("%s",sInfo);												\
	throw sInfo;														\
}
#else
    #define GST_THROW(s)    {   throw s;	   }
#endif
#ifdef WIN32
	#include <assert.h>
	#define GST_VERIFY(isTrue,info) {	if(!(isTrue)) {		GST_THROW(info);		/*assert(0);*/ }	}
	#define LOGP printf
#else
	#define GST_VERIFY(isTrue,info) {	if(!(isTrue)) {		GST_THROW(info);}	}
	//#define G_INT_64  long long 
#endif


//似乎比std::min std::max更适合比较不同的数据类型		7/11/2015	cys
//#undef MAX2
//#undef MIN2
#define MAX2(a, b)    ( (a)>(b)?(a):(b) )
#define MIN2(a, b)    ( (a)<(b)?(a):(b) )

#define ZERO_DEVIA(y_0,y_1)	( (y_0) == (y_1) || fabs((y_0)-(y_1))<1.0e-6*(fabs(y_0)+fabs(y_1)) ) 
//#define ZERO_DEVIA(y_0,y_1)	( (y_0) == (y_1) ) 
//#define FLOA_64BIT



#define GST_OK 0x0

#define PI 3.1415926535897932384626433832795
#define EXP_UNDERFLOW	(-708)
#define EXP_OVERFLOW	(709)
#define IOFFE_epsi (1.0e-12)

//x is less than(-1), equal to(0), or greater than(0) y
#define CompareTo(x,y)	( (x)==(y) : 0 ? (x)<(y)?-1 : 1 )

#if defined (WIN32)
	/* Yes, this is exceedingly ugly.  Blame Microsoft, which hopelessly */
	/* violates the IEEE 754 floating-point standard in a bizarre way. */
	/* If you're using an IEEE 754-compliant compiler, then x != x is true */
	/* iff x is NaN.  For Microsoft, (x < x) is true iff x is NaN. */
	/* So either way, this macro safely detects a NaN. */
	#define SCALAR_IS_NAN(x)	(((x) != (x)) || (((x) < (x))))
	#define IS_ZERO(x)	(((x) == 0.) && !SCALAR_IS_NAN(x))
	#define IS_NONZERO(x)	(((x) != 0.) || SCALAR_IS_NAN(x))
	#define IS_LTZERO(x)	(((x) < 0.) && !SCALAR_IS_NAN(x))
//http://jacksondunstan.com/articles/983
	#define IS_NAN_INF(x)	( (x)*0!=0 )
	#define IS_FLOAT(x)		( (x)*0==0 )
#else
/* These all work properly, according to the IEEE 754 standard ... except on */
/* a PC with windows.  Works fine in Linux on the same PC... */
	#define SCALAR_IS_NAN(x)	((x) != (x))
	#define SCALAR_IS_ZERO(x)	((x) == 0.)
	#define SCALAR_IS_NONZERO(x)	((x) != 0.)
	#define SCALAR_IS_LTZERO(x)	((x) < 0.)

	#define IS_NAN_INF(x)	( (x)*0!=0 )
	#define IS_FLOAT(x)		( (x)*0==0 )
#endif


//位操作
#define BIT_FLAG_1				0x00000001
#define BIT_FLAG_H				0x80
#define BIT_SET( val,flag ) ((val) |= (flag))	
#define BIT_RESET( val,flag ) ((val) &= (~(flag)) ) 
#define BIT_TEST( val,flag ) (((val)&(flag))==(flag))
#define BIT_IS( val,flag ) (((val)&(flag))!=0)

//(r,c) <==> pos	行优先排序		5/30/2008
#define G_RC2POS( r,c,ld )	((r)*(ld)+(c))
#define G_POS2R( pos,ld )		((pos)/(ld))
#define G_POS2C( pos,ld )		((pos)%(ld))
#define G_RC_VALID( r,r_0,r_1 )	((r)>=(r_0) && (r)<=(r_1) )
#define G_RC_NORMAL( r,r_0,r_1 )	{(r)=MAX2((r),(r_0));(r)=MIN2((r),(r_1)); }

#define G_MINMAX(y,y0,y1)			{(y1)=MAX2((y),(y1));(y0)=MIN2((y),(y0)); }

//返回距a最近的整数，采用(int)操作(舍去小数部分)		2/11/2010	cys
//#define G_DOUBLE2INT(a)			(int)( (a)+0.5 )
#define G_DOUBLE2INT(a)			(int)( a>=0.0 ? (a)+0.5 : (a)-0.5 )
#define G_DOUBLE2BYTE(a)			( (a)>255.5 ? 0xFF : (G_U8)((a)+0.5) )
#define G_RADIEN2DEGREE( a )	(int)( (a)<0.0 ? (6.283185307179+(a))*57.295779513082+0.5 : (a)*57.295779513082+0.5 )

#define G_CLOCK2SEC(a)			( (a)*1.0/CLOCKS_PER_SEC )
#define G_TV2SEC(a,b)			( ((a).tv_sec-(b).tv_sec + (double)((a).tv_usec-(b).tv_usec)/1.0e6) )

#define G_COPY(dst,src,size)		{		memcpy(	(dst),(src),(size) );			}

#define I2STR( i )				(std::to_string(_ULonglong(i)).c_str())


/****	std::string 读写	****/
#define SREAD_i(str,num,err_info)	{ GST_VERIFY( sscanf( (str).c_str(),"%d",&(num) )==1,(err_info) );	}
#define SREAD_d(str,num,err_info)	{ GST_VERIFY( sscanf( (str).c_str(),"%lf",&(num) )==1,(err_info) );	}

/****	文件读写	****/
#define FREAD_I64(num,fp)	{ GST_VERIFY( fread( &(num),sizeof(int64_t),1,(fp) )==1,"" );	}
#define FWRIT_I64(num,fp) { GST_VERIFY( fwrite( &(num),sizeof(int64_t),1,(fp) )==1,"" );	}

#define FREAD_i(num,fp)	{ GST_VERIFY( fread( &(num),sizeof(int),1,(fp) )==1,"" );	}
#define FWRIT_i(num,fp) { GST_VERIFY( fwrite( &(num),sizeof(int),1,(fp) )==1,"" );	}

#define FREAD_f(num,fp)	{ GST_VERIFY( fread( &(num),sizeof(float),1,(fp) )==1,"" );	}
#define FWRIT_f(num,fp) { GST_VERIFY( fwrite( &(num),sizeof(float),1,(fp) )==1,"" );	}
#define FREAD_d(num,fp)	{ GST_VERIFY( fread( &(num),sizeof(double),1,(fp) )==1,"" );	}
#define FWRIT_d(num,fp) { GST_VERIFY( fwrite( &(num),sizeof(double),1,(fp) )==1,"" );	}

#define FREAD_arr(arr,size,count,fp) { GST_VERIFY( fread( (arr),(size_t)(size),(size_t)(count),(fp) )==((size_t)count),"" );	}
#define FWRIT_arr(arr,size,count,fp) { GST_VERIFY( fwrite( (arr),(size_t)(size),(size_t)(count),(fp))==((size_t)count),"" );	}

/*	读buffer */
//#define FREAD_i(num,fp) if( (iRet=fread( &(num),sizeof(int),1,(fp) ))!=1 )	throw -104;
#define READ_buf(dst,size,pb) {		memcpy((dst),(pb),(size) );	pb+=size;		}


typedef enum{
	CPU_MODE=0x0,		//cpu mode
	GPU_MODE,		//gpu mode
	HPU_MODE
}GST_PROCESS_UNIT;

//各种测度
typedef enum {
	MESER_L2=0x1,MESER_COS=10,
}GST_MEASURE;

#ifdef _WINDOWS
	#define LOGE(msg) printf(msg);
	#define LOGI(msg) printf(msg);
	#define LOGD(msg) printf(msg);
#elif defined ANDROID
    #define ASSERT(expression)
    #define assert(expression)
    #define stricmp strcasecmp
	#define strnicmp strncasecmp

    #include <jni.h>
    #include <android/log.h>

    #define LOGE(msg) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, msg)
    #define LOGI(msg) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, msg)
    #define LOGD(msg) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, msg)
    extern INT_32 gd_level;
    #define LOGP(...) { if(gd_level>0) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__);}
#endif
#endif
