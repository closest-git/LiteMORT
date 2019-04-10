// GiFace.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <io.h>
#include <vector>
#include <iostream>
#include <sstream>
#include "GruST\image\GST_bitmap.hpp"
#include "GruST\image\BMPfold.hpp"
#include "GruST\learn\MetricLearn.hpp"

#include "RF_ShapeRegress.h"

#include "GruST/face/FaceAlign_.h"
#include "GruST/face/FaceMorph.h"
#include "GruST/face/HeadPose.h"
#include "GruST/face/FacePose.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

using namespace std;
using namespace Grusoft;
/*
#if defined _WIN64
	#ifdef _DEBUG
		#pragma comment( lib, "F:\\GiPack\\lib\\x64\\Debug\\dlib.lib" )
	#else
		#pragma comment( lib, "F:\\GiPack\\lib\\x64\\Release\\dlib.lib" )
	#endif
#else	
#endif*/

#define _LIMIT_ENTRY_ 100000
#define _FACE_MARK_0_		68 
#define _FACE_MARK_			51 
#define _FACE_NW_			256 
#define _FACE_NH_			256 
static SHAPE_IMAGE siFace(_FACE_NW_,_FACE_NH_,3);	
static char sPath[MAX_PATH*2];

static bool isDetecFace=false;	
static bool isTestLigner=false;
static bool isTestMorph=false;
//未见效果，真奇怪。似乎是需要放宽边界，迭代过程中，有些点的位置在roi之外		1/6/2015
static bool isNormalFace=false;	


#ifdef _DEBUG
	//static int nMaxTrain=5000;	
	static int nMaxTrain=100;
	#define nMaxTest	10
#else
	static int nMaxTrain=10000;	
	//#define nMaxTrain	100
	#define nMaxTest	1000
#endif
static vector<wstring> g_arrPath;
static wstring wDumpFolder=L"F:\\GiFace\\trace\\";
static string sDumpFolder="F:\\GiFace\\trace\\";
static dlib::frontal_face_detector facor = dlib::get_frontal_face_detector();

bool isImageFile( wchar_t *path )	{
	wchar_t ext[_MAX_EXT];
	int len = _tcslen( path );
	if( len<4 || path[len-4]!='.' )
		return false;
	_tcscpy_s( ext,_MAX_EXT,path+len-3 );
	_tcslwr_s( ext,_MAX_EXT );
	if ( (_tcsnicmp(ext,_T("bmp"),3)==0 ) || (_tcsnicmp(ext,_T("jpg"),3)==0) || (_tcsnicmp(ext,_T("png"),3)==0 ) )
		return true;
	return false;
}
void _List_File_( wstring sCurDir,int filter,int flag )		{
	try{
    HANDLE hSearch=NULL;
    WIN32_FIND_DATA data;
	int no=0,type,nzOld,nzDir=0,len,nFind=0,folder=-1;
	wchar_t sPath[_MAX_PATH];
	vector<wstring> arrFolder;
	if( SetCurrentDirectory( sCurDir.c_str() )==0 )	{
		int code = GetLastError();
		return;	//
	}

	_stprintf( sPath,_T("%s*"),sCurDir.c_str() );

	hSearch=FindFirstFile(sPath ,&data);	
	do{
		if( BIT_TEST(data.dwFileAttributes,FILE_ATTRIBUTE_DIRECTORY) )	{
			len = _tcslen (data.cFileName );
			if( len<=2 && 
				(_tcscmp (data.cFileName,_T("."))==0 || _tcscmp (data.cFileName,_T(".."))==0) )
				continue;	
			if( data.cFileName[0]=='$' )	//Each NTFS metadata file begins with $ !
				continue;
			if( data.cFileName[0]=='_' )	{	
				//_vti_cnf,_vti_pvt,_vti_script,_vti_txt,FrontPage server side extensions 
				if( data.cFileName[1]=='v' && data.cFileName[2]=='t' && data.cFileName[3]=='i' )
					continue;
			}
			if( len>4 && data.cFileName[len-4]=='.' )	{
				wchar_t *sFix = data.cFileName+len-3;
				int isPass = _tcscmp(sFix,_T("cab"))==0 || _tcscmp(sFix,_T("zip"))==0;
				if( isPass==1 )
				{	continue;	}
			}

			if( data.cFileName[len-1]!='\\' )	{
				_stprintf( sPath,_T("%s%s\\"),sCurDir.c_str(),data.cFileName );		
			} else			{
				_stprintf( sPath,_T("%s%s"),sCurDir.c_str(),data.cFileName );	//	sPath = sCurDir+data.cFileName;	
			}
			arrFolder.push_back( sPath );
		}else	{
			if( isImageFile(data.cFileName) ) {
				_stprintf( sPath,_T("%s%s"),sCurDir.c_str(),data.cFileName );
				if( g_arrPath.size( )>=_LIMIT_ENTRY_ )
					break;
				g_arrPath.push_back( sPath );						
			}
		}
	}while(FindNextFile(hSearch,&data));
    FindClose(hSearch);
	int nFolder = arrFolder.size( ),i;
	for( i = 0; i < nFolder; i++ )	{
		_List_File_( arrFolder[i],filter,flag );
    }  
	arrFolder.clear( );
	}
	catch( ... )	{
		int i = 0;
	}
}

void GetPaths( _TCHAR *sInputPath,bool isDir,int flag )    {
    g_arrPath.clear( );    
    wstring sPath=sInputPath;
    int ret;
    if( isDir ) {
		if( sInputPath[_tcslen(sInputPath)-1]!='\\' )
			sPath += '\\';
        _List_File_( sPath,0,flag );
    }else       {
		assert(0);
    }
    std::sort( g_arrPath.begin(),g_arrPath.end() );
}

char *W2CHAR(const wchar_t *wtxt,char *buffer )	{
	if (wtxt == NULL)
		throw "GST_util::W2CHAR -- str is NULL!!!";
	int len = _tcslen(wtxt);
	if (len>0)	{
#ifdef WIN32
		len = ::WideCharToMultiByte(CP_UTF8, 0, wtxt, len, buffer, 1000,0,0);
#else
		throw "GST_util::W2CHAR";
#endif
	}
	buffer[len] = '\0';
	return buffer;
}

//#include "ge_face_detect.h"
extern "C" int FaceLocate_ge( BIT_8 *pixel,int wth,int hei,int*left,int*rigt,int*botom,int *top,INT_32 detect );

static int nFaceDetect=0;
//暂时取最大人脸
bool FaceDetect_( GST_BMP *hSrc,BIT_8 *gray,int wth,int hei,RoCLS &roi,int flag=0x0 ){
	int left,rigt,botom,top,i,box_0=0;
	double a;
	GST_TIC( tick );
	if( 0 ){	//问题太多 1 不支持64-bit 2 准确率差
		if( FaceLocate_ge( gray,wth,hei,&left,&rigt,&botom,&top,0x0 )>0 ){	//取框偏大
		}else
			return false;	
	}else{		//debug的时间实在太长，莫名其妙
		dlib::array2d<unsigned char> img(hei,wth);
		BIT_8 *pixel=&(img[0][0]);
		int r;
		for( r=0;r<hei;r++){	//flip vertical
			memcpy( pixel+r*wth,gray+(hei-1-r)*wth,sizeof(BIT_8)*wth );	
		}
		std::vector<dlib::rectangle> dets = facor(img);
	/*	if( dets.size()==0 ){
			pyramid_up(img);
			dlib::save_jpeg(img, "F:\\GiFace\\trace\\gray.jpg");
			dets = facor(img);
		}*/
		if( dets.size()==0 )
			return false;	
		for( i=0;i<dets.size( );i++ )	{
			if( dets[i].width()>box_0 )	{
				left=dets[i].left(),rigt=dets[i].right(),botom=hei-1-dets[i].bottom(),top=botom+dets[i].height();
				box_0=dets[i].width();
			}
		}
	}
	if( (a=GST_TOC( tick ))>1.0 )		printf( "time=%g,",a );
	roi = RoCLS(left,botom,rigt-left,top-botom,0,0);
	if( 0 && hSrc!=nullptr ){
		BMP_DRAW *hTrace=new BMP_DRAW(hSrc);		//BMP_DRAW有内存泄漏
		hTrace->Line( left,rigt,botom,botom );		hTrace->Line( rigt,rigt,top,botom );
		hTrace->Line( left,rigt,top,top );			hTrace->Line( left,left,top,botom );
		//printf( "%d",rigt-left );
		char sPath[1000];
		sprintf( sPath,"F:\\GiFace\\trace\\%d.bmp",nFaceDetect++ );
		hTrace->Save( sPath );
		delete hTrace;
	}
	return true;
}

static char *sObjTrainDir = ("F:\\GiFace\\train\\");
static char *sObjTestDir = ("F:\\GiFace\\test\\");

static wchar_t *sTrainSetDir = _T("G:\\face\\ibug\\trainset\\");
//static wchar_t *sTrainSetDir = _T("G:\\face\\1000\\");
//static wchar_t *sPicDir = _T("G:\\face\\ibug\\testset\\");		//

bool NormalizeFace( ShapeBMPfold* hFold,RoCLS &roi,double *pts,int flag=0x0 ){
	hBMPP hbm=hFold->hbmp;
	int i,r,c,r0,c0,W0=hbm->Width(),H0=hbm->Height();
	//roi.Expand( 0.5 );
	SHAPE_IMAGE *si=hbm->spIMAGE( );	si->wth=roi.wth;		si->hei=roi.hei;
	hBMPP hSub=new BMPP( si );
	BIT_8 cr,cg,cb;
	for( r=0;r<si->hei;r++){
	for( c=0;c<si->wth;c++){
		c0=c+roi.c_0;	r0=r+roi.r_0;
		if( G_RC_VALID(c0,0,W0-1) && G_RC_VALID(r0,0,H0-1) )
		{	hbm->GetPixelColor( c0,r0,&cr,&cg,&cb );	}
		else{
			cr=0,		cg=0,		cb=0;
		}
		hSub->SetPixelColor( c,r,cr,cg,cb );
	}
	}
	double sZoom=hSub->Width( );
	hSub->ToSize( _FACE_NW_,_FACE_NH_ );
	sZoom=hSub->Width( )/sZoom;
	bool isDump=true;
	r0=roi.r_0+roi.hei/2;		c0=roi.c_0+roi.wth/2;
	for( i=0; i<_FACE_MARK_0_; i++ ){	
		pts[2*i]	=(pts[2*i]-roi.c_0)*sZoom;		pts[2*i+1]	=(pts[2*i+1]-roi.r_0)*sZoom;
	//	if( isDump )	hSub->SetPixelColor( pts[2*i],pts[2*i+1],0,255,0 );
	}
	if( isDump ){
		hSub->Save( "G:\\face\\ibug\\normals\\"+hFold->nam );
	}
	hFold->hbmp=hSub;			delete hbm;
	RoCLS box(0,0,hSub->Width(),hSub->Height());
	hFold->SetShapeMarks( pts,_FACE_MARK_0_,&box );		//sfold->UnFold( );
	if( hFold->lenUnit<0.3 )
		printf( "STRANGE lenunit=%g @\"%s\"\n",hFold->lenUnit,hFold->nam.c_str() );

	return true;
}

static GST_BMP::INITIAL szInital(FaceAligner::wMax/2,FaceAligner::hMax/2);
static FaceAligner falign("F:\\GiFace\\ligner\\rf_D.dat","F:\\GiFace\\ligner\\rf_T.dat",0x0 );
ShapeBMPfold *Fold_300wMarks( const string&sPath,int flag=0x0 ){
	hBMPP hbmp = new BMPP( sPath,szInital );		hbmp->Median( );
	ShapeBMPfold *sFold=new ShapeBMPfold(hbmp,FaceAligner::spFace0 );	
	float x,y,*sX=sFold->vY.data(),*sY=sX+_FACE_MARK_0_;
	string sMark=sPath.substr(0,sPath.size()-3)+"pts";
	int i,nPt=0,wth=hbmp->Width( ),hei=hbmp->Height( ),c0=INT_MAX,c1=-INT_MAX,r0=INT_MAX,r1=-INT_MAX;
	char sLine[1000];
	FILE *fp=fopen( sMark.c_str(),"rt");
	if( fp!=NULL ){
		while( fgets( sLine,1000,fp )){
			if( sscanf( sLine,"%f %f",&x,&y)==2 ){
				x=x*hbmp->init.sW;						
				y=hei-y*hbmp->init.sH;			//"pts"文件的坐标原点在左上角
				c0=MIN(c0,x);	c1=MAX(c1,x);			r0=MIN(r0,y);	r1=MAX(r1,y);
				sX[nPt]=x;			sY[nPt]=y;			nPt++;
			}
		}		
		fclose( fp );
		assert( nPt==_FACE_MARK_0_ );			
		sFold->ReSize( _FACE_MARK_ );			
		sX=sFold->vY.data();		sY=sX+_FACE_MARK_;
		sFold->InterocDis( );			
	}else
		return nullptr;
	RoCLS roiMark(c0,r0,c1-c0+1,r1-r0+1);		
	falign.nam=sPath;		falign.roiTo=roiMark;
	if( falign.Solve( hbmp,nullptr,0x0 )==FaceAligner::OK )	{
		sFold->roi=falign.roi;
		if( !sFold->roi.isOverLap(RoCLS( c0,r0,c1-c0,r1-r0) )){			
			delete sFold;	return nullptr;	
		}
		/*if( isNormalFace )
			NormalizeFace( sfold,roi,pts );
		else
			sFold->SetShapeMarks( pts,_FACE_MARK_0_,&roi );	*/
		for( i=0;i<_FACE_MARK_;i++ ){
			sFold->roi.UnMap1( sX[i],sY[i],sX[i],sY[i] );
			if( i==0 ){
				assert( sX[i]<0 && sY[i]>0 );
			}
		}
		sFold->InterocDis( );			
		if( 0 )	{
			sMark+="_n";
			sFold->Serial( sMark,false );	
		}
	//		sfold->Serial( sMark,false );
	}else
	{	delete sFold;	return nullptr;	}
	return sFold;
}

bool ReadLandMarks( string sPicPath,ShapeBMPfold* sfold,bool isDetect,string sOutDir,int flag=0x0 ){
	bool bRet=false;
	int len=sPicPath.size(),nPt=0,c0=INT_MAX,c1=-INT_MAX,r0=INT_MAX,r1=-INT_MAX;
	string sMark=sPicPath.substr(0,len-3)+"pts";
	if( !isDetect  ){
		sMark=sPicPath.substr(0,len-3)+"pts_n";
		if( sfold->Serial( sMark )==0 )		{
			if( _FACE_MARK_!=_FACE_MARK_0_ )
				sfold->ReSize( _FACE_MARK_ );
			if( sfold->lenUnit<0.3 )
				printf( "strange lenunit=%g @ %s\n",sfold->lenUnit,sfold->nam.c_str() );
			return true;
		}	else {
			if( isTestLigner){
				hBMPP hbm=sfold->hbmp;
				sfold->SetROI( RoCLS(0,0,hbm->Width(),hbm->Height() ) );
				sfold->ReSize( _FACE_MARK_ );				sfold->vY.resize(0,0);
			}
			return false;
		}
	}
	hBMPP hbm=sfold->hbmp;
	int i,r,c,wth=hbm->Width(),hei=hbm->Height(),left,rigt,botom,top,ext;
	double sW,sH,s,normal=128.0,pts[_FACE_MARK_0_*2];
	float x,y;
	char sLine[1000];
	FILE *fp=fopen( sMark.c_str(),"rt");
	if( fp!=NULL ){
		while( fgets( sLine,1000,fp )){
			if( sscanf( sLine,"%f %f",&x,&y)==2 ){
				x=x*hbm->init.sW;						y=hei-y*hbm->init.sH;
				c0=MIN(c0,x);	c1=MAX(c1,x);			r0=MIN(r0,y);	r1=MAX(r1,y);
				pts[2*nPt]=x;	pts[2*nPt+1]=y;			nPt++;
			}
		}
		assert( nPt==_FACE_MARK_0_ );
		fclose( fp );
	}
//	sfold->SetShapeMarks( pts,_FACE_MARK_0_,nullptr );		
	//hObj->ToBmp( "F:\\GiFace\\train\\",0x0 );
	RoCLS roi;
	BIT_8 *gray=new BIT_8[wth*hei];
	hbm->Channel_( GST_BMP::GRAY,gray );	
	if( !FaceDetect_( hbm,gray,wth,hei,roi ) )	{	
		goto EXIT;	
	}
	if( c1<=roi.c_0 || c0>=roi.c_0+roi.wth || r1<roi.r_0 || r0>=roi.r_0+roi.hei )
	{	printf( "\n%s is MISMATCH!!!\n",sPicPath.c_str() );		goto EXIT;	}
	if( isNormalFace )
		NormalizeFace( sfold,roi,pts );
	else
		sfold->SetShapeMarks( pts,_FACE_MARK_0_,&roi );	
	sMark+="_n";
	sfold->Serial( sMark,false );	/**/
	if( 0 ){		//用以校验标记的数据是否准确		2968784797_1.jpg就不准
		SHAPE_IMAGE si(_FACE_NW_,_FACE_NH_,3);		si.bpp=24;
		sfold->vS=sfold->vY;
		sfold->TraceBmp( "F:\\GiFace\\trace\\6.bmp",si,0x0 );
	}
	bRet=true;
EXIT:
	delete[] gray;
	delete sfold->hbmp;		sfold->hbmp=nullptr;		//为了节省内存
	return bRet;
	
}

void GetMeanShape( vector<ShapeBMPfold*> &Trains,ShapeBMPfold &meanShape,vector<ShapeBMPfold*>&means,int flag=0x0 ){
	int nMark=_FACE_MARK_0_,ldMk=2,i,nSample=Trains.size( );
	char sPath[_MAX_PATH];
	if( !isDetecFace ){
		//char *sPath="F:\\GiFace\\idea_face_2409.dat";
		char *sPath="F:\\GiFace\\idea_face_2708.dat";
		printf( "\n\n********* Load mean shape from \"%s\"\n",sPath );
		meanShape.Serial( sPath );
		if( _FACE_MARK_!=_FACE_MARK_0_ )
			meanShape.ReSize( _FACE_MARK_ );
	}else{
		int i,j,ld=nMark*ldMk,wth;
		double s=1.0;
		meanShape.vY.setZero( );
		for each ( ShapeBMPfold* hShape in Trains ){
			meanShape.vY +=hShape->vY*s;
		}
		meanShape.vY/=nSample;	
		sprintf( sPath,"F:\\GiFace\\idea_face_%d.dat",nSample );
		meanShape.Serial( sPath,false );
	}

	ShapeBMPfold::PTFs vMarks;
	for( i=0;i<meanShape.vY.rows();i++ )		vMarks.push_back( ShapeBMPfold::PTonF(0,0,i,ShapeBMPfold::FOLD_MAP) );
	if( 1 ){	//save to bmp
		SHAPE_IMAGE si(_FACE_NW_,_FACE_NH_,3);		si.bpp=24;
		GST_BMP bmp( &si );

		meanShape.SetROI( RoCLS( 0,0,_FACE_NW_,_FACE_NH_ ) );
		meanShape.PtsOnFold( vMarks,meanShape.vY );		
		for each ( ShapeBMPfold::PTonF pt in  vMarks ){
			bmp.SetPixelColor( pt.gc,pt.gr,255,0,0 );
		};
		bmp.Save( sDumpFolder+"mean.bmp" );
	}
	/*double delta[9][2]={//失败的尝试			1/4/2016
			{-off,-off},	{-off,0},	{-off,off},
			{-0,-off},		{-0,0},		{-0,off},
			{off,-off},		{off,0},	{off,off},
	};
	for( i=0;i<9;i++ ){		
		ShapeBMPfold *hX=new ShapeBMPfold(meanShape);	
		Eigen::RowVector2d vd={delta[i][0],delta[i][1] };
		hX->vY.rowwise( )+=vd;
		means.push_back(hX);
	}*/
}
string TITLE(const string &sPath){
	int len=sPath.size(),last=sPath.rfind("\\");
	string title=sPath.substr(last+1,len);
	return title;
}
bool LoadSamplesAt( wchar_t *sFolder,vector<ShapeBMPfold*>&Samples,int nMost=100,int flag=0x0 ){
	Samples.clear( );
	g_arrPath.clear( );
 	GetPaths( sFolder,true,0x0 );		
	vector<string> arrPath;
	vector<wstring> arrWPath;
	int nImage,nMark=_FACE_MARK_0_,ldMk=2,i=0,samp,wMax=480,hMax=640,nFail=0;
	SHAPE_PtSet spsv(nMark,ldMk);
	//if( isNormalFace )	{	wMax*=2,		hMax*=2;	}	
	//wMax=240,		hMax=320;		//太小了
	char sPath[_MAX_PATH];
	for each ( wstring wPath in g_arrPath ){
		if( arrPath.size()>=nMost )		break;	
		W2CHAR( wPath.c_str(),sPath );
		if( isTestMorph ){
			//arrWPath.push_back( wPath );		continue;
			arrPath.push_back( sPath );		continue;
		}else{
			//wPath = L"G:\\face\\ibug\\testset\\3045005984_1.jpg";		
			//wPath = L"G:\\face\\ibug\\trainset\\2365877276_1.jpg";		
		}
		string sMark(sPath);
		sMark=sMark.substr(0,sMark.size()-3)+"pts_n";
		if(_access(sMark.c_str(),0)!=0 )			{	
			if( !isDetecFace ){
				if( isTestLigner ){		//测试图片，无需.pts_n
					arrPath.push_back( sPath );
				}else{
					if( (nFail++)%100==0)
						printf( "\r%d: %s \tFAILED\n",nFail,sMark.c_str() );	
				}
				continue;	
			}
		}else{
			if( isDetecFace )
			{	::DeleteFileA( sMark.c_str() );		}
		}
		arrPath.push_back( sPath );
	}
	if( isTestMorph )	{
		//Morph_Pick1000( arrWPath,0x0 );
		Morph_Test( arrPath,0x0 );
		exit(-100);
	}

	nImage=arrPath.size();
	ShapeBMPfold spMean(nullptr,spsv),*hX=nullptr;

	int nThread=isDetecFace?1:8;
	_tprintf( L"********* Samples pics=%d  @\"%s\"...detect=%d,normal=%d\r\n",nImage,sFolder,isDetecFace,isNormalFace );
	//nThread=1;
#pragma omp parallel for num_threads(nThread) private( i ) 
	for ( i=0; i<nImage; i++ )	{
		string sBmp=arrPath[i];
		if( isNormalFace )
			sBmp="G:\\face\\ibug\\normals\\"+TITLE(arrPath[i]);
		hBMPP hbm=new BMPP( sBmp.c_str(),GST_BMP::INITIAL(wMax,hMax) );		
		if( hbm==nullptr )	continue;		
		hbm->Median( );		
		//hbm->Normalise( GST_BMP::CONTRAST,0 );		
		ShapeBMPfold *spY=new ShapeBMPfold(hbm,spsv );		spY->nam=TITLE(arrPath[i]);
		GST_VERIFY( spY!=nullptr,"spY is 0" );
		if( ReadLandMarks( arrPath[i].c_str(),spY,isDetecFace,sObjTrainDir ) || isTestLigner ){		//测试图片，无需.pts_n
#pragma omp critical
			{	Samples.push_back( spY );	
				if( spY->vY.size()==0 )		printf( "\n%s: vY is NULL!!!\n",arrPath[i].c_str() );
				if( i%50==0)	printf( "\r%s......",arrPath[i].c_str() );
			}		
		}else{
			delete spY;
			printf( "\r%d<%s>...\tFAILED\n",i,arrPath[i].c_str() );
		}
	}
	std::sort( Samples.begin(),Samples.end( ),ShapeBMPfold::isBig );

	int nSample=Samples.size( );
	if( nSample>0 ){
		Eigen::VectorXd unit(nSample),dBox(nSample);
		for( i=0;i<nSample;i++ ){
			unit(i)=Samples[i]->lenUnit;		assert( unit(i)>0 || Samples[i]->vY.size()==0 );
			dBox(i)=Samples[i]->roi.wth;
		}
		double lenUnit=unit.mean( ),off=0.2*lenUnit;
		printf( "\n\t _FACE_MARK_=%d,lenUnit<%d>=(%g,%g,%g) \n",_FACE_MARK_,nSample,unit.minCoeff(),unit.maxCoeff(),lenUnit );
		printf( "\t BOX<%d>=(%g,%g,%g) \n",nSample,dBox.minCoeff(),dBox.maxCoeff(),dBox.mean() );
	}
	return true;
}

static double tAlign=0,errAlign=0;

//参见pertub_1_14_2016.dat
static float pertub[]={ 0.1,0.05,0.04,0.03,0.02,0.01,0.005 };
void Pertube( vector<ShapeBMPfold*> &Trains,ShapeBMPfold *hSamp,int no,int flag=0x0 ){
	Eigen::MatrixXf delta(_FACE_MARK_,2);
	float eta,a,*off=delta.data(),nrmY=hSamp->vY.norm(),angle,*vS,e1;
	int nP=sizeof(pertub)/sizeof(float),i,j,ldS=hSamp->vY.size( );
	std::uniform_real_distribution<float> uniFloat(-PI,PI);
	std::mt19937 rander(no+1000);
	Eigen::Matrix2f mP;
	for(i=0; i<nP;i++ ){
		ShapeBMPfold *hX=new ShapeBMPfold(*hSamp);		
		hX->vS = hX->vY;		vS = hX->vS.data( );		
		for( j=0;j<_FACE_MARK_;j++ ){	
			angle=uniFloat(rander);		
			vS[j]+=pertub[i]*cos(angle); 
			vS[j+_FACE_MARK_]+=pertub[i]*sin(angle); 
		};
		e1=(hX->vS-hX->vY).norm()/hX->vY.norm( );
	/*	for( j=0;j<_FACE_MARK_*2;j++ ){	off[j]=uniFloat(rander); }
		hX->vS = hX->vY+delta;*/
		hX->nam=hSamp->nam+"_"+to_string(e1);
		Trains.push_back( hX );
	}
}

void TestLigner( vector<ShapeBMPfold*>&Samples,ShapeBMPfold::VECT& sp0,int flag=0x0 ){
	FaceAligner falign;
	int *trees=nullptr,no=0;	
	T_MOVE *moves=nullptr;
	size_t nzM=0,nzT=0;
	File2Array( falign.param.pRF_D.c_str(),&nzM,&moves );
	File2Array( falign.param.pRF_T.c_str(),&nzT,&trees );
	if( nzM%(_FACE_MARK_*2)!=0 ){
		GST_THROW( "FaceAligner::pRF_D is X" );
	}
	falign.SetData( nzM,moves,nzT,trees );
	delete[] moves;		delete[] trees;		

	PICS_N pics;
	pics.nails.clear( );
	bool isOK=false;
	double e0,e1;
	for each( ShapeBMPfold *hSamp in Samples )	{
		hSamp->hbmp->no=no++;
		hSamp->vS=sp0;		hSamp->type=ShapeBMPfold::SAMP_TEST;
		if( 1 )	{
			bool isCheckErr=!hSamp->vY.isZero( );
			if( isCheckErr )	{	hSamp->UpdateErr( );		e0 = hSamp->NormalErr( );	}
			
			isOK = falign.Solve(hSamp->hbmp,hSamp->vS.data())==FaceAligner::OK;
			falign.Morph( hSamp->hbmp,hSamp->vS.data() );
			if( isOK ){
				if( isCheckErr )	{	
					hSamp->UpdateErr( );		e1 = hSamp->NormalErr( );		errAlign+=e1;
				}
				hSamp->SetROI( falign.roi );
			}
		}	else	{
			//isOK= FaceAlign( hSamp )==FALIGN_OK;
		}
		if( !isOK ){
			delete hSamp->hbmp;		hSamp->hbmp=nullptr;		
		}
	}

	no=0;
	std::sort( Samples.begin(),Samples.end( ),ShapeBMPfold::isBigErr );
	for each( ShapeBMPfold *hSamp in Samples )	{
		if( hSamp->hbmp!=nullptr) {
			printf("\t%d:%s - %g\n",no++,hSamp->nam.c_str(),hSamp->err );
			hSamp->arrS.clear( );
			pics.nails.push_back( hSamp->TraceBmp("",siFace,0x0) );
		}
	}
	pics.PlotNails( "F:\\GiFace\\trace\\align_test.bmp",siFace,20,true );
}

//没啥效果	1/17/2016
int AdaptiveTrains( int cas,vector<ShapeBMPfold*> &Trains,int nOOB,float T_err,int flag=0x0 ){
	PICS_N pics;
	pics.nails.clear( );
	int nOK=0,nSample=Trains.size( );
	if( nSample<nOOB*3 )
		return 0;
	vector<ShapeBMPfold*>::iterator it=Trains.begin(),ite=Trains.end()-nOOB;
	while( it!=ite ){
		ShapeBMPfold* hFold=*it;
		if( hFold->err<T_err ){
			it=Trains.erase( it );		ite=Trains.end()-nOOB;
			pics.nails.push_back( hFold->TraceBmp("",siFace,0x0) );
			delete hFold;			nOK++;
		}
		else
			it++;
	}
	if( nOK>0 ){
		sprintf( sPath,"F:\\GiFace\\trace\\Adaptive_%d.bmp",cas );
		pics.PlotNails( sPath,siFace,20,true );
		printf( "********* ADAPTIVE[%d]: %d is OK. %d samples now.\n\n",cas,nOK,Trains.size() );
	}
	return 0x0;
}

#include "F:\GiFace\GruFacer\GruFacer.h"

void GruFacer_Test( )	{
	int iRet=GFACER_init( 1, _T("G:\\KANKAN\\GruFacer_Test\\Model\\"),0x0 );
	if( iRet!=GFACE_OK )
		return;
	//GFACER_dir( _T("I:\\algin\\InfraRed_1\\"),100,0x0 );
	int nImage=1,i,width,height,mode=0x0,r;
	float *feat=new float[nImage*GFACE_FEAT_LEN],*info=new float[nImage*GFACE_INFO_LEN];
	GRAY_PIC *gray=nullptr;
//单张图片测试
	for( i=0;i<nImage;i++ ){
		//GST_BMP bmp( _T("I:\\看看数据\\Face\\InsidePeople\\chuyanghao_police\\0.jpg") );
		//gray=bmp.Gray(),width=bmp.Width(),height=bmp.Height( );
		dlib::array2d<unsigned char> img;
		dlib::load_bmp(img, "G:\\KANKAN\\GruFacer_Test\\Model\\0.bmp");
		width=img.nc(),height=img.nr();
		unsigned char *gray=new unsigned char[width*height];
		unsigned char *pixel=&(img[0][0]);
		for( r=0;r<height;r++){	//flip vertical
			memcpy( gray+(height-1-r)*width,pixel+r*width,sizeof(unsigned char)*width );	
		}

		GFACER_feat_1( gray,width,height,mode,feat+i*GFACE_FEAT_LEN,info+i*GFACE_INFO_LEN,0x0 );
		delete[] gray;
		//gray由调用程序自行释放
	}
#ifdef _DEBUG
	for(i=0;i<GFACE_FEAT_LEN;i++ )
		printf( "%g ",feat[i] );
#endif
	delete[] feat;		delete[] info;
	GFACER_clear( );
	return;

}
int FaceNormal_test( wchar_t *wPathFrom,wchar_t *wPathTo,float sZoom,int flag );
int KG_facial_test( int argc, _TCHAR* argv[] );
int VGG_test( int argc, _TCHAR* argv[] );
int TensorNet_Test( int argc, _TCHAR* argv[],int flag );
//_T("G:\\face\\lfw\\Silvio_Berlusconi\\")
int _tmain( int argc, _TCHAR* argv[] ){
	_tsetlocale( LC_ALL,TEXT("chs") );
#ifdef _DEBUG 
	//MasDist_Test( "F:\\GiFace\\DistMetric\\tensor_d_5871_128.feat",15000,128,0x0 );		return 0x0;
#endif
	if( 0 ){
		GruFacer_Test( );	return 0;
	}else
		return VGG_test( argc,argv );	
//	FacePos_test( argc,argv );		return 0;
	if( argc>=2 ){
		if( _tcscmp(argv[1],L"align" )==0 )	{
			isTestLigner=true;		isDetecFace=false;
			printf( "\n*********  ONLY Testing Face Aligner,NO TRAINING!!!  DetecFace is FALSE *********\r\n" );
		}else if( _tcscmp(argv[1],L"morph" )==0 )	{
			printf( "\n*********  MORPH test *********\r\n" );
			isTestMorph=true;
		}else{
			int parai;
			if( _stscanf(argv[1],_T("%d"),&parai )==1 ){
				nMaxTrain=parai;
			}
		}
	}
	printf("\n********* MaxTrain=%d from %s MaxTest=%d *********\n",nMaxTrain,sTrainSetDir,nMaxTest );	
	siFace.bpp=24;
	double err;
	vector<ShapeBMPfold*> Samples,means;
	ShapeBMPfold spMean(nullptr,SHAPE_PtSet(_FACE_MARK_0_,2)),*hX=nullptr;
//	if( !isTestLigner ) LoadSamplesAt( _T("G:\\face\\ibug\\trainset\\"),Samples,nMaxTrain );
	if( !isTestLigner ) LoadSamplesAt( sTrainSetDir,Samples,nMaxTrain );
/*
	1 简单增大dup，无效。
	2 nStep的合适值 (20优于100或10，why? )
*/
	int i,ca,nCascade=50,nTree=100,dup=20,nCand=400,cur,next,nSample=Samples.size(),ldPic,nOOB=nMaxTest,samp;
	int nStep=10;		//MULTI_TREE
//	int nStep=100;		//SINGLE TREE
	vector<ShapeBMPfold*> Trains;
	GetMeanShape( Samples,spMean,means );
	PICS_N pics;
	SHAPE_IMAGE si(_FACE_NW_,_FACE_NH_,3);		si.bpp=24;
	if( !isTestLigner && nSample>1 )		{
		printf( "\n*********  Augment-%d *********\r\n",dup );
		std::mt19937 mt(314159);
		std::uniform_int_distribution<int> dist(0,nSample-1);
		SHAPE_IMAGE si(96,96,3);		si.bpp=24;
		for ( samp=0;samp<nSample;samp++ ) {
			ShapeBMPfold *hSamp=Samples[samp];
			string name=hSamp->nam;
			for( i=0; i<dup; i++ ){
				hX=new ShapeBMPfold(*hSamp);	
				if( i<means.size() ){
					//hX->vS=spMean.vY;
					hX->vS=means[i]->vY;
				}else{
					while( (next=dist(mt))==samp );
					hX->vS = Samples[next]->vY;			
				}
				hX->nam=name+"_"+to_string(i);
				Trains.push_back( hX );
			}
			if(0)	Pertube( Trains,hSamp,samp,0x0 );
		}
		if( 0 ){	//all_train.bmp
			for each( ShapeBMPfold *hSamp in Trains ){
				pics.nails.push_back( hSamp->TraceBmp("",si,0x0) );
			}
			pics.PlotNails( "F:\\GiFace\\trace\\all_train.bmp",si,20,true );
		}
	}
NEXT:
	//LoadSamplesAt( _T("F:\\GiFace\\hard\\1\\"),Samples,nOOB );	//test set;
	LoadSamplesAt( _T("G:\\face\\ibug\\testset\\"),Samples,nOOB );	//test set;
	//double err=0.0;
	nOOB=Samples.size( );
	if( nOOB>0 ){	//输出all_test.bmp
		pics.nails.clear( );
		for each( ShapeBMPfold *hSamp in Samples )	{
			hSamp->vS=hSamp->vY;		
			/*if( isTestLigner ) {
				hSamp->vS=spMean.vY;		hSamp->type=ShapeBMPfold::SAMP_TEST;
				FaceAlign( hSamp );		
			}*/
			if( hSamp->hbmp!=nullptr) pics.nails.push_back( hSamp->TraceBmp("",si,0x0) );
		}
		pics.PlotNails( "F:\\GiFace\\trace\\all_test.bmp",si,20,true );
	}
	if( isTestLigner ){
		TestLigner( Samples,spMean.vY );
	}else{
		for( i=0;i<nOOB;i++ ){		//all test from spMean
			hX=Samples[i];		hX->vS=spMean.vY;
			hX->type=ShapeBMPfold::SAMP_TEST;
			Trains.push_back( hX );
			//hX->UpdateErr( );		err = hX->NormalErr( );
		}
	}
	if( isDetecFace || isTestLigner  )	{	
		printf("\n********* ALIGN ERR=<%d,%g> time=%g*********\n",nOOB,errAlign/nOOB,tAlign/nOOB );	
		for each( ShapeBMPfold *hSamp in Trains )
			delete hSamp;
		Trains.clear( );
		return 1;	
	}	
	
	char *sRFCpp="F:\\GiFace\\ligner\\rf.cpp",*sRFD="F:\\GiFace\\ligner\\rf_D.dat",*sRFT="F:\\GiFace\\ligner\\rf_T.dat";
	//i = ::DeleteFileA( sRFCpp );		i = ::DeleteFileA( sRFDpp );
	
	for( ca=0; ca<nCascade; ca++ ){		
		RF_ShapeRegress rf( Trains,spMean,nCand,nStep,nTree/nStep,nOOB,ca );
		rf.InitCPP( sRFCpp,sRFD,sRFT,ca );
		rf.Train( "",ca,0x0 );
		if( rf.rErr()==0.0 )
			break;
		if( ca==nCascade-1 )		rf.CoreInCPP( nCascade,0x0 );
		//AdaptiveTrains( ca,Trains,nOOB,0.03 );
	}
	Trains.clear( );

	return 0;
}


