#include <omp.h>
#include <io.h>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "FeatData.hpp"
using namespace Grusoft;
using json = nlohmann::json;

std::string DataSample::sDumpFolder;
std::string const DataSample::sBackSrc="background:";
std::string DataSample::sLastPicPath;


int DataSample::Save(const std::wstring sPath, int flag)	{
	int ret = -1, dim = nSample();
	FILE *fp = _tfopen(sPath.c_str(), _T("wb"));
	if (fp == NULL)
		goto _FIO_EXIT_;
	ret = save_fp(fp, flag);
_FIO_EXIT_:
	if (fp != NULL && fclose(fp) != 0)	{		ret = -7;	}
	if (ret == 0)	{
	}
	else	{
		ret = fp != NULL ? ferror(fp) : ret;
	}
	return ret;
}
int DataSample::Save(const std::string sPath, int flag)	{
	int ret = -1, dim = nSample();
	FILE *fp = fopen(sPath.c_str(), ("wb"));
	if (fp == NULL)
		goto _FIO_EXIT_;
	ret = save_fp(fp, flag);
_FIO_EXIT_:
	if (fp != NULL && fclose(fp) != 0)	{		ret = -7;	}
	if (ret == 0)	{
	} else	{
		ret = fp != NULL ? ferror(fp) : ret;
	}
	return ret;
}

int DataSample::Load(const std::wstring sPath, int flag)	{
	try{
		int ret = -1;
		FILE *fp = _tfopen(sPath.c_str(), _T("rb"));
		if (fp == NULL)
			goto _FIO_EXIT_;
		ret = load_fp(fp, flag);
	_FIO_EXIT_:
		if (fp != NULL && fclose(fp) != 0)
		{
			ret = -7;
		}
		if (ret == 0)	{
		} else	{
			ret = fp != NULL ? ferror(fp) : ret;
			//		G_PRINTF( ("\t!!!Failed to save %s. err=%d"),sLibPath,ret );
		}
		return ret;
	}
	catch (...)	{
		throw exception("DataSample::Load !!!");
	}
}
int DataSample::Load(const std::string sPath, int flag)	{
	try{
		int ret = -1;
		FILE *fp = fopen(sPath.c_str(), ("rb"));
		if (fp == NULL)
			goto _FIO_EXIT_;
		ret = load_fp(fp, flag);
	_FIO_EXIT_:
		if (fp != NULL && fclose(fp) != 0)
		{
			ret = -7;
		}
		if (ret == 0)	{
		} else	{
			ret = fp != NULL ? ferror(fp) : ret;
			//		G_PRINTF( ("\t!!!Failed to save %s. err=%d"),sLibPath,ret );
		}
		return ret;
	}
	catch (...)	{
		throw exception("DataSample::Load !!!");
	}
}

int DataSample::SelectTo(hDATASET hTo, vector<int>& nEach,bool isMove, int flag){
	assert( hTo!=nullptr );
	int cls,nCls=nEach.size( ),i,nz,iT=0, iS=0,no;
	int nData=nSample(),nz_0=nData,nz_1=0,nLeft=nData,nTo;
	vector<int> list;
	for( cls=0; cls<nCls; cls++ ){
		nz=0;		list.clear( );
		for( i=0; i<nData; i++ )		{
			if(tag[i]==cls)	{	
				list.push_back(i);			nz++;
			}
		}
		//if( test!=nullptr )
		std::random_shuffle( list.begin(),list.end() );
		nEach[cls] = MIN( nEach[cls],nz );		assert(nEach[cls]>=0);
		nz_0=MIN(nz_0,nEach[cls]);					nz_1=MAX(nz_1,nEach[cls]);	
		for( i=0; i<nz; i++ )	{
			no=list[i];			assert( tag[no]==cls );
			if( i<nEach[cls] )	{
				TransTo(no, hTo, iS++);
			}
			else if( isMove ){
				TransTo(no, this, iT++);
			}
		}
	}
	for( i = iS; i < hTo->nMost; i++ )		hTo->tag[i]=-1;		nTo = hTo->nSample();
	if( isMove ){
		for( i = iT; i < nData; i++ ){
			if( tag[i]>=nCls)
					TransTo(i, this, iT++);
		}
		for( i = iT; i < nData; i++ )	tag[i]=-1;
		nLeft = nSample();
		assert(nLeft + nTo <= nData);
		//GST_util::print( "===>Move from %d to %d. Left=%d\teach===", nData, nTo, nLeft );
	}else{
		//GST_util::print( "===>Copy from %d to %d\teach===", nData, nTo );
	}
	//for( i=0; i<nCls; i++ )		GST_util::print( "%d,",nEach[i]);
	//GST_util::print( "===\n" );

	return 0;
}

//注意与copy保持一致
void DataSample::CopyInfo(size_t to, const hDATASET hS, size_t from, int nSample, int flag )	{
	assert(to >= 0 && to < nMost && from >= 0 && from < hS->nSample());		
	memcpy(tag + to, hS->tag + from, sizeof(int)*nSample);
	if( info!=nullptr && hS->info!=nullptr )
		memcpy(info + to, hS->info + from, sizeof(int)*nSample);
	if( outY!=nullptr ){
		memcpy(outY+to*ldY, hS->outY+from*ldY, sizeof(double)*ldY*nSample);
	}
	return;
}

double _Devia_8(BIT_8 *X, int ldX, int flag)	{
	double mean = 0.0, devia = 0.0, a0 = FLT_MAX, a1 = -FLT_MAX, a;
	int i;
	for (i = 0; i < ldX; i++)	{
//		a0 = min(a0, X[i]);		a1 = max(a1, X[i]);
		a0 = a0<X[i] ? a0:X[i];		a1 = a1>X[i] ? a1:X[i];
		mean += X[i];
	}
	mean /= ldX;
	for (i = 0; i < ldX; i++)	{ a = X[i] - mean;		devia += a*a; }
	devia = sqrt(devia / ldX);
	return devia;
}

void DataSample::Dump(string sTitle, int flag)	{
	
}

FeatData* FeatData::read_json(const string &sPath, int flag) {
	std::ifstream ifile("F:/Project/LiteMORT/data/1.json");
	json js;
	ifile >> js;
	int i=0,j, nSamp =js.size(), ldX=8,ldY=1,nCls=2,nEle;
	FeatData *hData = new FeatData(nSamp, ldX, ldY, nCls,flag);

	float label;
	for (json::iterator it = js.begin(); it != js.end(); ++it,++i) {
		nEle = it->size();		assert(nEle == ldX+ldY);
		float *tX = (float *)hData->Sample(i);
		double *Y = hData->Y(i);
		for (j = 0; j < nEle-1; j++)
			tX[j] = it->at(j);
		label = it->at(nEle - 1);		assert(label == 0 || label == 1);
		hData->tag[i] = label;
		Y[0] = label;
		//std::cout << *it << '\n';
	}
	//hData->Reshape();
	return hData;
}
FeatData::FeatData(const string &sPath, int flag) : DataSample_T<float>(),
	hBase(nullptr), feats(nullptr), mark(nullptr), distri(nullptr), qOrder(false) {
	if (flag == 0x100) {
		read_json(sPath, flag);
	}
	//SERIAL ar( sPath,true );
	//Serial( ar,flag );
	int n = nSample();
	mark = new int[n];
	if (nCls>0) {		//仅适用于multiclass
		for (int i = 0; i<nCls; i++)	nEach.push_back(0);
		distri = new float[n*nCls]();
	}
}


