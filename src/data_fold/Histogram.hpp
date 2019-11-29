#pragma once
#include <vector>
#include <map>  
#include <algorithm>
#include <cmath>
#include <numeric>  
#include "../util/samp_set.hpp"
#include "../include/LiteBOM_config.h"

using namespace std;
typedef map<int, int> MAP_CATEGORY;


/*
	https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
	比较复杂，暂时保留
*/
template <typename T>
void sort_indexes_NA(int nSamp, const T*v, vector<size_t>& idx, int flag = 0x0) {
	size_t i, nz0=0, nGood=0,*map=new size_t[nSamp];
	T *vGood =new T[nSamp];
	for (i = 0; i < nSamp; i++) {
		if (IS_NAN_INF(v[i])) {
			continue;
		}	else {
			vGood[nGood] = v[i];		map[nGood]= i;		nGood++;
		}
	}
	vector<size_t> idx_1;
	idx_1.resize(nGood);// initialize original index locations
	iota(idx_1.begin(), idx_1.end(), 0);
	// sort indexes based on comparing values in v
	std::sort(idx_1.begin(), idx_1.end(), [&vGood](size_t i1, size_t i2) {return vGood[i1] < vGood[i2]; });
	for (i = 0; i < nGood-1; i++) {
		assert(vGood[idx_1[i]] <= vGood[idx_1[i + 1]]);
	}
	idx.resize(nSamp);
	for (i = 0; i < nGood; i++) {
		idx[i] = map[idx_1[i]];
	}
	for (i = 0; i < nSamp; i++) {
		if (IS_NAN_INF(v[i])) {
			idx[nGood++]=i;
		}		
	}
	assert(nGood==nSamp);
	delete[] vGood;			delete[] map;
	return;
}

template <typename T>
void sort_indexes(int nSamp,const T*v, vector<tpSAMP_ID>& idx, int flag = 0x0) {
	if (false) {
		//sort_indexes_NA(nSamp,v,idx,flag);
		return;
	}
		
	size_t i;
	idx.resize(nSamp);// initialize original index locations
	iota(idx.begin(), idx.end(), 0);
	// sort indexes based on comparing values in v
	std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
	for (i = 0; i < nSamp - 1; i++) {
		assert(v[idx[i]] <= v[idx[i + 1]]);
	}
	return;
}

template <typename T>
void sort_indexes(const vector<T> &v, vector<tpSAMP_ID>& idx, int flag = 0x0) {
	T *arr = (T*)(v.data());
	sort_indexes(v.size(),arr, idx,  flag);	
	return;
}



namespace Grusoft {
	class FeatsOnFold;
	class FeatVector;
	//typedef short tpQUANTI;
	//typedef int tpQUANTI;
	//用于快速split,histogram其实是最简单的GreedyCluster
	class GreedyCluster {

	};

	typedef enum {
		BY_VALUE = 0x0, BY_DENSITY
	}SPLIT_HISTOGRAM;

	class HistoGRAM;

	//feature vector的数据类型多变，所以映射很复杂
	struct HISTO_BIN {
		//double split_F = 0;		
		size_t nz = 0;
		//if (val[pos] < bins[noBin+1].split_F) {		quanti[pos] = noBin;		}
		tpFOLD fold = -1;			//每个BIN属于一个FOLD
		uint16_t tic = 0;			
		double G_sum = 0, H_sum = 0;		//Second-order approximation from Tianqi Chen's formula

		/*static bool isSplitSmall(const HISTO_BIN &l, const HISTO_BIN &r)		{
			return l.split_F<r.split_F;
		}*/

		
	};

	//HISTO_BIN仅记录基本信息 feat analysis is here...
	struct BIN_FEATA {
		double split_F = 0;
		char fold = -1;			//每个BIN属于一个FOLD
		int tic = 0;			//X		可删除
		double density, bandwidth;		//The bandwidth of the kernel is a free parameter which exhibits a strong influence on the resulting estimate
		
		bool operator==(const BIN_FEATA& right) const {
			if (split_F != right.split_F)
				return false;
			if (fold != right.fold)
				return false;			
			if (tic != right.tic)
				return false;			
			return  true;
		}
	};

	class FRUIT{
	public:
		//FRUIT(HistoGRAM *his_ = nullptr,int flag=0x0) ;
		FRUIT(FeatsOnFold *hFold, const HistoGRAM *his_, int flag = 0x0);
		int best_feat_id=-1;
		const HistoGRAM *histo_refer = nullptr;		//仅指向，不再删除
		HISTO_BIN bin_S0, bin_S1;						//有变化，比较危险诶
		double split_0,adaptive_thrsh, split_1;					//可指向binSplit中间，所以保留		splitonY也需要这个值
		SPLIT_HISTOGRAM split_by = BY_VALUE;
		double Thrshold(bool isQuanti) {
			//assert(binSplit.split_Q >= 0);
			if (isQuanti) {
				return bin_S0.tic+1;
				//return bin_S1.tic;
			}	else {
				if(true)
					return adaptive_thrsh;
				else 	
					return adaptive_thrsh;	//split_1-EPSILON
					//return bin_S1.split_F;
			}
		}
		//MAP_CATEGORY mapFold;
		tpFOLD *mapFolds=nullptr;
		inline tpFOLD GetFold(int pos) {
			return mapFolds[pos];
			/*if (mapFold.find(pos) == mapFold.end()) {
				return -1;
			}	else {
				int fold = mapFold[pos];
				return fold;
			}*/
		}

		//需要重新设计，参见SplitOn以及AtLeaf函数
		bool isY = false;
		bool isNanaLeft = false;
		//size_t pos_0 = 0, pos_1 = 0;
		size_t nLeft = 0, nRight = 0;
		double tic_left = 0;
		double mxmxN = -1;		//mean*mean*N
		std::string sX;		//用于调试信息

		//char buf[256] = { 0 };		//https://stackoverflow.com/questions/4523497/typedef-fixed-length-array	Arrays can't be passed as function parameters by value in C.
		//double lft_impuri,rgt_impuri;		//为了调试
		virtual ~FRUIT();
		
		virtual void Set(FeatsOnFold *hFold,const HistoGRAM*histo,int flag=0x0);
	};

	class HistoGRAM_BUFFER;
	/*
		与特定FeatVector对应的HistoGRAM
	*/
	class HistoGRAM {
	protected:
		//FeatVector*	hFeat = nullptr;
		virtual void UpdateBestGain(int tic, double g1, size_t nLef, size_t nRight, int flag = 0x0);
	public:
		static size_t nAlloc;
		static double  memAlloc;
		SPLIT_HISTOGRAM split_by = BY_VALUE;
		int nBigBins = 0,nMostBins=0,nBins=0;
		//bool isFilled = false;
		size_t nSamp, nLeft = 0, nRight=0;
		typedef struct {	//为了并行诶
			double mxmxN = -1;		//mean*mean*N
			size_t nLeft = 0, nRight = 0;
			int tic=-1;
			bool isY = false;
			double adaptive_thrsh = 0;

			void Clear() {
				mxmxN = -1;		nLeft = 0,	nRight = 0;		tic = -1;
				isY = false;	adaptive_thrsh = 0;
			}
		}FRUIT_INFO;
		FRUIT_INFO	fruit_info;
		FeatVector *hFeat = nullptr;	//仅仅指向
		//vector<HISTO_BIN> bins;
		HistoGRAM_BUFFER *buffer = nullptr;
		HISTO_BIN *bins = nullptr;
		//NA value---样本的某featrue确实missing value ,但总体上还是有down direction
		HISTO_BIN* hBinNA( )	{
			assert(nBins>0);	return bins + nBins - 1;
		}		//总是放在最后
		//{	return &(bins[bins.size()-1]);	 }		//总是放在最后

		HistoGRAM(FeatVector*hFeat_,size_t nMost, int flag = 0x0) : hFeat(hFeat_){
			nAlloc++;
			nSamp = nMost;
			//a1 = -DBL_MAX, a0 = DBL_MAX;	
		}
		virtual ~HistoGRAM();		

		virtual void CheckValid(const LiteBOM_Config&config, vector<BIN_FEATA>*featas=nullptr,  int flag = 0x0);
		virtual void CompressBins(int flag=0x0);
		virtual void Dump(const vector<BIN_FEATA>&binFeatas, MAP_CATEGORY&mapCategory, int flag = 0x0);
		//virtual void TicMap(tpQUANTI*map,int flag);

		//virtual double split_F(int no, int flag = 0x0) const;

		HistoGRAM* FromDiff(const HistoGRAM*hP, const HistoGRAM*hB,bool isBuffer, int flag = 0x0);

		template<typename Tx,typename Tf>
		bool At(Tx x, Tf&f,int flag = 0x0) const {
			double rigt=DBL_MAX;
			bool isFind=false;
			size_t i;
			for (i=0;i<nBins;i++) {
				rigt = i<nBins-1 ? bins[i+1].tic : DBL_MAX;
				const HISTO_BIN& bin= bins[i];
				if(bin.nz>0 && bin.tic<=x && x<rigt)				{	
					f = bin.G_sum/ bin.nz;
					isFind = true;		break;
				}
				
			}
			return isFind;
		}

		template<typename Tx>
		int AtBin_(Tx x, int flag = 0x0) const {
			double rigt = DBL_MAX;
			//bool isFind = false;
			int pos=-1;
			size_t i;
			for (i = 0; i<nBins; i++) {
				rigt = i<nBins - 1 ? bins[i + 1].tic : DBL_MAX;
				const HISTO_BIN& bin = bins[i];
				if (bin.nz>0 && bin.tic <= x && x<rigt) {
				//if (bin.tic <= x && x<rigt) {
					return i;
				}

			}
			return pos;
		}

		//Optimal Data-Based Binning for Histograms
		virtual void OptimalBins(size_t nMost, size_t nSamp, double a0_, double a1_, int flag = 0x0);

		virtual void ReSet(size_t nMost, int flag=0x0);

		virtual void RandomCompress(FeatVector *hFV,bool, int flag = 0x0);
		virtual void MoreHisto(const FeatsOnFold *hData_, vector<HistoGRAM*>&more, int flag = 0x0);
		virtual void GreedySplit_X(FeatsOnFold *hData_, const SAMP_SET& samp_set, int flag = 0x0);
		//virtual void GreedySplit_X(const FeatsOnFold *hData_, const SAMP_SET& samp_set, int flag = 0x0);
		virtual void GreedySplit_Y(FeatsOnFold *hData_, const SAMP_SET& samp_set,bool tryX, int flag = 0x0);
		virtual void Regress(const FeatsOnFold *hData_, const SAMP_SET& samp_set,  int flag = 0x0);

		virtual void CopyBins(const HistoGRAM &src, bool isReset, int flag);
	};
	//typedef map<int, HistoGRAM*> MAP_HistoGRAM;
	class HistoGRAM_BUFFER {
		const FeatsOnFold *hData_=nullptr;
		typedef map<int, int> MAP_FEAT;
		MAP_FEAT mapFeats;
		HISTO_BIN *bins_buffer = nullptr;
		vector<HistoGRAM*> buffers;
		int NodeFeat2NO(int node, int feat)	const;
		int ldFeat_=0, nMostNode=0,nMostFeat=0,nzMost=0;
		size_t nzMEM, nMostBin=0;
		virtual size_t SetBinsAtBuffer(const FeatsOnFold *hData_, vector<int>& pick_feats, int flag);
	public:
		HistoGRAM_BUFFER(const FeatsOnFold *hData_, int flag=0x0);
		virtual ~HistoGRAM_BUFFER( );
		HistoGRAM*Get(int node,int feat, int flag = 0x0)	const;

		//virtual void Set(int feat, HistoGRAM*histo);

		virtual void BeforeTrainTree( vector<int>& pick_feats,size_t nPickSamp,int flag);

		virtual void Clear(int flag = 0x0);
	};

	/*
		from Tianqi Chen's formula
	
	class Histo_CTQ : public HistoGRAM {
	protected:
		//double gSum=0,hSum=0;
	public:
		Histo_CTQ(size_t nMost, int flag = 0x0) : HistoGRAM(nMost,flag){
		}

		virtual void GreedySplit_X(const FeatsOnFold *hData_, const SAMP_SET& samp_set, int flag = 0x0);
		//virtual void GreedySplit_Y(const FeatsOnFold *hData_, const SAMP_SET& samp_set, bool tryX = true, int flag = 0x0) { ; }
		virtual void Regress(const FeatsOnFold *hData_, const SAMP_SET& samp_set, int flag = 0x0) { ; }
	};*/
}