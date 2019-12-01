#pragma once

#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <numeric>
#include <time.h>
using namespace std;
#include "../include/LiteBOM_config.h"
#include "../tree/BiSplit.hpp"
#include "../util/GST_def.h"
#include "../EDA/Feat_Selection.hpp"

#include "EDA.hpp"

#ifdef WIN32
#include <tchar.h>
#include <assert.h>
#else    
#include <assert.h>
//#define assert(cond)
#endif

/*
	&v[0]	since the spec now guarantees vectors store their elements contiguously:
	using c++0x better to use v.data().
*/
template<typename T>
T* VECTOR2ARR(vector<T>& vecs) {
	return (T*)(vecs.data());
}

namespace Grusoft {
	class FeatsOnFold;
	class Distribution;

	class ARR_TREE {
	public:
		typedef tpFOLD * FOLD_MAP;
		int nNodes = 0;
		double *thrsh_step = nullptr, weight = 1.0;
		int *feat_ids = nullptr, *left = nullptr, *rigt = nullptr, *info = nullptr;
		FOLD_MAP *folds = nullptr;

		virtual void Init(int nNode_, int flag = 0x0) {
			nNodes = nNode_;
			thrsh_step = new double[nNodes];
			feat_ids = new int[nNodes * 4];
			left = feat_ids + nNodes;		rigt = left + nNodes;
			info = rigt + nNodes;
			folds = new FOLD_MAP[nNodes];
			for (int i = 0; i < nNodes; i++)
				folds[i] = nullptr;
		}
		~ARR_TREE() {
			if (thrsh_step != nullptr)			delete[] thrsh_step;
			if (feat_ids != nullptr)			delete[] feat_ids;
			if (folds != nullptr) {
				for (int i = 0; i < nNodes; i++) {
					if (folds[i] != nullptr)
						delete[] folds[i];
				}
				delete[] folds;
			}
		}
	};

	class FeatVector {
	protected:
		const FeatsOnFold *hFold_ = nullptr;
		Distribution *distri_ = nullptr;
	public:
		static bool OrderByName(const FeatVector *l, const FeatVector *r) { return l->nam<r->nam; }

		PY_COLUMN *PY;
		struct SELECT {
			double vari_1=0;
			float user_rate=0;
			bool isPick = true;
			bool hasCheckGain = false;
		};
		//FeatVector *fvMergeLeft = nullptr;			//仅指向

		void SetDistri(Distribution*d_, int flag = 0x0);
		Distribution *myDistri(int flag = 0x0) {
			return distri_;
		}	

		/*
		Distribution *histoDistri(int flag = 0x0)	const;	
		//histo空间，总是来自于train_data
		HistoGRAM *tHISTO(int flag = 0x0);*/

		Feature_Selection* select_bins=nullptr;
		double wSplit=0, wSplit_last=0;		//"split", result contains numbers of times the feature is used in a model.
		double wGain=0;			//"gain", result contains total gains of splits which use the feature.
		float *wBins = nullptr;
		enum {		//需要和Distribution合并
			//CATEGORY = 0x100,		DISCRETE = 0x200,
			VAL_REFER = 0x1000,
			//V_ZERO_DEVIA = 0x10000,	//常值，一般可忽略
			IS_BUNDLE = 0x20000,	//in Feature Bundle	参见FeatsOnFold::nPick4Split
			AGGREGATE = 0x80000,
			REPRESENT_ = 0x1000000,
			//RIGTH_MERGE = 0x2000000
		};
		size_t type = 0x0;
		tpSAMP_ID *map4set = nullptr;// , *map4feat = nullptr;
		//bool isSelect = false;
		//float select_factor = 1;
		SELECT select;
		bool isCategory()	const	{ return	BIT_TEST(type, Distribution::CATEGORY); }	
		bool isReferVal()	const	{ return	BIT_TEST(type, VAL_REFER); }		
		virtual bool isMerged()	const { return	false; }
		const FeatsOnFold *AtFold_()		{ return	hFold_; }
		//virtual bool isPass()	const;
		typedef enum {
			COPY_MEAN,
		}BINARY_OPERATE;
		int id = -1;		//非常重要，唯一标识
		int agg_no = -1;		//展开为一系列聚合特征
		
		string nam = "", desc = "";

		FeatVector()	{}
		virtual ~FeatVector();
		virtual inline size_t size()	const	{ throw"FeatVector::size() is ..."; }

		//virtual void GetMergeSampSet(const SAMP_SET&samp_set,int * int flag = 0x0);

		virtual void Empty(int flag = 0x0)		{		}
		virtual void FreeVals(int flag = 0x0)	{		}
		virtual void Set(size_t len, void* p, int flag = 0x0) { throw"FeatVector::Set_void_p is ..."; }
		virtual void Set(size_t len, PY_COLUMN *col, int flag = 0x0) { throw"FeatVector::Set_PY_COLUMN is ..."; }
		virtual void Set(size_t pos, double a, int flag = 0x0) { throw"FeatVector::Set_a is ..."; }
		virtual void CopyFrom(const FeatVector*src, int flag = 0x0) { throw"FeatVector::CopyFrom is ..."; }
		virtual void loc(vector<tpSAMP_ID>&poss, double target, int flag = 0x0) { throw"FeatVector::loc is ..."; }

		virtual void UpdateType(int flag=0x0);
		//virtual tpQUANTI *GetQuantiBins(int flag=0x0) { throw"FeatVector::GetQuantiBins is ..."; }
		virtual inline int left_rigt(const size_t& t, const double& thrsh, const int lft, const int rgt, int flag = 0x0) { throw"FeatVector::left_rigt is ..."; }
		virtual inline int left_rigt(const void *pVal, const ARR_TREE*arr_tree,int no, int flag = 0x0) { throw"FeatVector::left_rigt is ..."; }
		virtual void SplitOn(FeatsOnFold *hData_, MT_BiSplit *hBest, int flag = 0x0) { 
			throw"FeatVector::Split is ..."; 
		}
		//根据MT_BLIT的模型，来预测
		virtual void Update_step(FeatsOnFold *hData_, MT_BiSplit *hBlit, int flag = 0x0) { throw"FeatVector::UpdatePredict is ..."; }
		virtual void Update_regression(FeatsOnFold *hData_, MT_BiSplit *hBlit, tpY* target, int flag = 0x0) { throw"FeatVector::UpdatePredict is ..."; }
		//与UpdatePredict配对使用
		virtual Regression *InitRegression(FeatsOnFold *hData_, MT_BiSplit *hBlit, int flag = 0x0) { throw"FeatVector::RegressionAt is ..."; }

		virtual void Set(double a, int flag = 0x0) { throw "FeatVector::Set is ..."; }
		//注意，统计信息记录在some_set
		virtual void STA_at(SAMP_SET& some_set, int flag = 0x0) { throw "FeatVector::STA_at is ..."; }
		virtual void Value_AtSamp(const SAMP_SET*samp_set, void *samp_values, int flag = 0x0) { throw "FeatVector::Values_AtSamp is ..."; }
		virtual inline void *pValue_AtSamp(const size_t&, int flag = 0x0) { throw "FeatVector::One_Value_AtSamp is ..."; }
		virtual void Observation_AtSamp(LiteBOM_Config config, SAMP_SET& samp, Distribution&distri, int flag=0x0)	{	throw "FeatVector::Observation_AtSamp is ...";	}
		virtual size_t UniqueCount(const SAMP_SET&samp_set, int flag=0x0)				{ throw "FeatVector::UniqueCount is ..."; }
		/*vResi=predict-target		pDown=target-predict*/
		virtual double UpdateResi(FeatsOnFold *hData_, int flag = 0x0) { throw "FeatVector::UpdateResi is ..."; }

		//virtual HistoGRAM *GetHisto(int flag = 0x0) {	return nullptr; }
		//static bin mapping	生成基于EDA的格子	参见Samp2Histo
		virtual void UpdateHisto(const FeatsOnFold *hData_, bool isOnY, bool isFirst, int flag = 0x0) { throw "FeatVector::UpdateHisto is ..."; }
		virtual void PerturbeHisto(const FeatsOnFold *hData_, int flag = 0x0) { throw "FeatVector::PerturbeHisto is ..."; }
		
		virtual void Merge4Quanti(const SAMP_SET*samp_0, int flag=0x0)	{ throw "FeatVector::Merge4Quanti is ..."; }
		virtual void Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, const tpSAMP_ID *samps4quanti=nullptr, int flag = 0x0) const
		{ throw "FeatVector::_Samp2Histo_ is ..."; }
		virtual void InitDistri(const FeatsOnFold *hFold, Distribution *tDistri, const SAMP_SET *samp_set, bool isGenHisto, int flag) { throw "FeatVector::InitDistri is ..."; }
		virtual void Distri4Merge(const FeatsOnFold *hFold, Distribution *M_Distri, const SAMP_SET *samp_set, bool isGenHisto, int flag) { throw "FeatVector::Distri4Merge is ..."; }

		virtual void QuantiAtEDA(ExploreDA *eda, void *quanti, int sizeofQ, int nMostBin, const FeatsOnFold *hData_, int flag) { ; }
		//virtual void Split2Quanti(const LiteBOM_Config&config, const ExploreDA *eda, vector<double>& vThrsh, HistoGRAM *qHisto, tpDOWN *yDown, int nMostBin, int flag = 0x0) { throw "FeatVector::SplitSort is ..."; }
		//virtual void UpdateFruit(const FeatsOnFold*,MT_BiSplit *hBlit, int flag = 0x0) {}
		virtual void RefineThrsh(const FeatsOnFold *hData_, const MT_BiSplit *hBlit, int flag = 0x0) {}
		//virtual void SetSplitInfo(MT_BiSplit *hBlit, FeatBlit&box, int flag = 0x0) { throw "FeatVector::SetSplit is ..."; }
		virtual void BinaryOperate(FeatVector*, BINARY_OPERATE opt, int flag = 0x0) { throw "FeatVector::BinaryOperate is ..."; }
	};
	
	FeatVector *FeatVecQ_InitInstance(FeatsOnFold *hFold, FeatVector *hFeat, int x, int flag = 0x0);

}


