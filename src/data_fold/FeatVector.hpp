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


	class FeatVector {
	protected:
	public:
		Distribution *hDistri = nullptr;
		enum {
			//CATEGORY = 0x100,
			AGGREGATE = 0x200,

			//V_ZERO_DEVIA = 0x10000,	//常值，一般可忽略
			IS_BUNDLE = 0x20000,	//in Feature Bundle	参见FeatsOnFold::nPick4Split
		};
		size_t type = 0x0;
		bool isCategory() { return BIT_TEST(type, Distribution::CATEGORY); }

		typedef enum {
			COPY_MEAN,
		}BINARY_OPERATE;
		int id = -1;
		int agg_no = -1;		//展开为一系列聚合特征
		
		string nam = "", desc = "";
		//virtual void Clear(int flag = 0x0) {		}
		virtual ~FeatVector() {
			if (hDistri != nullptr)
				delete hDistri;
		}
		virtual void Empty(int flag = 0x0) {		}
		virtual void Set(size_t len, void* p, int flag = 0x0) { throw"FeatVector::SetY is ..."; }
		virtual void Set(size_t pos, double a, int flag = 0x0) { throw"FeatVector::SetY is ..."; }
		virtual void loc(vector<tpSAMP_ID>&poss, double target, int flag = 0x0) { throw"FeatVector::loc is ..."; }
		virtual void SplitOn(FeatsOnFold *hData_, MT_BiSplit *hBest, int flag = 0x0) { throw"FeatVector::Split is ..."; }
		//根据MT_BLIT的模型，来预测
		virtual void Update_step(FeatsOnFold *hData_, MT_BiSplit *hBlit, int flag = 0x0) { throw"FeatVector::UpdatePredict is ..."; }
		virtual void Update_regression(FeatsOnFold *hData_, MT_BiSplit *hBlit, tpY* target, int flag = 0x0) { throw"FeatVector::UpdatePredict is ..."; }
		//与UpdatePredict配对使用
		virtual Regression *InitRegression(FeatsOnFold *hData_, MT_BiSplit *hBlit, int flag = 0x0) { throw"FeatVector::RegressionAt is ..."; }

		virtual void Set(double a, int flag = 0x0) { throw "FeatVector::Set is ..."; }
		virtual void Observation_AtSamp(LiteBOM_Config config, SAMP_SET& samp, Distribution&distri, int flag=0x0)	{	throw "FeatVector::Observation_AtSamp is ...";	}

		/*vResi=predict-target		pDown=target-predict*/
		virtual double UpdateResi(FeatsOnFold *hData_, int flag = 0x0) { throw "FeatVector::UpdateResi is ..."; }

		//static bin mapping	生成基于EDA的格子	参见Samp2Histo
		virtual void UpdateHisto(const FeatsOnFold *hData_, bool isOnY, bool isFirst, int flag = 0x0) { throw "FeatVector::UpdateHisto is ..."; }
		//virtual MT_BiSplit*OnPredict(const MT_BiSplit *cur_, const size_t no, int flag = 0x0)	 { throw"FeatVector::OnPredict is ..."; }
		virtual void Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, int flag = 0x0) { throw "FeatVector::Samp2Histo is ..."; }
		virtual void EDA(const LiteBOM_Config&config, ExploreDA *edaX, int flag) { throw "FeatVector::EDA is ..."; }
		virtual void QuantiAtEDA(const ExploreDA *eda, tpQUANTI *quanti, int nMostBin, int flag = 0x0) { ; }
		//virtual void Split2Quanti(const LiteBOM_Config&config, const ExploreDA *eda, vector<double>& vThrsh, HistoGRAM *qHisto, tpDOWN *yDown, int nMostBin, int flag = 0x0) { throw "FeatVector::SplitSort is ..."; }
		virtual void UpdateFruit(const FeatsOnFold*,MT_BiSplit *hBlit, int flag = 0x0) {}
		virtual void RefineThrsh(const FeatsOnFold *hData_, const MT_BiSplit *hBlit, int flag = 0x0) {}
		//virtual void SetSplitInfo(MT_BiSplit *hBlit, FeatBlit&box, int flag = 0x0) { throw "FeatVector::SetSplit is ..."; }
		virtual void BinaryOperate(FeatVector*, BINARY_OPERATE opt, int flag = 0x0) { throw "FeatVector::BinaryOperate is ..."; }
	};




}

