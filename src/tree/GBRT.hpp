#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <assert.h>
#include "BoostingForest.hpp"
using namespace std;

namespace Grusoft{	
	class FeatsOnFold;
	/*
		residual boosting mayber better 
	*/
	class GBRT : public BoostingForest {
		string sPre;
		//random_state
	protected:
		double shrinkage=0.1;
		double nzWeak;
		bool isCalcErr;
		int nBlitThread;
		virtual bool GetFeatDistri(WeakLearner *hWeak, float *distri = nullptr, int flag = 0x0);
		//virtual bool LeafModel(WeakLearner *hWeak, int flag = 0x0);
		//virtual void UpdateFeat(int flag);
		//virtual void BlitSamps(WeakLearner *hWeak, SAMPs &fnL, SAMPs &fnR, int flag = 0x0);
		//virtual hBLIT GetBlit(WeakLearner *hWeak, int flag = 0x0);
		virtual void GetYDistri(WeakLearner *hWeak, float *distri = nullptr, int flag = 0x0);
		//virtual void Confi_Impuri(WeakLearner *hWeak, int flag);
		virtual void AfterTrain(FeatsOnFold *hData, int cas, int nMulti, int flag = 0x0);
	public:

		tpDOWN mOff, mSum;

		typedef enum {
			SINGLE_TREE, MULTI_TREE
		}REGULAR;
		REGULAR regular = SINGLE_TREE;
		arrPFNO Tests;
		//double eta, lenda;

		typedef enum {
			BT_ALL, BT_MAX_ERR, BT_MIN_ERR, BT_RANDOM_3
		}BOOT;
		BOOT boot;
		int rounds, dup, nOOB, no;

		GBRT(FeatsOnFold *hTrain, FeatsOnFold *hEval, double sOOB, MODEL mo_, int nTree, int flag = 0x0);
		virtual ~GBRT() {
		}
		const LiteBOM_Config& Config() const {
			return hTrainData->config;
		}
		virtual void BeforeTrain(FeatsOnFold *hData, int flag = 0x0);
		virtual int Train(string sTitle, int cas, int flag=0x0);
		virtual int Prune(int flag = 0x0);
		virtual int IterTrain(int round,int flag);
		virtual double Predict(FeatsOnFold *hData,bool scan_nodes=true,bool checkLossy=false,bool resumeLast=false, int flag=0x0);
		virtual int Test(string sTitle, BoostingForest::CASEs& TestSet, int nCls, int flag);
		virtual bool isPass(hMTNode hNode, int flag = 0x0);

		
	};
}
