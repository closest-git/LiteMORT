#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <assert.h>
#include "GST_fno.hpp"
#include "../data_fold/DataFold.hpp"
#include "../util/GRander.hpp"
//#include "./Eigen/Dense"
#include "./ManifoldTree.hpp"
using namespace std;

namespace Grusoft{
	struct EARLY_STOPPING {
		vector<double> errors;
		double e_best=DBL_MAX;
		int best_no=-1,best_round=-1, nBraeStep=0;
		int early_round = 10;
		int LR_jump = 0;
		EARLY_STOPPING(){}
		EARLY_STOPPING(int nEarly) : early_round(nEarly)
		{	}
		double curERR() {
			if (errors.size() == 0)
				return DBL_MAX;
			else
				return errors[errors.size() - 1];
		}
		void Add(double err, int best_round,bool& isLRjump, int flag=0x0);
		bool isOK(int cur_round);

		//Brae：苏格兰丘陵
		void CheckBrae(int flag = 0);	
	};

	typedef bool (*isTrueObj)(void* user_data,int flag);
	/*
		提升算法于maniflod-tree的结合

		广义的随机森林(GBRT一样可以看做是随机森林)
		建议从"../learn/LMachine.hpp"派生
		v0.1	cys
			7/17/2018
	*/
	class BoostingForest	{
	public:
		GRander rander_;
		typedef enum{
			CLASIFY,REGRESSION,
			//OUTLIER,		任然可以看做是REGRESSION
		}MODEL;
		enum{		//constant
			SAMPL_OOB=100,SAMPL_InB,
			RAND_REINIT=9991,
		};
		struct SKDU{		//Learn Schdule
			//each cascade contain nSteps.each step contain 1 or n trees
			int cascad,step,nStep,noT,nTree,noLeaf;		
			bool isLastStep( )	{	return step==nStep-1;	}
			bool isLastTree( )	{	return noT==nTree-1;	}
			BoostingForest* hForest;
			float rBase,rMax,rMin,gamma,lr;	
			SKDU( ):cascad(0),step(0),nStep(0),nTree(0),noT(-1),noLeaf(-1){}
		};
		
		EARLY_STOPPING stopping;
		HistoGRAM_BUFFER *histo_buffer=nullptr;

		/*
		需要整合
			traindata,typedef arrPFNO SAMPs;
		*/
		struct CASE{
			float label = FLT_MAX,predict = FLT_MAX,residual=FLT_MAX;		//for classification and 1-var regression
			int nBag;
			CASE( float l_=0,float p_=0 ):nBag(0),label(l_),predict(p_)	{	UpdateErr();	}
			virtual ~CASE( )							{;}
			double UpdateErr() {
				residual = (predict - label);
				//vR.noalias() = (vS-vY)*toIdea.transpose( );
				return residual;
			}
			void Move(const tpDOWN &step) {
				if (1)
					predict -= step;
				else {	//momentum update,竟然没用，真是古怪		1/20/2016
					//vMomen = 0.1*vMomen - step;
					//vS += vMomen;
				}
				//vS-=step*toThis.transpose( );
			}
		};
		typedef vector<CASE*> CASEs;
		CASEs SamplSet;

	protected:
		bool isDumpLeaf;
		//int rand_seed_0;
		//hRANDER hRander;
		MODEL model;
		RF_STAGE stage;
		int nTree,maxDepth,nThread;
		//int step;		//only for debug
		isTrueObj hfIsObj;
		void *user_data;

		double impurity,sBalance,eOOB,eInB;
		int nFeat,nClass,nPickWeak;
		vector<string>FeatNames;
		//vector<FeatsOnFold *> arrDat;
		
		FeatsOnFold *hTrainData=nullptr;
		FeatsOnFold *hTestData = nullptr;
		FeatsOnFold *hEvalData = nullptr;
		//FeatsOnFold at current stage(Train,test...)
		//vector<F4NO>ginii;	//gini importance
		
		
		int UniformInt(const int & min, const int & max );
		//bool TreeClasify( DecisionTree hTree,float *vec,double *distr,int flag=0x0 );
		virtual void InitFeat( int type,int nFlag=0x0 );
		void ErrEstimate( FeatsOnFold *hData,DForest& trees,int dataset,int flag );
		virtual hBLIT GetBlit( WeakLearner *hWeak,int flag=0x0 )		{	GST_THROW("BoostingForest::GetBlit is ...");	}

		//virtual void Confi_Impuri( WeakLearner *hWeak,int flag=0x0 )		{	GST_THROW("BoostingForest::Impurity is ...");	}
		virtual bool GetFeatDistri( WeakLearner *hWeak,float *distri=nullptr,int flag=0x0 )
		{		GST_THROW("BoostingForest::GetFeatDistri is ...");		}
		//需要给SAMPs赋值
		virtual void BlitSamps( WeakLearner *hWeak,SAMPs &fnL,SAMPs &fnR,int flag=0x0 )
		{		GST_THROW("BoostingForest::GetFeatDistri is ...");		}
		//virtual void GetYDistri( WeakLearner *hWeak,float *distri=nullptr,int flag=0x0 )		{		GST_THROW("BoostingForest::GetYDistri is ...");		}
		virtual bool LeafModel( WeakLearner *hWeak,int flag=0x0 )		{		GST_THROW("BoostingForest::LeafModel is ...");		}
		virtual double ErrorAt( WeakLearner *hWeak )		{		GST_THROW("BoostingForest::GetError is ...");		}
		virtual double ErrorAt( arrPFNO& samps )		{		GST_THROW("BoostingForest::GetError is ...");		}
		virtual void BootSample( DecisionTree *hTree,arrPFNO &boot,arrPFNO &oob,FeatsOnFold *hDat,int flag=0x0 ){
			hTree->BootSample( boot,oob,hDat,flag );
		}
		virtual int nPickAtSplit( WeakLearner *hWeak ){	
			return (int)( sqrt(hTrainData->nFeat()) );	
		}
		virtual void ToCPP(WeakLearner *hWeak,int flag=0x0){	GST_THROW("BoostingForest::ToCPP is ...");		}
		virtual void DumpTree( int nox,DecisionTree *hTree,int flag=0x0 ){	GST_THROW("BoostingForest::DumpTree is ...");		}
	public:
		DForest forest;
		SKDU skdu;
		string name;
		bool isClasify;
		BoostingForest( ):nFeat(0),nClass(0),nTree(0),impurity(0),stage(RF_UNDEF),hfIsObj(nullptr),user_data(nullptr),nThread(1),
			sBalance(-1),maxDepth(100000000),isDumpLeaf(false),hTrainData(nullptr),hTestData(nullptr)
		{ ;}
		virtual ~BoostingForest( ){	Clear( );	}

		virtual void SetUserData( void*ud_,isTrueObj hf,int flag=0x0 );
		bool isStage( RF_STAGE stg )	{	return stg==stage;	}
		virtual size_t nSample(int flag = 0x0) const { return hTrainData->nSample(); }

		//hRANDER InitRander( unsigned seed );
		virtual void Clear( );
		virtual void ClearData( );
		//FeatsOnFold *GurrentData() { FeatsOnFold *hData = hTrainData;		assert(hData != nullptr);		return hData; }

		virtual bool isPass(hMTNode hNode, int flag = 0x0) { GST_THROW("BoostingForest::isPass is ..."); }
		//尽量基于(InX,OutY)进行训练
		virtual void Train( FeatsOnFold *hData=nullptr,int flag=0x0 );
		virtual void AfterTrain( FeatsOnFold *hData,int flag=0x0 )	{;}
		virtual void TestOOB( FeatsOnFold *hData,int flag=0x0 );
		virtual void VerifyTree( DecisionTree hTree,FeatsOnFold *,int flag=0x0 );
		virtual void Clasify( int nSamp,FeatsOnFold *hSamp,int *cls,int flag=0x0 );
		virtual void Clasify( int nSamp,FeatsOnFold *hSamp,int flag=0x0 );
		//virtual int Clasify( float *val,double *distr,int flag=0x0 );
		virtual double rErr( )	{	return eOOB;	}
		//virtual int Save( string sPath,int flag=0x0 );
		//virtual int Load( string sPath,int flag=0x0 );
		virtual void ToCPP( const char *sPath_0,int flag ){		GST_THROW("BoostingForest::ToCPP is ...");		}
	friend RandClasify;
	friend DecisionTree;
	friend ManifoldTree;
	friend RandRegress;
	friend WeakLearner;
	friend MT_BiSplit;
	};
	
}
