#pragma once

#include ".\GruST\learn\DecisionTree.hpp"
#include ".\GruST\image\BMPfold.hpp"

namespace Grusoft{

class RF_ShapeRegress : public RandomForest{
	string sPre;
	char sLine[1000];

protected:
	bool isToCPP;
	FILE *fpC,*fpD,*fpT;

	float *cand_dis;
	arrPFNO featLines;
	void RandCandidate( int nCand,ShapeBMPfold &mean,ShapeBMPfold::PTFs &cands,int flag=0x0 );
	void RandCandidate_2( int nCand,ShapeBMPfold &mean,ShapeBMPfold::PTFs &cands,int flag=0x0 );
	virtual hBLIT GetBlit( WeakLearner *hWeak,int flag=0x0 );
	virtual void BlitSamps( WeakLearner *hWeak,SAMPs &fnL,SAMPs &fnR,int flag=0x0 );
	//virtual void ToCPP(WeakLearner *hWeak,int flag=0x0);
	virtual void Confi_Impuri(  WeakLearner *hWeak,int flag );
	virtual bool GetFeatDistri( WeakLearner *hWeak,float *distri=nullptr,int flag=0x0 );
	//bool Confi_Regress(  WeakLearner *hWeak,int flag );
	virtual bool LeafModel(  WeakLearner *hWeak,int flag=0x0 );
	virtual double ErrorAt( arrPFNO& samps );
	virtual void BootSample( DecisionTree *hTree,arrPFNO &boot,arrPFNO &oob,FeatData *hDat,int flag=0x0 );
	void FeatLineBmp( string sPath,int flag=0x0 );
	virtual int nPickAtSplit( WeakLearner *hWeak ){	
			return nPickWeak;	
		}
	virtual void DumpTree( int nox,DecisionTree *hTree,int flag=0x0 );
	virtual void UpdateFeat( int flag=0x0 );
	virtual void OnMultiTree( int cas,int nMulti,int flag=0x0 );
	virtual void AfterTrain( int cas,int nMulti,int flag=0x0 );

	ShapeBMPfold& spMean;
	ShapeBMPfold::PTFs cands,*arrPTF;
	double nzWeak;
	bool isCalcErr;
	int nBlitThread;
public:
	typedef enum{
		SINGLE_TREE,MULTI_TREE
	}REGULAR;
	REGULAR regular;

	typedef enum{
		BT_ALL,BT_MAX_ERR,BT_MIN_ERR,BT_RANDOM_3
	}BOOT;
	BOOT boot;
	ShapeBMPfold::PT_INDEX index;
	int nTree,dup,nOOB,no;
//	SHAPE_PtSet sp;
	arrPFNO Tests;
	double eta,lenda;
	//Eigen::MatrixXd mOff,mSum;
	ShapeBMPfold::VECT mOff,mSum;
	vector<ShapeBMPfold*> Trains;
	RF_ShapeRegress( ):RandomForest( ),nOOB(0),regular(SINGLE_TREE),boot(BT_MIN_ERR),spMean(ShapeBMPfold::VIRTU),
		isToCPP(false),fpC(NULL),fpD(NULL),fpT(NULL),arrPTF(nullptr){	;	}
	RF_ShapeRegress( vector<ShapeBMPfold*>&Trains,ShapeBMPfold &spMean,int nCand,int nStep,int nEach,int nOB,int cas,int flag=0x0);
	~RF_ShapeRegress();

	virtual int Train( string sTitle,int cas,int flag );
	virtual void AfterTrain( FeatData *hData,int flag=0x0 );
	void TraceBmp( string sPath,int type,int flag=0x0 );

	bool InitCPP( char *pathC,char *pathD,char *pathT,int type,int flag=0x0 );
	virtual void ToCPP( DecisionTree *hTree,int cas,int step,int tree,int flag=0x0 );
	virtual void ToCPP( int cas,int flag=0x0 );
	void CoreInCPP( int cas,int flag );
friend class RF_ConfiRegress;
};

}

