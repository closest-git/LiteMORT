#pragma once

#include ".\GruST\learn\DecisionTree.hpp"
#include ".\GruST\image\BMPfold.hpp"
#include "RF_ShapeRegress.h"

namespace Grusoft{

/*
	暂时从RF_ShapeRegress派生，借用其数据结构
	要归纳基类及虚拟接口确实很难			2/26/2016
*/
class RF_ConfiRegress : public RF_ShapeRegress{
	RF_ShapeRegress *hRegress;
	string sPre;
	char sLine[1000];

protected:

	//void RandCandidate( int nCand,ShapeBMPfold &mean,ShapeBMPfold::PTFs &cands,int flag=0x0 );
	//void RandCandidate_2( int nCand,ShapeBMPfold &mean,ShapeBMPfold::PTFs &cands,int flag=0x0 );
	virtual hBLIT GetBlit( WeakLearner *hWeak,int flag=0x0 );
	//virtual void ToCPP(WeakLearner *hWeak,int flag=0x0);
	virtual void Confi_Impuri(  WeakLearner *hWeak,int flag );
	//virtual bool GetFeatDistri( WeakLearner *hWeak,float *distri=nullptr,int flag=0x0 );
	bool L_Regress(  WeakLearner *hWeak,int flag );
	double LR( int nPt, Eigen::MatrixXd &mA,Eigen::VectorXd &rhs,Eigen::VectorXd &beta,int flag=0x0 );
	virtual bool LeafModel(  WeakLearner *hWeak,int flag=0x0 );
	virtual double ErrorAt( arrPFNO& samps );
	//virtual void BootSample( DecisionTree *hTree,arrPFNO &boot,arrPFNO &oob,FeatData *hDat,int flag=0x0 );

	virtual int nPickAtSplit( WeakLearner *hWeak ){	
			return nPickWeak;	
		}
	virtual void DumpTree( int nox,DecisionTree *hTree,int flag=0x0 );
	//virtual void UpdateFeat( int flag=0x0 );
	virtual void OnMultiTree( int cas,int nMulti,int flag=0x0 );
	virtual void AfterTrain( int cas,int nMulti,int flag=0x0 );


public:
	RF_ConfiRegress( RF_ShapeRegress *hReg_ );
	//RF_ConfiRegress( FeatData*hFeatDat,int flag=0x0 );
	virtual ~RF_ConfiRegress(){
		cand_dis=nullptr;
	}

	virtual int Train( string sTitle,int cas,int flag );
	virtual void AfterTrain( FeatData *hData,int flag=0x0 );
	//void TraceBmp( string sPath,int type,int flag=0x0 );
};

}

