#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <assert.h>
using namespace std;

namespace Grusoft{
	class LMachine	{
	public:
		typedef std::mt19937* hRANDER;		//(pseudo) random generator
		typedef enum{
			CLASIFY,REGRESSION
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
			LMachine* hMachine;
			float rBase,rMax,rMin,gamma,lr;	
			SKDU( ):cascad(0),step(0),nStep(0),nTree(0),noT(-1),noLeaf(-1){}
		};
		

		struct CASE{
			float label,predict;		//for classification and 1-var regression
			int nBag;
			CASE( ):nBag(0),label(0.0),predict(0.0)		{	;	}
			virtual ~CASE( )							{;}
		};
		typedef vector<CASE*> CASEs;
		CASEs SamplSet;

	protected:
		bool isDumpLeaf;
		hRANDER hRander;
		MODEL model;
		int nThread;
		SKDU skdu;		
		void *user_data;

		double impurity,sBalance,eOOB,eInB;
		int nFeat,nClass,nPickWeak;
		vector<string>FeatNames;
		//vector<FeatsOnFold *> arrDat;
	public:
		string name;

		hRANDER InitRander( unsigned seed );
		virtual void Clear( );		

	};
}
