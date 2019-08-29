#pragma once

#include <vector>
#include <random>
#include <queue>
#include <algorithm>
#include <assert.h>
#include "GST_fno.hpp"
#include "./BiSplit.hpp"
//#include "../util/FeatsOnFold.hpp"
#include "../data_fold/DataFold.hpp"
#include "../util/GRander.hpp"
#ifdef ENGINE
	#include "./Eigen/Dense"
#endif
using namespace std;

namespace Grusoft{
	//暂时借用，是否合适?
	typedef arrPFNO SAMPs;

	typedef enum{
		RF_UNDEF=0,RF_TRAIN=1,RF_TEST,RF_VERIFY
	}RF_STAGE;


	struct BiSplit{		//binary split
		int id;
		double thrsh,gain;
		BiSplit( ): id(-1),thrsh(0),gain(0)	{	;	}
		virtual ~BiSplit( )		{	}
		virtual void SplitAt( SAMPs &samps,SAMPs &left,SAMPs &right,int flag=0x0 ){
			//for each ( F4NO *fn in samps  )	{
			for (auto fn : samps) {
					if( fn->val<=thrsh ){
					left.push_back(fn);
				}else
					right.push_back(fn);
			}
		}

	};
	typedef BiSplit *hBLIT;	
	
	struct BLIT_Diff : public BiSplit{
		int other;
		BLIT_Diff( ) : BiSplit( ),other(-1)	{	;	}
		virtual ~BLIT_Diff( )	{	}
	};
	
	class FeatsOnFold;
	class DecisionTree;
	class BoostingForest;
	struct WeakLearner	{	
		DecisionTree *hTree;
		//static double minSet,minGain;
		arrPFNO samps;
		float *distri,confi;
		hBLIT hBlit;
		int feat,depth,id,nLastSamp;
		double thrsh,impuri,err;
		WeakLearner	*left,*right;

		WeakLearner( ):feat(-1),thrsh(0.0),left(nullptr),right(nullptr),hTree(nullptr),distri(nullptr),hBlit(nullptr),depth(0){;}
		WeakLearner( DecisionTree *hTr,arrPFNO&sam_,int depth,int flag=0x0 );
		virtual ~WeakLearner( );	
		virtual void ClearSamps( int flag=0x0 )	{
			//for each(F4NO *hfn in samps)
			for ( auto hfn : samps )
				delete hfn;
			samps.clear( );		
		}

		static bool isBig( const WeakLearner *l,const WeakLearner *r)			{	return l->nLastSamp>r->nLastSamp;	}	

		virtual double s_plit( int cur,int *nnz,int *nos, double &thrsh,int flag=0x0 );
		virtual bool Split( int flag=0x0 );//	{	throw "WeakLearner::Split is ..."; }
		//virtual DecisionTree BinaryClasify( float *vec,int flag=0x0 );
		virtual void AtLeaf( int flag=0x0 );
		virtual bool Model2Distri( int nCls,double *distr,int flag=0x0 );

		virtual void ToCPP(int flag=0x0);
	};

	/**/
	struct RandClasify : public WeakLearner	{
	//	static double rou;
		RandClasify( DecisionTree *hTr,arrPFNO&sam_,int flag=0x0):WeakLearner(hTr,sam_,flag){;}
		virtual ~RandClasify( ){;}		
		virtual bool Split( int flag=0x0 )	{	throw (0); }
	};

	
	struct RandRegress : public WeakLearner	{
	//	static double rou;
		//Eigen::MatrixXf move,sum;
		tpDOWN move, sum;
#ifdef ENGINE		
		Eigen::VectorXd gressor;
#endif

		RandRegress( DecisionTree *hTr,arrPFNO&sam_,int depth,int flag=0x0) { throw (0); }
		virtual ~RandRegress( ){	;	}		
		virtual bool Split( int flag=0x0 ) { throw (0); }
		void AtLeaf( int flag ) { throw (0); }
		virtual double ConstRegress_0( int x=0x0,int flag=0x0 ) { throw (0); }
	};

	typedef std::vector<WeakLearner*>WeakLearners;
	class DecisionTree{
	protected:
		void *user_data=nullptr;
		FeatsOnFold *hData_ = nullptr;
		BoostingForest *hForest = nullptr;
		WeakLearner* root = nullptr;
		int nLeaf;
		//WeakLearners nodes;
		double impurity;
		//RF_STAGE stage;
		vector<F4NO>ginii;	//gini importance

		void BootSample( arrPFNO &boot,arrPFNO &oob,FeatsOnFold *,int flag=0x0 );
	public:
		arrPFNO oob;

		string name;			//only for debug
		DecisionTree( )	{}
		DecisionTree(BoostingForest *hF, FeatsOnFold *hD,int flag=0x0);
		virtual ~DecisionTree( );
		WeakLearner* hRoot( )	{	return root;	}
		FeatsOnFold *GetDat( )		{	return hData_;	}
		void GetNodes( WeakLearners&vNodes,int flag=0x0 );
		void GetLeaf( WeakLearners&vLeaf,int flag=0x0 );
		bool isTrained(  int flag=0x0 ){
			return hRoot()->feat>=0;
		}
		virtual void Train( int flag=0x0 );
		virtual void Clasify( FeatsOnFold *hSamp,arrPFNO &points,float *distri,int flag=0x0 );
		virtual void Regress( arrPFNO &points,int flag=0x0 );
	friend class WeakLearner;
	friend class RandClasify;
	friend class RandRegress;
	friend class BoostingForest;
	friend class MT_BiSplit;
	};
	typedef std::vector<DecisionTree*>DForest;

	class ManifoldTree : public DecisionTree {
	protected:
		ManifoldTree *hGuideTree = nullptr;
		//vector<int> samp_folds;		//fold越少越好
		virtual void OnNewLeaf(hMTNode node, FeatsOnFold *hData_, const vector<int> &pick_feats,bool isOnlyAdd, int flag = 0x0);
		virtual void GrowLeaf(hMTNode node,const char*info,bool isAtLeaf,int flag=0x0);
		virtual void BeforeEachBatch(size_t no_0,size_t no_1,int flag=0x0);
	public:
		int iter_refine = 1;
		MT_Nodes nodes;
		class _leaf_compare_ {
		public:
			bool operator()(hMTNode n0, hMTNode n1) const			{
				//return n0->gain_train<n1->gain_train;
				return n0->gain_<n1->gain_;
			}
		};
		std::priority_queue<hMTNode, std::vector<hMTNode>, _leaf_compare_> leafs;		//需要节省内存
		ManifoldTree(BoostingForest *hF, FeatsOnFold *hData, string nam_, int flag = 0x0);
		//切记	必须删除
		virtual void ClearSampSet( );			
		virtual ~ManifoldTree( )	{
			//for each(MT_BiSplit *node in nodes)
			for (auto node : nodes)
				delete node;
			nodes.clear( );		//samp_folds.clear( );
			if (hGuideTree != nullptr)
				delete hGuideTree;
		}		
		hMTNode hRoot() {	assert(nodes.size()>0);	return nodes[0]; }

		virtual void Train(int flag = 0x0);
		virtual void AddScore(INIT_SCORE *score,int flag=0x0);
		virtual void Dump(int flag = 0x0);
		virtual void DelChild(hMTNode hNode,int flag=0x0);
		virtual bool To_ARR_Tree(FeatsOnFold *hData_, ARR_TREE &arrTree, int flag = 0x0);
		//virtual void Predict(int flag = 0x0);

		virtual void SetGuideTree(ManifoldTree*hET, int flag = 0x0) {	hGuideTree = hET;	 }
	};
	
}
	double Impurity( int nCls,int nz,int *distri );

using namespace Grusoft;