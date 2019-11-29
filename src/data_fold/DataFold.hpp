#pragma once

#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <numeric>
#include <time.h>
#include "FeatVector.hpp"
#include "EDA.hpp"
using namespace std;
#include "../include/LiteBOM_config.h"
#include "./Representive.hpp"
#include "../tree/BiSplit.hpp"
#include "../util/GST_def.h"
//#include "../util/BLAS_t.hpp"
#include "../util/GRander.hpp"
#include "../EDA/Feat_Selection.hpp"
#include "Move.hpp"

//#include "../util/samp_set.hpp"

#ifdef WIN32
#include <tchar.h>
#include <assert.h>
#else    
#include <assert.h>
//#define assert(cond)
#endif

namespace Grusoft {
	class FeatsOnFold;
	class ManifoldTree;
	class FeatVec_LOSS;
	class BinarySwarm_GBDT;
	template<typename Tx> class FeatVec_T;

	
	/*
	比较微妙的数据结构
	1 train,test可以用不同的init
	2 从文件导入
	3 需要把INIT_SCORE转化到leaf->down_step

	v0.1	cys
		11/8/2018
	*/
	class INIT_SCORE {
	public:
		double step = 0;
		FeatVector *fVec = nullptr;
		virtual void Init(FeatsOnFold *Data_, int flag = 0x0);
		//virtual void ToDownStep(int flag = 0x0);
	};

	class Feat_Importance {
	protected:
		FeatsOnFold *hFold;
	public:
		vector<double> split_sum;
		vector<double> gain_sum;		//Friedman  节点分裂时收益越大，该节点对应的特征的重要度越高
		//https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances_faces.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-faces-py

		Feat_Importance(FeatsOnFold *hData_, int flag = 0x0);
		virtual ~Feat_Importance() {
			split_sum.clear();		gain_sum.clear();
		}
	};

	/*
	v0.1	cys
		7/27/2018
	*/
	class FeatsOnFold {
	public:
		string nam;
	private:
		tpDOWN *down = nullptr;	//negative_gradient	暂时借用
	protected:
		vector<float> sample_weight;
		//configuration paramters
	public:
		Feature_Selection *feat_salps = nullptr;
		Feat_Importance *importance = nullptr;
		Move_Accelerator *hMove = nullptr;
		GRander rander_samp, rander_feat,rander_bins,rander_nodes;
		INIT_SCORE init_score;
		vector<FeatVector*> feats;				//featrue values of X
		int nPickFeat = 0;
		vector<FeatVector*> merge_lefts;				//!DF_MERGE
		Representive present;

		struct BUFFER {
			tpSAMP_ID *samp_root_set = nullptr,*samp_left = nullptr,*samp_rigt = nullptr;
			double *samp_values = nullptr;
			
			virtual void Init(size_t nSamp, int flag = 0x0) {
				samp_root_set = new tpSAMP_ID[nSamp];
				samp_left = new tpSAMP_ID[nSamp];
				samp_rigt = new tpSAMP_ID[nSamp];
				samp_values = new double[nSamp];
			}
			virtual void Clear() {
				FREE_a(samp_root_set);			FREE_a(samp_left);
				FREE_a(samp_rigt);				FREE_a(samp_values);
			}
			virtual ~BUFFER() {
				Clear();				
			}
		};
		BUFFER buffer;

		virtual void InitMost(size_t nMost_,int flag=0x0) {
			nMost = nMost_;
			buffer.Init(nMost);
		}


		struct STAT {
			double dY;
			int nCheckGain = 0;

			double tX = 0;	//统计时间
			double tCheckGain = 0;
			double tHisto = 0;
			double tSamp2Histo = 0;
		};
		static STAT stat;
		ExploreDA *edaX = nullptr;
		LiteBOM_Config config;			//configuration paramters
										//tree每个node都有samp_set，会影响data
										//SAMP_SET samp_set;
										//label or target values of Y，Only one,多个Y显然需要多个模型来训练。	
		FeatVec_LOSS *lossy = nullptr;
		virtual string LOSSY_INFO(double err, int flag = 0x0);
		FeatVector* GetPrecict();
		FeatVector* GetY();
		template<typename T>
		T* GetY_(int flag=0x0) {
			FeatVector *hY = GetY();
			assert(hY!=nullptr);
			FeatVec_T<T> *hYd = dynamic_cast<FeatVec_T<T>*>(hY);
			T *y = hYd == nullptr ? nullptr : hYd->arr();
			return y;
		}
		template<typename T>
		T* GetPredict_(int flag = 0x0) {
			FeatVector *pred = GetPrecict();
			assert(pred != nullptr);
			FeatVec_T<T> *pred_ = dynamic_cast<FeatVec_T<T>*>(pred);
			T *y = pred_ == nullptr ? nullptr : pred_->arr();
			return y;
		}

		//pDown=target-predict
		tpDOWN *GetDownDirection()	const;
		tpDOWN *GetDeltaStep()	const;
		tpDOWN *GetHessian() const;
		tpDOWN *GetSampleDown()	const;
		tpDOWN *GetSampleHessian() const;	
		void *GetSampleValues()	const {		//不支持并行！！！
			return buffer.samp_values;
		}

		/**/
		bool atTrainTask()		const	{
			return config.task == LiteBOM_Config::kTrain;
		}
		bool atPredictTask()	const	{
			return config.task == LiteBOM_Config::kPredict;
		}
		bool isTrain()			const	{
			return BIT_TEST(dType, FeatsOnFold::DF_TRAIN);
		}
		bool isEval()			const {
			return BIT_TEST(dType, FeatsOnFold::DF_EVAL);
		}
		//内部数据必须保持不变！！！
		bool isMerge()			const {		
			return BIT_TEST(dType, FeatsOnFold::DF_MERGE);
		}

		enum {
			TAG_ZERO = 0x10000,
			DF_FLAG = 0xF00000, DF_TRAIN = 0x100000, DF_EVAL = 0x200000, DF_PREDIC = 0x400000,
			DF_MERGE = 0x800000,
		};
		size_t dType = 0x0;
		bool qOrder = false, isQuanti = false;
		static std::string sDumpFolder;
		double rOK = 0, err, impurity = 0;
		size_t nMost = 0;

		int *info = nullptr, *permut = nullptr;		//兼容以前的版本
		int nCls = 0;								//仅适用于分类问题

		vector<int>nEach;						//仅适用于分类问题???

		float *distri = nullptr;
		//tpDOWN *resi = nullptr, *move = nullptr;			//gradient boosting

		FeatsOnFold(LiteBOM_Config confi_, ExploreDA *eda, string nam_, int dtype = 0x0);
		virtual void InitRanders(int flag=0x0);
		/*
		template<typename Tx, typename Ty>
		void Init_T(size_t nMost_, size_t ldX_, size_t ldY_, int flag) {
			nMost = nMost_;
			size_t i;
			for (i = 0; i < ldX_; i++) {
				string desc = "feat_";
				feats.push_back(new FeatVec_T<Tx>(nMost, i, desc + std::to_string(i)));
			}
			//lossy.Init_T<Ty>(nMost, x, flag);
			//samp_set.Alloc(nMost);
			if (hMove != nullptr)
				hMove->Init_T<Tx, Ty>(nMost);
			importance = new Feat_Importance(this);
		}*/

		virtual ~FeatsOnFold();

		int *Tag();
		virtual void Empty(int flag) {

		}
		//virtual void Compress(int flag = 0x0);
		virtual void ExpandFeat(int flag = 0x0);
		virtual void ExpandMerge(const vector<FeatsOnFold *>&merge_sets, int flag = 0x0);
		virtual void Feature_Bundling(int flag = 0x0);
		virtual void Reshape(int flag = 0x0) {
			throw "FeatsOnFold::Reshape is ...";
		}
		virtual void Shuffle(int flag = 0x0) {
			throw "FeatsOnFold::Reshape is ...";
		}
		size_t nSample(bool isCheck = false) const {
			if (isCheck) {		//有时nSample<nMost，必须重新设定nMost
								//size_t i;
								//for (i = 0; i < nMost ; i++)					;
			}
			return nMost;
		}

		/*float *GetSampWeight(int flag=0x0) {
			if (lossy == nullptr)
				return nullptr;
			return lossy->GetSampWeight(flag);
		}*/
		virtual size_t nFeat()	const { return feats.size(); }
		FeatVector *Feat(int no) {
			if (no == -1)
				//return lossy.predict;
				return GetPrecict();
			if (no < -1 || no >= nFeat()) {
				throw "Feat no is OUT OF RANGE!!!";
			}
			return feats[no];
		}
		FeatVector *Feat(const string&name) {
			return nullptr;
		}
		//histo空间所依据的Distri，总是来自于train_data
		Distribution *histoDistri(const FeatVector *hFeat, int flag=0x0)	const;

		virtual int *Rank4Feat(int type, int flag = 0x0);	
		virtual void nPick4Split(vector<int>&picks, GRander&rander, BoostingForest *hForest,int flag = 0x0);

		//核心函数 
		virtual void SplitOn(MT_BiSplit *hBlit, int flag = 0x0);

		template<typename Ty>
		bool DeltastepOnTree(const ARR_TREE&tree, int flag) {
			size_t nSamp = nSample(), nNode = tree.nNodes, no, nFeat = feats.size(), step = nSamp;
			G_INT_64 t;
			double *thrsh_step = tree.thrsh_step;
			tpDOWN *delta_step = GetDeltaStep();			assert(delta_step != nullptr);
			int *feat_ids = tree.feat_ids, *left = tree.left, *rigt = tree.rigt;
			int num_threads = OMP_FOR_STATIC_1(nSamp, step);
#pragma omp parallel for schedule(static,1)
			for (int thread = 0; thread < num_threads; thread++) {
				size_t start = thread*step, end = min(start + step, nSamp), t;
				for (t = start; t < end; t++) {
					int no = 0, feat;
					while (no != -1) {
						if (left[no] == -1) {
							delta_step[t] = thrsh_step[no];		//Adpative_LR也需要该信息
							break;
						}
						else {
							assert(rigt[no] != -1);
							FeatVector *hFT = feats[feat_ids[no]];
							no = hFT->left_rigt(hFT->pValue_AtSamp(t), &tree, no);
						}
					}
				}
			}
			return true;
		}

		//int  OMP_FOR_STATIC_1(const size_t nSamp, size_t& step, int flag = 0x0);
		template<typename Ty>
		bool PredictOnTree(ManifoldTree*hTree,const ARR_TREE&tree, int flag) {
			if (tree.weight == 0) {
				return true;	//pass
			}
			size_t nSamp = nSample(), nNode = tree.nNodes, no, nFeat = feats.size(), step = nSamp;
			G_INT_64 t;
			double *thrsh_step = tree.thrsh_step;
			FeatVec_T<Ty> *predict = dynamic_cast<FeatVec_T<Ty>*>(GetPrecict());
			if (predict == nullptr)
				return false;
			tpDOWN *delta_step = GetDeltaStep();			//assert(delta_step != nullptr);
			Ty *pred = predict->arr();
			int *feat_ids = tree.feat_ids, *left = tree.left, *rigt = tree.rigt, flagLR=0x0;
			if(false){	//仅用于莫名其妙
				FeatVector *hFT = feats[30];
				double *vals = new double[nSamp];
				hFT->Value_AtSamp(nullptr,vals);
				printf("%g", vals[0]);
			}
			/**/
			int num_threads = OMP_FOR_STATIC_1(nSamp, step);
#pragma omp parallel for schedule(static,1)
			for (int thread = 0; thread < num_threads; thread++) {
				size_t start = thread*step, end = min(start + step, nSamp), t;
				for (t = start; t < end; t++) {
					/*if (nSamp == 2955)				{//仅用于调试
						t = 1;		flagLR = 0x1000000;
						FeatVector *hFT = feats[10];
						void *pVal = hFT->pValue_AtSamp(t);
						printf("\nTree[%s]@%d\t", hTree->name.c_str(), t);
					}*/
					//if (t == 7)		
					//	t = 7;
					int no = 0, feat;
					while (no != -1) {
						if (left[no] == -1) {
							pred[t] += thrsh_step[no]* tree.weight;
							if(delta_step != nullptr)
								delta_step[t] = thrsh_step[no];		//Adpative_LR也需要该信息
							//samp_at_leaf[t] = no;
							break;
						}
						else {
							assert(rigt[no] != -1);
							FeatVector *hFT = feats[feat_ids[no]];
							void *pVal = hFT->pValue_AtSamp(t);
							//no = hFT->left_rigt(t,thrsh_step[no], left[no], rigt[no]);		
							no = hFT->left_rigt(pVal, &tree,no, flagLR);
						}
					}					
				}

			}
			
			return true;
		}

		/*
			尽量优化
		
		template<typename Tx, typename Ty>
		bool PredictOnTree_0(const ARR_TREE&tree, int flag) {
			size_t nSamp = nSample(), nNode = tree.nNode, no, nFeat = feats.size(), step = nSamp;
			G_INT_64 t;
			double *thrsh_step = tree.thrsh_step;
			FeatVec_T<Ty> *predict = dynamic_cast<FeatVec_T<Ty>*>(GetPrecict());
			if (predict == nullptr)
				return false;
			Ty *pred = predict->arr();
			int *feat_ids = tree.feat_ids, *left = tree.left, *rigt = tree.rigt;
			Tx **arrFeat = new Tx*[nFeat], *val = nullptr;
			for (t = 0; t < nFeat; t++) {
				if (feats[t]->hDistri != nullptr && feats[t]->hDistri->isPass())
				{		arrFeat[t] = nullptr;		}
				else {
					FeatVec_T<Tx>*hFT = dynamic_cast<FeatVec_T<Tx>*>(feats[t]);
					if (hFT == nullptr)
					{						delete[] arrFeat;	return false;
					}
					arrFeat[t] = hFT->arr();
				}
			}
	
			int num_threads = OMP_FOR_STATIC_1(nSamp, step);
#pragma omp parallel for schedule(static,1)
			for (int thread = 0; thread < num_threads; thread++) {
				size_t start = thread*step, end = min(start + step, nSamp), t;
				for (t = start; t < end; t++) {
					int no = 0, feat;
					while (no != -1) {
						if (left[no] == -1) {
							pred[t] += thrsh_step[no];		
							//samp_at_leaf[t] = no;
							break;
						}
						else {
							assert(rigt[no] != -1);
							Tx *feat = arrFeat[feat_ids[no]];
							if (IS_NAN_INF(feat[t]) ) {
								no = rigt[no];
							}	else {
								if (feat[t] < thrsh_step[no]) {
									no = left[no];
								}
								else {
									no = rigt[no];
								}
							}
						}
					}
				}

			}
			delete[] arrFeat;
			return true;
		}*/

		template<typename Tx, typename Ty>
		bool UpdateStepOnReduce(ARR_TREE*treeEval, ARR_TREE*tree4train, int flag=0x0) {
			ARR_TREE& tree = *treeEval;
			size_t nSamp = nSample(), nNode = tree.nNodes, no, nFeat = feats.size(), step = nSamp;
			G_INT_64 t;
			double *thrsh_step = tree.thrsh_step;
			FeatVec_T<Ty> *predict = dynamic_cast<FeatVec_T<Ty>*>(GetPrecict());
			if (predict == nullptr)
				return false;
			FeatVec_T<Ty> *ftY = dynamic_cast<FeatVec_T<Ty>*>(GetY());
			if (ftY == nullptr)
				return false;
			Ty *pred = predict->arr(),*Y=ftY->arr();
			int *feat_ids = tree.feat_ids, *left = tree.left, *rigt = tree.rigt;
			Tx **arrFeat = new Tx*[nFeat], *val = nullptr;
			double *reduce = new double[nNode]();
			for (t = 0; t < nFeat; t++) {
				if (feats[t]->hDistri != nullptr && feats[t]->hDistri->isPass())	{
					arrFeat[t] = nullptr;
				}
				else {
					FeatVec_T<Tx>*hFT = dynamic_cast<FeatVec_T<Tx>*>(feats[t]);
					if (hFT == nullptr)
					{
						delete[] arrFeat;	return false;
					}
					arrFeat[t] = hFT->arr();
				}
			}
			/**/
			int num_threads = OMP_FOR_STATIC_1(nSamp, step);			
//#pragma omp parallel for schedule(static,1)
			for (int thread = 0; thread < num_threads; thread++) {
				size_t start = thread*step, end = min(start + step, nSamp), t;
				double err_0,err_1;
				for (t = start; t < end; t++) {
					int no = 0, feat;
					while (no != -1) {
						if (left[no] == -1) {
							err_0 = fabs(Y[t]-pred[t]);
							err_1 = fabs(Y[t] -( thrsh_step[no]+pred[t]));
							reduce[no] += err_0-err_1;
							break;
						}
						else {
							assert(rigt[no] != -1);
							Tx *feat = arrFeat[feat_ids[no]];
							if (IS_NAN_INF(feat[t])) {
								no = rigt[no];
							}
							else {
								if (feat[t] < thrsh_step[no]) {
									no = left[no];
								}
								else {
									no = rigt[no];
								}
							}
						}
					}
				}
			}
			for (no = 0; no < nNode; no++) {
				if (left[no] >= 0)		continue;
				if (reduce[no] < 0) {	//没有改进，不如归零
					thrsh_step[no] = 0;
					tree4train->thrsh_step[no] = 0;
				}
			}
			delete[] arrFeat;
			delete[] reduce;
			return true;
		}

		/*
			Train(Observation_AtSamp) predict(Update_step)
		*/
		virtual void AtLeaf(MT_BiSplit *hBlit, int flag = 0x0) {
			//FeatVector *hF_ = Feat(hBlit->feat_id);
			assert(hBlit->isLeaf());
			if (atTrainTask()) {
				hBlit->Observation_AtLocalSamp(this);
				//hBlit->Init_BFold(this);
			}
			else {
				FeatVector *predict = GetPrecict();
				/*if (hBlit->regression != nullptr ) {
				if(hBlit->feat_regress < 0)		throw	"At_Leaf feat_regress is <0";
				FeatVector *hFeat = Feat(hBlit->feat_regress);
				hFeat->Update_regression(this, hBlit, target, flag);
				}	else if( (hBlit->fruit != nullptr && hBlit->fruit->histo != nullptr) ){
				FeatVector *hFeat = Feat(hBlit->feat_id);
				hFeat->Update_regression(this, hBlit, target, flag);
				} else*/ {
					predict->Update_step(this, hBlit, flag);
				}
			}
		}

		void Distri2Tag(int *mark, int nCls, int flag);

		static FeatsOnFold* read_json(const string &sPath, int flag);
		//virtual void Stat(string sTitle, int flag = 0x0)	{ throw ("FeatsOnFold::Stat unimplemented!!!"); }

		virtual int Shrink(int nTo, int flag) { throw ("FeatsOnFold::Shrink unimplemented!!!"); }

		virtual void BeforePredict(int flag = 0x0);
		virtual void BeforeTrain(BoostingForest *hGBRT, int flag = 0x0);
		virtual void AfterTrain(int flag = 0x0);
		//sample_gradient,sample_hessian
		virtual void PickSample_GH(MT_BiSplit*hBlit, int flag=0x0);

		/*virtual void At(size_t k, void **X_, void **Y_, void**p_) const { throw ("FeatsOnFold::At is ...!!!"); }
		virtual bool ImportFile(size_t no, const wchar_t *sPath, int tg, int flag = 0x0) { throw ("FeatsOnFold::ImportFile unimplemented!!!"); }
		virtual int load_fp(FILE *fp, int flag) { throw ("FeatsOnFold::load_fp unimplemented!!!"); }
		virtual int save_fp(FILE *fp, int flag) { throw ("FeatsOnFold::save_fp unimplemented!!!"); }*/

		virtual int ToBmp(int epoch, int _x = 0, int flag = 0x0) { throw ("FeatsOnFold::ToBmp unimplemented!!!"); }


		virtual double Normal(int type, int flag) { throw ("FeatsOnFold::Normal unimplemented!!!"); }
	};
	typedef FeatsOnFold* hDataFold;

	template<typename Tx>
	class FeatVec_T : public FeatVector {
	protected:
		//std::vector<Tx> val;
		size_t nSamp_0=0;
		Tx *val=nullptr;
		//map<hMTNode,Tx> BLIT_thrsh,BLIT_mean;
	public:
		FeatVec_T() { 	}
		FeatVec_T(size_t _len, int id_, const string&des_, int flag = 0x0)  {
			id = id_;
			nSamp_0 = _len;
			desc = des_;	assert(_len > 0);	
			type = flag;
			if (isReferVal()) {

			}	else
				val = new Tx[_len];	// val.resize(_len);
		}

		inline size_t size()	const		{ 
			assert(nSamp_0>0); return nSamp_0; 
		}
		/*virtual size_t nSamp() const {	
			assert(nSamp_0>0); return nSamp_0;
		}*/

		virtual ~FeatVec_T() {
			FreeVals();
		}

		virtual void FreeVals(int flag=0x0) {
			if (isReferVal()) {
				;// printf("");
			}
			else if (val != nullptr)			{
				delete[] val;		val = nullptr;
			}
		}

		//{	assert(_len>0);	val.resize(_len,(Tx)0);	 }

		//Tx* arr() { return VECTOR2ARR(val); }
		Tx* arr()					{	return val;				}
		const Tx* arr()	const		{	return (const Tx*) val;	}

		virtual void Set(size_t len, PY_COLUMN *col, int flag = 0x0) { 
			if (len != size())
				throw "FeatVec_T::Set len mismatch!!!";
			if (isReferVal()) {
				val = (Tx*)(col->data);
			}
			else {
				col->CopyTo_(len, val);
			}
		}


		virtual void Set(size_t len, void* src, int flag = 0x0) {
			//assert(len == val.size());
			size_t szT = sizeof(Tx);
			if (len != size())
				throw "FeatVec_T::Set len mismatch!!!";
			void* dst = arr();
			if (isReferVal()) {
				val = (Tx*)src;
			}
			else
				memcpy(dst, src, sizeof(Tx)*len);
		}
		virtual void CopyFrom(const FeatVector*src, int flag = 0x0) {
			size_t nSamp_ = size();
			assert(nSamp_ == src->size());
			const FeatVec_T<Tx> *tSrc = dynamic_cast<const FeatVec_T<Tx>*>(src);
			assert(tSrc != nullptr);
			memcpy(this->val, tSrc->val, sizeof(Tx)*nSamp_);
		}

		virtual void Set(double a, int flag = 0x0) {
			size_t len = size(), i;
			Tx *x_ = arr();
			for (i = 0; i < len; i++)
				x_[i] = a;
		}
		virtual void Set(size_t pos, double a, int flag = 0x0) {
			val[pos] = (Tx)a;
		}
		/*virtual void Clear(int flag = 0x0) {
			val.clear();
		}*/
		virtual void Empty(int flag = 0x0) {
			memset(val, 0, size() * sizeof(Tx));
		}

		virtual void STA_at(SAMP_SET& some_set, int flag = 0x0) {
			size_t nS = some_set.nSamp, i, pos;
			Tx  a_0, a_1, *val_0 = arr();
			double a2, a_sum;
			some_set.STA_at_<Tx>(val_0, a2, a_sum, a_0, a_1, false);
		}

		virtual void Value_AtSamp(const SAMP_SET*samp_set, void *samp_val_, int flag = 0x0) {
			size_t i, nMost = this->size(),nSamp=nMost;
			Tx *samp_values = (Tx *)(samp_val_);
			if (samp_set == nullptr) {
				for (i = 0; i < nSamp; i++) {
					samp_values[i] = val[i];
				}
			}
			else {
				nSamp = samp_set->nSamp;
				tpSAMP_ID samp, *samps = samp_set->samps;
				//#pragma omp parallel for schedule(static)
				for (i = 0; i < nSamp; i++) {
					samp = samps[i];
					assert(samp >= 0 && samp < nMost);
					samp_values[i] = val[samp];
				}
			}
		}
		virtual inline void* pValue_AtSamp(const size_t& samp, int flag = 0x0) {
			//assert(samp >= 0 && samp < size());
			if (samp < 0 || samp >= size()) {
				printf("!!!pValue_AtSamp!!! samp=%lld,size=%lld", samp, size());
				//throw "!!!pValue_AtSamp!!!";
				return val + size()-1;
			}
			return val + samp;
		}

		//参见MT_BiSplit::Observation_AtSamp(FeatsOnFold *hData_, int flag) 
		virtual void Observation_AtSamp(LiteBOM_Config config, SAMP_SET& some_set, Distribution&distri, int flag = 0x0) {
			assert(0);
			size_t nS = some_set.nSamp, i, pos;
			Tx  a_0, a_1, *val_s = new Tx[nS], *val_0 = arr();
			double a2, a_sum;
			some_set.STA_at_<Tx>(val_0, a2, a_sum, a_0, a_1, false);
			distri.vMin = a_0, distri.vMax = a_1;
			for (i = 0; i < nS; i++) {
				pos = some_set.samps[i];
				val_s[i] = val_0[pos];
			}
			distri.X2Histo_<Tx, Tx>(config, nS, val_s, nullptr);
			delete[] val_s;
		}

		virtual void Merge4Quanti(const SAMP_SET*samp_set, int flag = 0x0) {
			assert(BIT_TEST(type, FeatVector::AGGREGATE));
			//if (samp4quanti == nullptr)
			//	delete[] samp4quanti;
			size_t nS = samp_set == nullptr ? this->size() : samp_set->nSamp, i;
			tpSAMP_ID pos;
			int map;
			if (samp_set == nullptr) {
				for (i = 0; i < nS; i++) {
					if (IS_NAN_INF(val[i]) || val[i] < 0) {
						map4set[i] = tpSAMP_ID_NAN;
					}	else {
						map = (int)(val[i]);
						map4set[i] = (tpSAMP_ID)map;
					}
				}
			}
			else {
				for (i = 0; i < nS; i++) {
					pos = samp_set->samps[i];
					if (IS_NAN_INF(val[pos]) || val[i] < 0){
						map4set[i] = tpSAMP_ID_NAN;
					}	else {
						map = (int)(val[pos]);
						map4set[i] = (tpSAMP_ID)map;
					}
				}			

			}
		}


		virtual void loc(vector<tpSAMP_ID>&poss,double target,int flag=0x0) {
			poss.clear();
			size_t i, nSamp_=size();
			for (i = 0; i < nSamp_; i++) {
				if (val[i] == target)
					poss.push_back(i);
			}
		}

		virtual Regression *InitRegression(FeatsOnFold *hData_, MT_BiSplit *hBlit, int flag = 0x0) {
			throw "Regression *InitRegression is ...";
			/*size_t i, nSamp = hBlit->nSample();
			tpDOWN *down = hData_->GetDownDirection();
			if (nSamp < 10000)
				return nullptr;

			Regression *regress = new Regression(hData_->config.leaf_regression);
			if (regress->Fit<Tx, tpDOWN>(nSamp, hBlit->samp_set.samps, arr(), down) == false) {
				//printf("!!!InitRegression failed!!! nSamp=%d", nSamp);
				delete regress;
				return nullptr;
			}
			else {
				//printf("InitRegression ... nSamp=%d", nSamp);
				return regress;

			}*/
		}		

		/*
			2/26/2019	Adaptive Learning_rate需要重新设计(用eval调整,会加大overfit)
		*/
		virtual void Update_step(FeatsOnFold *hData_, MT_BiSplit *hBlit, int flag = 0x0) {
			bool isTrain = BIT_TEST(hData_->dType, FeatsOnFold::DF_TRAIN);
			bool isEval = BIT_TEST(hData_->dType, FeatsOnFold::DF_EVAL);
			bool isAdaptive = hData_->config.lr_adptive_leaf;
			assert(hData_->atPredictTask());
			double sDrop = hData_->config.drop_out;
			//double shrink = hData_->config.learning_rate;
			tpDOWN step_base = hBlit->GetDownStep();
			if (isAdaptive) {
				assert(hBlit->lr_eta ==1.0);
				double s[] = { 0.1,0.5,1,2,5,10 };
				double err, min_err=DBL_MAX,a, eta_bst=1.0;
				FeatVector* hY0=hData_->GetY();
				FeatVec_T<Tx> *hY = dynamic_cast<FeatVec_T<Tx>*>(hY0);		assert(hY!=nullptr);
				const Tx* target = (hY->val);
				size_t i,loop, nLoop = sizeof(s) / sizeof(double), nSamp= hBlit->nSample();
				tpSAMP_ID *samps=hBlit->samp_set.samps, samp;
				for (loop = 0; loop < nLoop; loop++) {
					double step = step_base*s[loop];
					for (err=0,i = 0; i<nSamp; i++) {
						samp = samps[i];
						a = val[samp] + step-target[samp];		//对应于UpdateResi
						err += a*a;
					}					
					err = sqrt(err/nSamp);
					if (err < min_err) {
						min_err = err;		eta_bst = s[loop];
					}
				}
				hBlit->lr_eta = eta_bst;
				hBlit->samp_set.Update(val, step_base*hBlit->lr_eta);

			}
			else/**/ {
				/*if (sDrop < 1.0 ) {
					std::uniform_real_distribution<double> unif(0, 1);
					double current = unif(hData_->rng);
					if (current < sDrop)
						isUpdate = false;	//hBlit->samp_set.Update(val, step);
				}*/
				hBlit->samp_set.Update(val, step_base);
			}
			//printf("%.4g; ", step);
			/**/
		}

		virtual void Update_regression(FeatsOnFold *hData_, MT_BiSplit *hBlit, tpY* target, int flag = 0x0) {
			assert(hData_->atPredictTask());
			double sDrop = hData_->config.drop_out, f = 0;
			double shrink = 1;//hData_->config.learning_rate;
			tpDOWN step = hBlit->GetDownStep()*shrink;

			bool isTrain = BIT_TEST(hData_->dType, FeatsOnFold::DF_TRAIN);
			tpSAMP_ID samp;
			size_t i, nSamp = hBlit->nSample();
			Tx *arrX = arr();
			Regression *regress = hBlit->regression;
			const HistoGRAM *histo = hBlit->fruit->histo_refer;
			//printf("%.4g; ", step*shrink);
			if (regress == nullptr && histo == nullptr)	throw "Update_regression regress=nullptr!!!";
			for (i = 0; i < nSamp; i++) {
				samp = hBlit->samp_set.samps[i];
				if (regress != nullptr)
					f = regress->At(arrX[samp])*shrink;
				else {
					if (histo->At(arrX[samp], f))
						f *= shrink;
					else
						f = step;
					//f = step;
				}
				target[samp] += f;		//对应于UpdateResi
			}
		}

		virtual inline int left_rigt(const size_t& t, const double& thrsh, const int lft, const int rgt,int flag=0x0) {
			if (IS_NAN_INF(val[t])) {
				return rgt;
			}	else {
				if (val[t] < thrsh) {
					return lft;
				}	else {
					return rgt;
				}
			}
			return lft;
		}
		virtual inline int left_rigt(const void *pVal, const ARR_TREE*arr_tree, int no, int flag = 0x0) {
		//virtual inline int left_rigt(const size_t& t, const ARR_TREE*arr_tree,int no, int flag = 0x0) {
			//void *pVal = pValue_AtSamp(t);		//需要优化啊
			Tx val_t=*(Tx*)pVal;// = val[t];
			
			const int lft = arr_tree->left[no], rgt = arr_tree->rigt[no],feat=arr_tree->feat_ids[no];
			assert(no >= 0 && no < arr_tree->nNodes);
			const tpFOLD *fold_map = arr_tree->folds[no];
			if (fold_map == nullptr) {
				if (IS_NAN_INF(val_t)) {
#ifdef _DEBUG
					if (flag == 0x1000000) {
						printf("\t%dF%d<>=%d", no, feat, rgt);
					}
#endif					
					return rgt;
				}	else {
					int child = val_t < arr_tree->thrsh_step[no] ? lft : rgt;
#ifdef _DEBUG
					if (flag == 0x1000000) {
						printf("\t%dF%d<%.8g,%.8g>=%d",no, feat, double(val_t), arr_tree->thrsh_step[no],child);
					}
#endif
					return child;

				}
			}
			else {
				if (IS_NAN_INF(val_t)) {
#ifdef _DEBUG
					if (flag == 0x1000000) {
						printf("\t%dF%d<>=%d", no, feat, rgt);
					}
#endif	
					return rgt;
				}	else {
					int i_val = (int)(val_t);
					tpFOLD fold = fold_map[i_val];
					assert(fold==0 || fold==1);
					int child = (fold <= 0) ? lft : rgt;
#ifdef _DEBUG
					if (flag == 0x1000000) {
						printf("\t%dF%d<%d:%d>=%d", no, feat, i_val, fold,child);
					}
#endif					
					return child;
				}
			}
			return lft;
		}

		void _core_isY_(bool isQuanti, const tpSAMP_ID samp,const tpFOLD *mapFolds, int pos, tpSAMP_ID*left, G_INT_64&nLeft, tpSAMP_ID*rigt, G_INT_64&nRigt, int flag) {
			//assert(samp >= 0 && samp <= 1152);
			tpFOLD fold = mapFolds[pos];	// hBlit->fruit->GetFold(pos);
			assert(fold == 0 || fold == 1);
			if (fold <= 0)
				left[nLeft++] = samp;
			else
				rigt[nRigt++] = samp;
		}

		//不支持并行！！！
		virtual void SplitOn(FeatsOnFold *hData_, MT_BiSplit *hBlit, int flag = 0x0) {
			Tx *samp_val = (Tx*)(hData_->GetSampleValues());
			//assert (BLIT_thrsh.find(hBlit) != BLIT_thrsh.end());
			tpSAMP_ID samp;
			bool isQuanti = hData_->isQuanti;		//predict,test对应的数据集并没有格子化!!!
			//double thrsh = hData_->isQuanti ? hBlit->fruit->T_quanti : hBlit->fruit->thrshold;
			//double thrsh = hBlit->fruit->Thrshold(hData_->isQuanti);
			//if( hBlit->fruit->isY )
			//	thrsh = hBlit->fruit->thrshold;
			MT_BiSplit *left = hBlit->left, *rigt = hBlit->right, *child = nullptr;
			int pos,fold=-1;
			size_t nSamp = hBlit->nSample(), i, nLeft = 0, nRigt = 0, step;
			if (hData_->atTrainTask())
				assert(left != nullptr && rigt != nullptr);
			else
				assert(left != nullptr && rigt != nullptr);
			assert(left->nSample() == 0 && rigt->nSample() == 0);
			/*
				还是有问题	12/19/2018	cys
				1/11/2019	发现train的histo->AtBin_不适用于test
			*/
			//double T_0 = hBlit->fruit->bin_S0.split_F, T_1 = hBlit->fruit->bin_S1.split_F;
			double T_1 = hBlit->fruit->adaptive_thrsh;
			if (hBlit->fruit->isY) {
				GST_TIC(t1);
				const tpFOLD *mapFolds = hBlit->fruit->mapFolds;
				SAMP_SET &lSet = left->samp_set, &rSet = rigt->samp_set;
				tpSAMP_ID *samps = hBlit->samp_set.samps,  *rigt = hBlit->samp_set.rigt;//*left = hBlit->samp_set.left,
				lSet = hBlit->samp_set;		rSet = hBlit->samp_set;	//直接复制父节点的一些数据
				lSet.isRef = true;			rSet.isRef = true;
				const HistoGRAM *histo = hBlit->fruit->histo_refer;
				int num_threads = OMP_FOR_STATIC_1(nSamp, step);
				G_INT_64 *pL = new G_INT_64[num_threads](), *pR = new G_INT_64[num_threads](), nLeft = 0, nRigt = 0;
#pragma omp parallel for schedule(static,1)
				for (int th_ = 0; th_ < num_threads; th_++) {
					size_t start = th_*step, end = min(start + step, nSamp), i;
					if (end <= start)		{			continue;		}
					G_INT_64	nL = 0, nR = 0;
					for (i = start; i < end; i++) {
						tpSAMP_ID samp = samps[i];
						_core_isY_(isQuanti, samp, mapFolds, (int)(samp_val[i]), samps + start, nL, rigt + start, nR, 0x0);		continue;
						//_core_isY_(isQuanti, samp, mapFolds, (int)(val[samp]), samps + start, nL, rigt+ start, nR, 0x0);		continue;
					}
					pL[th_] = nL;	 pR[th_] = nR;
					assert(pL[th_] + pR[th_] == end - start);
				}
				for (int th_ = 0; th_ < num_threads; th_++) {
					size_t start = th_*step, end = min(start + step, nSamp);
					if (end <= start) { continue; }
					memcpy(samps + nLeft, samps + start, sizeof(tpSAMP_ID)*pL[th_]);
					nLeft += pL[th_];
				}
				for (int th_ = 0; th_ < num_threads; th_++) {
					size_t start = th_*step, end = min(start + step, nSamp);
					if (end <= start) { continue; }
					memcpy(samps + nLeft + nRigt, rigt + start, sizeof(tpSAMP_ID)*pR[th_]);
					nRigt += pR[th_];
				}
				delete[] pL;		delete[] pR;
				/*for (i = 0; i < nSamp; i++) {
					samp = hBlit->samp_set.samps[i];
					_core_isY_(isQuanti, samp, mapFolds,(int)(val[samp]), samps, nLeft, rigt, nRigt,0x0);		continue;
				}
				//memcpy(samps, left, sizeof(tpSAMP_ID)*nLeft);
				memcpy(samps + nLeft, rigt, sizeof(tpSAMP_ID)*nRigt);*/
				lSet.samps = samps;				lSet.nSamp = nLeft;
				rSet.samps = samps + nLeft;		rSet.nSamp = nRigt;
				//FeatsOnFold::stat.tX += GST_TOC(t1);
			}
			/*else if (hBlit->fruit->split_by==BY_DENSITY) {
				assert(0);
				SAMP_SET &lSet = left->samp_set, &rSet = rigt->samp_set;
				tpSAMP_ID *samps = hBlit->samp_set.samps, *left = hBlit->samp_set.left, *rigt = hBlit->samp_set.rigt;
				lSet = hBlit->samp_set;		rSet = hBlit->samp_set;	//直接复制父节点的一些数据
				lSet.isRef = true;			rSet.isRef = true;
				HistoGRAM *histo = tHisto();	// hDistri->histo;
				for (i = 0; i < nSamp; i++) {
					samp = hBlit->samp_set.samps[i];
					if (isQuanti) {	//训练数据格子化
						pos = val[samp];
					}	else if (isCategory()) {
						pos = this->hDistri->mapCategory[(int)(val[samp])];
						//pos = fold = hBlit->fruit->GetFold(pos);
					}	else {		//predict,test对应的数据集并没有格子化!!!
						pos = histo->AtBin_(val[samp]);		
					}
					const BIN_FEATA& feata = hDistri->binFeatas[pos];
					if(feata.density<T_1)
						left[nLeft++] = samp;
					else
						rigt[nRigt++] = samp;
				}
				memcpy(samps, left, sizeof(tpSAMP_ID)*nLeft);
				memcpy(samps + nLeft, rigt, sizeof(tpSAMP_ID)*nRigt);
				lSet.samps = samps;				lSet.nSamp = nLeft;
				rSet.samps = samps + nLeft;		rSet.nSamp = nRigt;
			}*/
			else/**/ {
				//Value_AtSamp(&hBlit->samp_set, samp_val);
				hBlit->SplitOn(hData_,samp_val, hData_->isQuanti,0x0);
			}
			//放这里不合适，应该在ManifoldTree::Train GrowLeaf之后
			//assert(left->nSample() == hBlit->bst_blit.nLeft && rigt->nSample() == hBlit->bst_blit.nRight);
		}

		

		//参见 FeatVec_Q::Samp2Histo
		virtual void  RefineThrsh(const FeatsOnFold *hData_, const MT_BiSplit *hBlit,int flag = 0x0) {
			/*assert(hBlit->fruit->bin_S0.nz>0);
			double T_0 = hBlit->fruit->bin_S0.split_F, T_1 = hBlit->fruit->bin_S1.split_F;
			FRUIT *fruit = hBlit->fruit;
			tpSAMP_ID *samps = hBlit->samp_set.samps,samp;
			size_t nSamp_0 = hBlit->samp_set.nSamp, i, nSamp=0;
			tpDOWN *hessian = hData_->GetHessian(),*down = hData_->GetDownDirection(); ;
			Tx *val_c = arr(),a=T_0-1,a_1= a;
			size_t minSet = hData_->config.min_data_in_leaf;
			string optimal = hData_->config.leaf_optimal, obj = hData_->config.objective;
			//double sum = samp_set.Y_sum_1, a = a0, errL = 0, g, gL = 0, g1 = 0, lft, rgt;
			double gL = 0, gR0, hL = 0, hR = 0, g, g1 = 0,gSum = 0, hSum = 0;

			vector<tpSAMP_ID> idx;
			for (i = 0; i < nSamp_0; i++) {
				samp = samps[i];
				a = val_c[samp];
				if (a<T_0 || a>T_1)		continue;
				a_1 = max(a_1, a);
				gSum += -down[samp];		hSum += hessian[samp];		nSamp++;
				idx.push_back(samp);
			}
			if(nSamp==0)
				throw "!!!RefineThrsh nSamp==0!!!";
			fruit->adaptive_thrsh = (a_1 + T_1) / 2;
			return;
			if (nSamp < 10)
				return;
			std::sort(idx.begin(), idx.end(), [&val_c](tpSAMP_ID i1, tpSAMP_ID i2) {return val_c[i1] < val_c[i2]; });
			size_t nLeft = 0,nRight = nSamp;		assert(nRight >= 0);
			for (auto samp : idx) {
				//for each(HISTO_BIN item in bins) {
				double gR = gSum - gL, hR = hSum - hL;
				/*if (nLeft<minSet || nRight<minSet) {
					goto LOOP;
				}*//*
				if (hL==0 || hR==0 )		{ goto LOOP; }
				if (val_c[samp] == a )		{	goto LOOP;	}
				a = val_c[samp];
				g = gL*gL / hL + gR*gR / hR;				

				if (g>g1 ) {
					g1 = g;
					//fruit->mxmxN = g1;			fruit->tic_left = item.tic;
					//fruit->nLeft = nLeft;		fruit->nRight = nRight;
					fruit->adaptive_thrsh = a;					
				}
			LOOP:
				gL += -down[samp];		hL += hessian[samp];
				nLeft ++;				nRight --;
			}
			assert(fruit->adaptive_thrsh>= T_0 && fruit->adaptive_thrsh < T_1);*/
			return ;
		}
		
		/*
			v0.3 no edaX 
			v0.4 on samp_set
		*/
		virtual void InitDistri(const FeatsOnFold *hFold,Distribution *tDistri,const SAMP_SET *samp_set, int flag) {
			//assert(hFold!=nullptr);
			size_t i, nSamp_=size();
			assert(distri_ == nullptr);
			if (tDistri == nullptr) {	//Y in train; feats in valid and eavl
				//assert(hFold==nullptr);
				distri_ = new Distribution();
			}	else {
				distri_ = tDistri;
			}

			Tx *samp_val = arr();
			/*if (samp_set != nullptr) {//EDA on replacement sampling
				nSamp_ = samp_set->nSamp;
				samp_val = new Tx[nSamp_];
				tpSAMP_ID *samps = samp_set->samps;
				for (i = 0; i < nSamp_; i++) {
					samp_val[i] = val[samps[i]];
				}
			}
			hDistri->nam = nam;*/
			distri_->EDA(hFold,nSamp_, samp_set, samp_val, tDistri!=nullptr, 0x0);

			//hDistri->STA_at(nSamp_, samp_val, true, 0x0);
			if (ZERO_DEVIA(distri_->vMin, distri_->vMax))
				BIT_SET(this->type, Distribution::V_ZERO_DEVIA);			
			if (samp_val != arr())
				delete[] samp_val;
		}

		//参见Distribution::STA_at，需要独立出来		8/20/2019
		void sorted_idx(vector<tpSAMP_ID>& sortedA,int flag=0x0) {
			sortedA.clear();
			size_t i = 0, i_0 = 0, nA = 0, nNA=0, nZERO=0,N= size();
			for (i = i_0; i < N; i++) {
				if (IS_NAN_INF(val[i])) {
					nNA++;	continue;
				}
				if (IS_ZERO_RANGE(val[i])) {
					nZERO++;
				}/**/
				//a = val[i];
			}

			if (nNA > 0 && nNA < N ) {
				vector<Tx> A;
				vector<tpSAMP_ID> map;
				A.resize(N - nNA);
				map.resize(N - nNA);
				for (i = 0; i < N; i++) {
					if (IS_NAN_INF(val[i])) {
						continue;
					}
					A[nA] = val[i];	map[nA++] = i;
				}
				assert(N - nNA == nA);

				vector<tpSAMP_ID> idx;
				sort_indexes(A, idx);
				sortedA.resize(N - nNA);
				for (i = 0; i < nA; i++) {
					sortedA[i] = map[idx[i]];
				}
				for (i = 0; i < nA - 1; i++) {
					assert(!IS_NAN_INF(val[sortedA[i]]));
					assert(val[sortedA[i]] <= val[sortedA[i + 1]]);
				}
			}
			else if(nNA== 0){
				sort_indexes(N, val, sortedA);
			}
		}

		/*
			isSameSorted 仅对应于isTrain()
			v0.1	cys
				11/4/2019
		*/
		template<typename tpQUANTI>
		void QuantiAtEDA_(ExploreDA *edaX, tpQUANTI *quanti, int nMostBin, const FeatsOnFold *hData_, int flag) {
			bool isSameSorted = hData_->isTrain();
			assert(quanti != nullptr && edaX != nullptr);
			tpQUANTI NNA = tpQUANTI(-1);
			size_t nSamp_ = size(), i, i_0 = 0, i_1, noBin = 0, pos, nzHisto=0, nFailed=0;
			vector<tpSAMP_ID> idx;
			Distribution *myDistri = distri_;
			if (myDistri->sortedA.size() > 0 && isSameSorted) {
				idx = myDistri->sortedA;
				for (i = 0; i < nSamp_; i++)
					quanti[i] = NNA;
			}
			else {
				//sort_indexes(nSamp_,val, idx);
				sorted_idx(idx);
			}
			size_t nA = idx.size();
			Tx a0 = val[idx[0]], a1 = val[idx[nA - 1]];
			double v0 = a0, v1, v2;
			assert(a0 <= a1);
			//const Distribution& distri = edaX->arrDistri[id];
			const Distribution* distri = edaX->GetDistri(id);
			//assert ( distri.histo != nullptr);
			//const vector<double>& vThrsh = distri.vThrsh
				
			if (isCategory()) {
				myDistri->mapCategory = distri->mapCategory;
				if (id == 31) {			//仅用于调试
					//id = 31;
				}				
				i_0 = 0;
				while (i_0 < nA) {
					pos = idx[i_0];			
					int key = (int)(val[pos]);		
					//assert(key >= distri.vMin && key <= distri.vMax);
					MAP_CATEGORY::iterator failed = myDistri->mapCategory.end();
					if (myDistri->mapCategory.find(key) == failed) {
						//hDistri->mapCategory.insert(pair<int, int>(key, hDistri->mapCategory));
						quanti[pos] = distri->histo->nBins - 1;	//很少的fail_match	也会严重降低准确率
						//quanti[pos] = 0;
						nFailed++;
					}else
						quanti[pos] = myDistri->mapCategory[key];
					i_0++;
					//assert(quanti[pos] >= 0 && quanti[pos] < NNA);
				}
				if (nFailed > 0) {		
					if(hData_->config.verbose>0)
						printf("!!!!!! %d   \"%s\" nFailed=%lld nA=%lld !!!!!!\n", id, nam.c_str(), nFailed,nA);
				}
			}
			else {
				const HistoGRAM *histo = distri->histo;		assert(histo!=nullptr);
				noBin = 0;	//v1 = vThrsh[noBin + 1];
				v1 = distri->binFeatas[noBin + 1].split_F;		// bins[noBin + 1].split_F;
				i_0 = 0;
				while (i_0 < nA) {
					pos = idx[i_0];
					assert(!IS_NAN_INF(val[pos]));
					if (val[pos]< distri->vMin || val[pos]>distri->vMax) {//确实有可能
						if (isSameSorted) {
							quanti[pos] = NNA;	i_0++;
						}
						else {	//强制扰动
							if (val[pos] < distri->vMin) {
								quanti[pos] = 0;
							}
							if (val[pos]>distri->vMax) {
								quanti[pos] = histo->nBins -2;
							}
							quanti[pos] = NNA;	i_0++;
						}
						continue;
					}
					if (val[pos] < v1) {
						quanti[pos] = noBin;	i_0++;
					}
					else {
						if (isSameSorted) {
							nzHisto += histo->bins[noBin].nz;
							if (i_0 != nzHisto) {
								;// throw "QuantiAtEDA i_0 != nzHisto!!!";
							}
						}
						noBin++;
						if (noBin >= histo->nBins)						{
							quanti[pos] = histo->nBins - 1;	i_0++;
							continue;
							//throw "QuantiAtEDA noBin is XXX";						
						}
						v1 = noBin+1==histo->nBins ? distri->vMax  : distri->binFeatas[noBin + 1].split_F;
					}
					assert(noBin >= 0 && noBin < NNA);

				}
				//int noNA = distri.histo->bins.size()-1;				
				int noNA = distri->histo->nBins - 1;
				assert(noNA <= NNA);
				HISTO_BIN* hNA=distri->histo->hBinNA();
				for (i = 0; i < nSamp_; i++) {
					//if (quanti[i] == -1)
					if (quanti[i] == NNA )
					{	quanti[i] = noNA;		hNA->nz++;		}
				}
				//hNA->split_F = DBL_MAX;
				return;
			}
		}

		virtual void QuantiAtEDA(ExploreDA *edaX, void *quanti,int sizeofQ, int nMostBin, const FeatsOnFold *hData_, int flag) {
			switch (sizeofQ) {
			case 1:
				QuantiAtEDA_(edaX, (uint8_t*)quanti, nMostBin, hData_, flag);
				break;
			case 2:
				QuantiAtEDA_(edaX, (uint16_t*)quanti, nMostBin, hData_, flag);
				break;
			case 4:
				QuantiAtEDA_(edaX, (uint32_t*)quanti, nMostBin, hData_, flag);
				break;
			default:
				throw "QuantiAtEDA is ...sizeofQ is XXX";
			}
		}


		//manifold node对应的一些statics		
		virtual void BinaryOperate(FeatVector *hY_, BINARY_OPERATE opt, int flag = 0x0) {
			FeatVec_T<Tx> *hY = dynamic_cast<FeatVec_T<Tx> *>(hY_);
			switch (opt) {
			case BINARY_OPERATE::COPY_MEAN:
				//hY->BLIT_mean = BLIT_mean;
				break;
			default:
				break;
			}
		}

		//friend class FeatVec_Q;
	};
	//假设nCls<SHT_MAX
	typedef FeatVec_T<int>	CLASS_VEC;

	class FeatVec_Bundle : public FeatVec_T<short> {
	protected:
		//FeatBlit box;
		vector<int> feat_ids;
		vector<double> vThrsh;
		HistoGRAM *qHisto = nullptr;
	public:
		FeatVec_Bundle(FeatsOnFold *hData_, int id_, const vector<int>&bun, size_t nMostDup, int flag = 0x0);
		virtual ~FeatVec_Bundle() {
			if (qHisto != nullptr)	delete qHisto;
		}
	};



}

