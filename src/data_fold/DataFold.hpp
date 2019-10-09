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

	struct ARR_TREE {
		int nNode = 0;
		double *thrsh_step = nullptr,weight=1.0;
		int *feat_ids = nullptr, *left = nullptr, *rigt = nullptr, *info = nullptr;

		virtual void Init(int nNode, int flag = 0x0) {
			thrsh_step = new double[nNode];
			feat_ids = new int[nNode * 4];
			left = feat_ids + nNode;		rigt = left + nNode;
			info = rigt + nNode;
		}
		~ARR_TREE() {
			if (thrsh_step != nullptr)			delete[] thrsh_step;
			if (feat_ids != nullptr)			delete[] feat_ids;
		}
	};

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

		virtual void Init(FeatsOnFold *Data_, int flag = 0x0);
		virtual void ToDownStep(int flag = 0x0);
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

		struct BUFFER {
			tpSAMP_ID *samp_root_set = nullptr,*samp_left = nullptr,*samp_rigt = nullptr;
			
			virtual void Init(size_t nSamp, int flag = 0x0) {
				samp_root_set = new tpSAMP_ID[nSamp];
				samp_left = new tpSAMP_ID[nSamp];
				samp_rigt = new tpSAMP_ID[nSamp];
			}
			virtual ~BUFFER() {
				if (samp_root_set != nullptr)		delete[] samp_root_set;
				if (samp_left != nullptr)			delete[] samp_left;
				if (samp_rigt != nullptr)			delete[] samp_rigt;
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

		enum {
			TAG_ZERO = 0x10000,
			DF_FLAG = 0xF00000, DF_TRAIN = 0x100000, DF_EVAL = 0x200000, DF_PREDIC = 0x400000,
			//FAST_QUANTI = 0x1000000,	//为了加速而采用量子化
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

		virtual ~FeatsOnFold() {
			for (auto hFeat : feats)
			//for each(FeatVector *hFeat in feats)
				delete hFeat;
			feats.clear();					//fold.clear();
			delete[] distri;
			if (importance != nullptr)			delete importance;
			if (hMove != nullptr)				delete hMove;
			//delete[] resi;			delete[] move;
		}
		int *Tag();
		virtual void Empty(int flag) {

		}
		virtual void Compress(int flag = 0x0);
		virtual void ExpandFeat(int flag = 0x0);
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
			if (no < -1 || no >= nFeat())	throw "Feat no is OUT OF RANGE!!!";
			return feats[no];
		}
		virtual int *Rank4Feat(int type, int flag = 0x0);	
		virtual void nPick4Split(vector<int>&picks, GRander&rander, BoostingForest *hForest,int flag = 0x0);

		//核心函数 
		virtual void SplitOn(MT_BiSplit *hBlit, int flag = 0x0) {
			FeatVector *hF_ = Feat(hBlit->feat_id);
			hF_->SplitOn(this, hBlit);

		}

		template<typename Ty>
		bool DeltastepOnTree(const ARR_TREE&tree, int flag) {
			size_t nSamp = nSample(), nNode = tree.nNode, no, nFeat = feats.size(), step = nSamp;
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
							no = hFT->left_rigt(t, thrsh_step[no], left[no], rigt[no]);
						}
					}
				}
			}
			return true;
		}

		//int  OMP_FOR_STATIC_1(const size_t nSamp, size_t& step, int flag = 0x0);
		template<typename Ty>
		bool PredictOnTree(const ARR_TREE&tree, int flag) {
			if (tree.weight == 0) {
				return true;	//pass
			}
			size_t nSamp = nSample(), nNode = tree.nNode, no, nFeat = feats.size(), step = nSamp;
			G_INT_64 t;
			double *thrsh_step = tree.thrsh_step;
			FeatVec_T<Ty> *predict = dynamic_cast<FeatVec_T<Ty>*>(GetPrecict());
			if (predict == nullptr)
				return false;
			tpDOWN *delta_step = GetDeltaStep();			//assert(delta_step != nullptr);
			Ty *pred = predict->arr();
			int *feat_ids = tree.feat_ids, *left = tree.left, *rigt = tree.rigt;
			
			/**/
			int num_threads = OMP_FOR_STATIC_1(nSamp, step);
#pragma omp parallel for schedule(static,1)
			for (int thread = 0; thread < num_threads; thread++) {
				size_t start = thread*step, end = min(start + step, nSamp), t;
				for (t = start; t < end; t++) {
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
							no = hFT->left_rigt(t,thrsh_step[no], left[no], rigt[no]);							
						}
					}
				}

			}
			
			return true;
		}

		/*
			尽量优化
		*/
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
			/**/
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
		}

		template<typename Tx, typename Ty>
		bool UpdateStepOnReduce(ARR_TREE*treeEval, ARR_TREE*tree4train, int flag=0x0) {
			ARR_TREE& tree = *treeEval;
			size_t nSamp = nSample(), nNode = tree.nNode, no, nFeat = feats.size(), step = nSamp;
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
		size_t nSamp_=0;
		Tx *val=nullptr;
		//map<hMTNode,Tx> BLIT_thrsh,BLIT_mean;
	public:
		FeatVec_T() { 	}
		FeatVec_T(size_t _len, int id_, const string&des_, int flag = 0x0) {
			id = id_;
			nSamp_ = _len;
			desc = des_;	assert(_len > 0);	
			type = flag;
			if (isReferVal()) {

			}	else
				val = new Tx[_len];	// val.resize(_len);
		}

		virtual size_t nSamp() {	return nSamp_;		}

		virtual ~FeatVec_T() {
			if (isReferVal()) {
				;// printf("");
			}	else if (val != nullptr)
				delete[] val;
			//val.clear();
		}

		//{	assert(_len>0);	val.resize(_len,(Tx)0);	 }

		//Tx* arr() { return VECTOR2ARR(val); }
		Tx* arr()			{ return val;		}
		size_t size()		{ return nSamp_;	}

		virtual void Set(size_t len, PY_COLUMN *col, int flag = 0x0) { 
			if (len != nSamp_)
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
			if (len != nSamp_)
				throw "FeatVec_T::Set len mismatch!!!";
			void* dst = arr();
			if (isReferVal()) {
				val = (Tx*)src;
			}
			else
				memcpy(dst, src, sizeof(Tx)*len);			
		}

		virtual void Set(double a, int flag = 0x0) {
			size_t len = nSamp_, i;
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
			//memset(arr(), 0, val.size() * sizeof(Tx));
			memset(val, 0, nSamp_ * sizeof(Tx));
		}

		virtual void STA_at(SAMP_SET& some_set, int flag = 0x0) {
			size_t nS = some_set.nSamp, i, pos;
			Tx a2, a_sum, a_0, a_1, *val_0 = arr();
			some_set.STA_at_<Tx>(val_0, a2, a_sum, a_0, a_1, false);			
		}

		//参见MT_BiSplit::Observation_AtSamp(FeatsOnFold *hData_, int flag) 
		virtual void Observation_AtSamp(LiteBOM_Config config, SAMP_SET& some_set, Distribution&distri, int flag=0x0) {
			assert(0);
			size_t nS = some_set.nSamp,i,pos;
			Tx a2, a_sum,a_0, a_1,*val_s=new Tx[nS],*val_0=arr();
			some_set.STA_at_<Tx>(val_0, a2, a_sum, a_0, a_1, false);
			distri.vMin = a_0,				distri.vMax = a_1;
			for (i = 0; i < nS;i++) {
				pos = some_set.samps[i];
				val_s[i] = val_0[pos];
			}
			distri.X2Histo_<Tx,Tx>(config, nS, val_s,nullptr);		
			delete[] val_s;
		}

		virtual void loc(vector<tpSAMP_ID>&poss,double target,int flag=0x0) {
			poss.clear();
			size_t i;
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
			HistoGRAM *histo = hBlit->fruit->histo;
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
			//Tx *feat = arrFeat[feat_ids[no]];
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

		virtual void SplitOn(FeatsOnFold *hData_, MT_BiSplit *hBlit, int flag = 0x0) {
			//assert (BLIT_thrsh.find(hBlit) != BLIT_thrsh.end());
			tpSAMP_ID samp;
			bool isQuanti = hData_->isQuanti;		//predict,test对应的数据集并没有格子化!!!
			//double thrsh = hData_->isQuanti ? hBlit->fruit->T_quanti : hBlit->fruit->thrshold;
			//double thrsh = hBlit->fruit->Thrshold(hData_->isQuanti);
			//if( hBlit->fruit->isY )
			//	thrsh = hBlit->fruit->thrshold;
			MT_BiSplit *left = hBlit->left, *rigt = hBlit->right, *child = nullptr;
			int pos;
			size_t nSamp = hBlit->nSample(), i, nLeft = 0, nRigt = 0;
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
				SAMP_SET &lSet = left->samp_set, &rSet = rigt->samp_set;
				tpSAMP_ID *samps = hBlit->samp_set.samps, *left = hBlit->samp_set.left, *rigt = hBlit->samp_set.rigt;
				lSet = hBlit->samp_set;		rSet = hBlit->samp_set;	//直接复制父节点的一些数据
				lSet.isRef = true;			rSet.isRef = true;
				HistoGRAM *histo = hBlit->fruit->histo;
				for (i = 0; i < nSamp; i++) {
					samp = hBlit->samp_set.samps[i];
					//int pos = hData_->isQuanti ? val[samp] : histo->AtBin_(val[samp]);
					if (isQuanti) {	//训练数据格子化
						pos = val[samp];
					}
					else if (isCategory()) {	
						pos = hBlit->fruit->mapCategory[(int)(val[samp])];
					}else{		//predict,test对应的数据集并没有格子化!!!
						pos = histo->AtBin_(val[samp]);		
					}
					int fold = pos < 0 ? 0 : histo->bins[pos].fold;
					if (fold <= 0)
						left[nLeft++] = samp;
					else
						rigt[nRigt++] = samp;
				}
				memcpy(samps, left, sizeof(tpSAMP_ID)*nLeft);
				memcpy(samps + nLeft, rigt, sizeof(tpSAMP_ID)*nRigt);
				lSet.samps = samps;				lSet.nSamp = nLeft;
				rSet.samps = samps + nLeft;		rSet.nSamp = nRigt;
				//hBlit->samp_set.SplitOn(ys, thrsh, left->samp_set, rigt->samp_set,1);
			}
			else if (hBlit->fruit->split_by==BY_DENSITY) {
				SAMP_SET &lSet = left->samp_set, &rSet = rigt->samp_set;
				tpSAMP_ID *samps = hBlit->samp_set.samps, *left = hBlit->samp_set.left, *rigt = hBlit->samp_set.rigt;
				lSet = hBlit->samp_set;		rSet = hBlit->samp_set;	//直接复制父节点的一些数据
				lSet.isRef = true;			rSet.isRef = true;
				HistoGRAM *histo = hDistri->histo;
				for (i = 0; i < nSamp; i++) {
					samp = hBlit->samp_set.samps[i];
					if (isQuanti) {	//训练数据格子化
						pos = val[samp];
					}	else if (isCategory()) {
						pos = hBlit->fruit->mapCategory[(int)(val[samp])];
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
			}
			else/**/ {
				hBlit->SplitOn(hData_,nSamp_,val, hData_->isQuanti);

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
		*/
		virtual void EDA(const LiteBOM_Config&config,bool genHisto, int flag) {
			size_t i;
			if (hDistri == nullptr) {	//only for Y
				hDistri = new Distribution();		
			}	else {

			}
			hDistri->nam = nam;
			hDistri->STA_at(nSamp_, val, true, 0x0);
			if (ZERO_DEVIA(hDistri->vMin, hDistri->vMax))
				BIT_SET(this->type, Distribution::V_ZERO_DEVIA);
			else if (config.eda_Normal != LiteBOM_Config::NORMAL_off) {
				Tx *val_c = arr();
				double mean = hDistri->mean, s = 1.0 / hDistri->devia;
				//for each(tpSAMP_ID samp in hDistri->sortedA
				for (i = 0; i < nSamp_; i++) {
					val_c[i] = (val_c[i] - mean)*s;
				}
				hDistri->STA_at(nSamp_, val, true, 0x0);/**/
			}
			if (genHisto) {
				if (hDistri->histo == nullptr) {	//参见LiteMORT_EDA->Analysis(config, (float *)dataX, (tpY *)dataY, nSamp_, nFeat_0, 1, flag);
					hDistri->X2Histo_(config, nSamp_, arr(), (double*)nullptr);
					//hDistri->Dump(this->id, false, flag);
				}			
			}
		}

		//参见Distribution::STA_at，需要独立出来		8/20/2019
		void sorted_idx(vector<tpSAMP_ID>& sortedA,int flag=0x0) {
			sortedA.clear();
			size_t i = 0, i_0 = 0, nA = 0, nNA=0, nZERO=0,N= nSamp_;
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
			参见ExploreDA::
		*/
		virtual void QuantiAtEDA(const ExploreDA *edaX, tpQUANTI *quanti, int nMostBin,bool isSameSorted, int flag) {
			assert(quanti != nullptr && edaX != nullptr);
			size_t nSamp_ = size(), i, i_0 = 0, i_1, noBin = 0, pos, nzHisto=0;
			vector<tpSAMP_ID> idx;
			if (hDistri->sortedA.size() > 0 && isSameSorted) {
				idx = hDistri->sortedA;
				for (i = 0; i < nSamp_; i++)
					quanti[i] = -1;
			}
			else {
				//sort_indexes(nSamp_,val, idx);
				sorted_idx(idx);
			}
			size_t nA = idx.size();
			Tx a0 = val[idx[0]], a1 = val[idx[nA - 1]];
			double v0 = a0, v1, v2;
			assert(a0 <= a1);
			const Distribution& distri = edaX->arrDistri[id];
			//assert ( distri.histo != nullptr);
			//const vector<double>& vThrsh = distri.vThrsh
				
			if (isCategory()) {
				MAP_CATEGORY mapCategory = distri.mapCategory;
				hDistri->mapCategory = mapCategory;
				i_0 = 0;
				while (i_0 < nA) {
					pos = idx[i_0];
					int key = (int)(val[pos]);
					assert(key >= distri.vMin && key <= distri.vMax);
					quanti[pos] = mapCategory[key];	i_0++;
				}
			}
			else {
				const HistoGRAM *histo = distri.histo;		assert(histo!=nullptr);
				noBin = 0;	//v1 = vThrsh[noBin + 1];
				v1 = distri.binFeatas[noBin + 1].split_F;		// bins[noBin + 1].split_F;
				i_0 = 0;
				while (i_0 < nA) {
					pos = idx[i_0];
					assert(!IS_NAN_INF(val[pos]));
					if (val[pos]< distri.vMin || val[pos]>distri.vMax) {//确实有可能
						if (isSameSorted) {
							quanti[pos] = -1;	i_0++;
						}
						else {	//强制扰动
							if (val[pos] < distri.vMin) {
								quanti[pos] = 0;
							}
							if (val[pos]>distri.vMax) {
								quanti[pos] = histo->nBins -2;
							}
							quanti[pos] = -1;	i_0++;
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
						if (noBin >= histo->nBins)
						{		throw "QuantiAtEDA noBin is XXX";						}
						v1 = distri.binFeatas[noBin + 1].split_F;// histo->bins[noBin + 1].split_F;
					}
				}
				//int noNA = distri.histo->bins.size()-1;				
				int noNA = distri.histo->nBins - 1;
				HISTO_BIN* hNA=distri.histo->hBinNA();
				for (i = 0; i < nSamp_; i++) {
					//if (quanti[i] == -1)
					if (quanti[i] <0)
					{	quanti[i] = noNA;		hNA->nz++;		}
				}
				//hNA->split_F = DBL_MAX;
				return;
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

		friend class FeatVec_Q;

	};
	//假设nCls<SHT_MAX
	typedef FeatVec_T<int>	CLASS_VEC;

	/*
		quantization
	*/
	class FeatVec_Q : public FeatVec_T<tpQUANTI> {
	protected:
		FeatVector *hFeatSource = nullptr;
		//FeatBlit box;
		HistoGRAM *qHisto_0 = nullptr;
		//由PerturbeHisto生成，需要重新设计
		HistoGRAM *qHisto_1 = nullptr;		
	public:
		FeatVec_Q(const FeatsOnFold *hData_, FeatVector *hFeat, int nMostBin, int flag = 0x0);
		virtual ~FeatVec_Q() {
			if (qHisto_0 != nullptr)			delete qHisto_0;
			if (qHisto_1 != nullptr)			delete qHisto_1;
			if (hFeatSource != nullptr) {
				delete hFeatSource;
				hDistri = nullptr;
			}
		}
		HistoGRAM *GetHisto(int flag=0x0)		{	return qHisto_0;	}
		void InitSampHisto(HistoGRAM* histo, bool isRandom, int flag = 0x0);

		virtual void Observation_AtSamp(LiteBOM_Config config, SAMP_SET& samp, Distribution&distri, int flag = 0x0) { 
			hFeatSource->Observation_AtSamp(config, samp, distri,flag);
		}

		virtual tpQUANTI *GetQuantiBins(int flag = 0x0) { 
			return val; 
		}

		//static bin mapping	生成基于EDA的格子	参见Samp2Histo
		virtual void PerturbeHisto(const FeatsOnFold *hData_, int flag = 0x0);
		virtual void UpdateHisto(const FeatsOnFold *hData_, bool isOnY, bool isFirst, int flag = 0x0);
		//根据样本集，修正每个格子的内容(Y_sum,nz...)
		virtual void Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* hParent, HistoGRAM* histo, int nMostBin, int flag = 0x0);
		virtual void Samp2Histo_null_hessian(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, int flag = 0x0);
		virtual void Samp2Histo_null_hessian_sparse(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, int flag = 0x0);

		virtual void UpdateFruit(const FeatsOnFold *hData_, MT_BiSplit *hBlit, int flag = 0x0) {
			//double split = hBlit->fruit->thrshold;
			if (hBlit->fruit->isY) {
				//vector<HISTO_BIN>& bins=hBlit->fruit->histo->bins;
				if (this->isCategory())
					hBlit->fruit->mapCategory = this->hDistri->mapCategory;
				else {
					/*assert(bins.size() == vThrsh.size());
					for (size_t i = 0; i < vThrsh.size(); i++) {
						bins[i].tic = vThrsh[i];
					}*/
				}
				//hBlit->fruit->T_quanti = -13;
			}
			else {
				/*tpQUANTI q_split = split;		assert(q_split == split);
				hBlit->fruit->T_quanti = q_split;
				//assert(split>a0 && split <= a1);
				float thrsh = vThrsh[q_split];		//严重的BUG之源啊
				hBlit->fruit->thrshold = thrsh;*/
				if(hData_->config.split_refine!= LiteBOM_Config::REFINE_SPLIT::REFINE_NONE)
					hFeatSource->RefineThrsh(hData_, hBlit);
			}
		}

		friend class FeatVec_Bundle;
	};

	class FeatVec_Bundle : public FeatVec_T<tpQUANTI> {
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

		virtual void UpdateFruit(MT_BiSplit *hBlit, int flag = 0x0);

		virtual void Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* hParent, HistoGRAM* histo, int nMostBin, int flag = 0x0);
	};


	

	class FeatVec_LOSS;

	
	



}

