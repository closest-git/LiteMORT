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
#include "../util/BLAS_t.hpp"
#include "../util/GRander.hpp"
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
	template<typename Tx> class FeatVec_T;

	struct ARR_TREE {
		int nNode = 0;
		double *thrsh_step = nullptr;
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
		Feat_Importance *importance = nullptr;
		Move_Accelerator *hMove = nullptr;
		GRander rander_samp, rander_feat;
		INIT_SCORE init_score;
		vector<FeatVector*> feats;				//featrue values of X
		int nPickFeat = 0;

		struct STAT {
			double dY;
			int nCheckGain = 0;

			double tX = 0;	//统计时间
		};
		static STAT stat;
		ExploreDA *edaX = nullptr;
		LiteBOM_Config config;			//configuration paramters
										//tree每个node都有samp_set，会影响data
										//SAMP_SET samp_set;
										//label or target values of Y，Only one,多个Y显然需要多个模型来训练。	
		FeatVec_LOSS *lossy = nullptr;

		FeatVector* GetPrecict();
		FeatVector* GetY();

		//pDown=target-predict
		tpDOWN *GetDownDirection()	const;
		tpDOWN *GetHessian() const;
		tpDOWN *GetSampleDown()	const;
		tpDOWN *GetSampleHessian() const;		
		
		bool isTrain() {
			return config.task == LiteBOM_Config::kTrain;
		}
		bool isPredict() {
			return config.task == LiteBOM_Config::kPredict;
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
			if (importance != nullptr)		delete importance;
			if (hMove != nullptr)			delete hMove;
			//delete[] resi;			delete[] move;
		}
		int *Tag();
		virtual void Empty(int flag) {

		}

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

		virtual size_t nFeat()	const { return feats.size(); }
		FeatVector *Feat(int no) {
			if (no == -1)
				//return lossy.predict;
				return GetPrecict();
			if (no < -1 || no >= nFeat())	throw "Feat no is OUT OF RANGE!!!";
			return feats[no];
		}
		virtual void nPick4Split(vector<int>&picks, GRander&rander, int flag = 0x0);

		//核心函数 
		virtual void SplitOn(MT_BiSplit *hBlit, int flag = 0x0) {
			FeatVector *hF_ = Feat(hBlit->feat_id);
			hF_->SplitOn(this, hBlit);
			if (isTrain())
				;// hBlit->Split_BsFold(this);

		}

		//int  OMP_FOR_STATIC_1(const size_t nSamp, size_t& step, int flag = 0x0);
		/*
			尽量优化
		*/
		template<typename Tx, typename Ty>
		bool PredictOnTree(const ARR_TREE&tree, int flag) {
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
				FeatVec_T<Tx>*hFT = dynamic_cast<FeatVec_T<Tx>*>(feats[t]);
				if (hFT == nullptr)
				{
					delete[] arrFeat;	return false;
				}
				arrFeat[t] = hFT->arr();
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
							pred[t] += thrsh_step[no];		break;
						}
						else {
							assert(rigt[no] != -1);
							Tx *feat = arrFeat[feat_ids[no]];
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
			delete[] arrFeat;
			return true;
		}

		/*
		Train(Observation_AtSamp) predict(Update_step)
		*/
		virtual void AtLeaf(MT_BiSplit *hBlit, int flag = 0x0) {
			//FeatVector *hF_ = Feat(hBlit->feat_id);
			assert(hBlit->isLeaf());
			if (isTrain()) {
				hBlit->Observation_AtLocalSamp(this);
				hBlit->Init_BFold(this);
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
		std::vector<Tx> val;
		//map<hMTNode,Tx> BLIT_thrsh,BLIT_mean;
	public:
		FeatVec_T() { 	}
		FeatVec_T(size_t _len, int id_, const string&des_, int flag = 0x0) {
			id = id_;
			desc = des_;	assert(_len > 0);		val.resize(_len);
			//hDistri = new Distribution();
		}

		virtual ~FeatVec_T() {
			val.clear();
		}


		//{	assert(_len>0);	val.resize(_len,(Tx)0);	 }

		Tx* arr() { return VECTOR2ARR(val); }

		virtual void Set(size_t len, void* src, int flag = 0x0) {
			assert(len == val.size());
			void* dst = arr();
			memcpy(dst, src, sizeof(Tx)*len);			
		}
		virtual void Set(double a, int flag = 0x0) {
			size_t len = val.size(), i;
			Tx *x_ = arr();
			for (i = 0; i < len; i++)
				x_[i] = a;
		}
		virtual void Set(size_t pos, double a, int flag = 0x0) {
			val[pos] = (Tx)a;
		}
		virtual void Clear(int flag = 0x0) {
			val.clear();
		}
		virtual void Empty(int flag = 0x0) {
			memset(arr(), 0, val.size() * sizeof(Tx));
		}

		//参见MT_BiSplit::Observation_AtSamp(FeatsOnFold *hData_, int flag) 
		virtual void Observation_AtSamp(LiteBOM_Config config, SAMP_SET& some_set, Distribution&distri, int flag=0x0) {
			assert(0);
			size_t nS = some_set.nSamp,i,pos;
			Tx a2, a_sum,a_0, a_1,*val_s=new Tx[nS],*val_0=arr();
			some_set.STA_at<Tx>(val_0, a2, a_sum, a_0, a_1, false);
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
			size_t i,dim = val.size();
			for (i = 0; i < dim; i++) {
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
			assert(hData_->isPredict());
			double sDrop = hData_->config.drop_out;
			//double shrink = hData_->config.learning_rate;
			tpDOWN step_base = hBlit->GetDownStep();
			if (isAdaptive) {
				assert(hBlit->lr_eta ==1.0);
				double s[] = { 0.1,0.5,1,2,5,10 };
				double err, min_err=DBL_MAX,a, eta_bst=1.0;
				FeatVector* hY0=hData_->GetY();
				FeatVec_T<Tx> *hY = dynamic_cast<FeatVec_T<Tx>*>(hY0);		assert(hY!=nullptr);
				const Tx* target = VECTOR2ARR(hY->val);
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
			assert(hData_->isPredict());
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


		virtual void SplitOn(FeatsOnFold *hData_, MT_BiSplit *hBlit, int flag = 0x0) {
			//assert (BLIT_thrsh.find(hBlit) != BLIT_thrsh.end());
			tpSAMP_ID samp;
			bool isQuanti = hData_->isQuanti;		//predict,test对应的数据集并没有格子化!!!
			//double thrsh = hData_->isQuanti ? hBlit->fruit->T_quanti : hBlit->fruit->thrshold;
			double thrsh = hBlit->fruit->Thrshold(hData_->isQuanti);
			//if( hBlit->fruit->isY )
			//	thrsh = hBlit->fruit->thrshold;
			MT_BiSplit *left = hBlit->left, *rigt = hBlit->right, *child = nullptr;
			int pos;
			size_t nSamp = hBlit->nSample(), i, nLeft = 0, nRigt = 0;
			if (hData_->isTrain())
				assert(left != nullptr && rigt != nullptr);
			else
				assert(left != nullptr && rigt != nullptr);
			assert(left->nSample() == 0 && rigt->nSample() == 0);
			/*
				还是有问题	12/19/2018	cys
				1/11/2019	发现train的histo->AtBin_不适用于test
			*/
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
						pos = histo->AtBin_(val[samp]);		//有BUG!!!
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
			else/**/ {
				hBlit->SplitOn(hData_,val, hData_->isQuanti);

			}
			//放这里不合适，应该在ManifoldTree::Train GrowLeaf之后
			//assert(left->nSample() == hBlit->bst_blit.nLeft && rigt->nSample() == hBlit->bst_blit.nRight);
		}

		//参见 FeatVec_Q::Samp2Histo
		virtual void  RefineThrsh(const FeatsOnFold *hData_, const MT_BiSplit *hBlit,int flag = 0x0) {
			assert(hBlit->fruit->bin_S0.nz>0);
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
				}*/
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
			assert(fruit->adaptive_thrsh>= T_0 && fruit->adaptive_thrsh < T_1);
			return ;
		}
		
		virtual void Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, int flag = 0x0) {
			throw "FeatVec_T<Tx>::Samp2Histo is deprecated!!!";
			size_t nSamp = samp_set.nSamp, i;			
			tpDOWN *err = nullptr;
			string optimal = hData_->config.leaf_optimal;
			bool isLambda = optimal == "lambda_0";
			double a1 = -DBL_MAX, a0 = DBL_MAX;
			const tpSAMP_ID *samps = samp_set.samps;
			tpSAMP_ID samp;
			double a, sum = 0;
			Tx *val_c = arr();
			for (i = 0; i < nSamp; i++) {
				samp = samps[i];
				a = val_c[samp];
				if (a0 > a) {
					a0 = a;		//ox.pos_0 = samp;
				}
				if (a1 < a) {
					a1 = a;		//box.pos_1 = samp;
				}
			}
			histo->OptimalBins(nMostBin, nSamp, a0, a1);

			int no = 0, nBin = histo->bins.size();
			if (nBin == 0)
				return;
			assert(nBin >= 2);
			double step = (a1 - a0) / (nBin - 1), r;
			for (i = 0; i < nBin; i++) {
				HISTO_BIN& bin = histo->bins[i];
				//histo->bins[i].tic = a0+(i+0.5)*step;		//每个bin的平均位置，取中间值
				histo->bins[i].tic = a0 + i*step;		//每个bin的最左边位置
			}

			for (i = 0; i < nSamp; i++) {
				samp = samps[i];
				r = (val_c[samp] - a0) / step;
				//no = MIN((int)(r), nBin - 1);				assert(no < nBin);
				no = (int)r;				assert(no < nBin);
				HISTO_BIN& bin = histo->bins[no];
				/*if (isTaylor2) {
					//http://bazyd.com/machine-learning-understanding-the-principles-of-gbdt-20171001/
					bin.H_sum += 2;		bin.G_sum += -2 * err[samp];
				}				else*/ {
					bin.G_sum += -err[samp];			bin.H_sum += 1;
				}
				bin.nz++;
			}
			if (histo->quanti != nullptr) {
				for (i = 0; i < nSamp; i++) {
					samp = samps[i];
					r = (val_c[samp] - a0) / step;
					no = MIN((int)(r), nBin - 1);				assert(no < nBin);
					histo->quanti[i] = no;
				}
			}
		}

		/*
			和edaX有区别（数据集有差别），这些区别应该有价值
		*/
		virtual void EDA(const LiteBOM_Config&config, ExploreDA *edaX, int flag) {
			size_t nSamp_ = val.size(), i;
			assert(hDistri == nullptr);
			hDistri = new Distribution();
			hDistri->STA_at(val, true, 0x0);
			if (ZERO_DEVIA(hDistri->vMin, hDistri->vMax))
				BIT_SET(this->type, Distribution::V_ZERO_DEVIA);
			else if (config.eda_Normal != LiteBOM_Config::NORMAL_off) {
				Tx *val_c = arr();
				double mean = hDistri->mean, s = 1.0 / hDistri->devia;
				//for each(tpSAMP_ID samp in hDistri->sortedA
				for (i = 0; i < nSamp_; i++) {
					val_c[i] = (val_c[i] - mean)*s;
				}
				hDistri->STA_at(val, true, 0x0);/**/
			}
			if (edaX != nullptr) {/*复制信息*/
				const Distribution& distri = edaX->arrDistri[id];
				if (distri.vMin != DBL_MAX) {
					/*assert(hDistri->vMin >= distri.vMin && hDistri->vMax <= distri.vMax);
					if (hDistri->vMin > distri.vMin)
						hDistri->vMin = distri.vMin;
					if (hDistri->vMax < distri.vMax)
						hDistri->vMax = distri.vMax;*/
				}
				nam = distri.nam;
				BIT_SET(type, distri.type);
			}
		}

		/*
			参见ExploreDA::
		*/
		virtual void QuantiAtEDA(const ExploreDA *edaX, tpQUANTI *quanti, int nMostBin, int flag = 0x0) {
			assert(quanti != nullptr && edaX != nullptr);
			size_t nSamp_ = val.size(), i, i_0 = 0, i_1, noBin = 0, pos;
			vector<tpSAMP_ID> idx;
			if (hDistri->sortedA.size() > 0) {
				idx = hDistri->sortedA;
				for (i = 0; i < nSamp_; i++)
					quanti[i] = -1;
			}
			else
				sort_indexes(val, idx);
			//sort_indexes(val, idx);
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
				v1 = histo->bins[noBin+1].split_F;
				i_0 = 0;
				while (i_0 < nA) {
					pos = idx[i_0];
					assert(!IS_NAN_INF(val[pos]));
					if (val[pos]< distri.vMin || val[pos]>distri.vMax) {//确实有可能
						quanti[pos] = -1;	i_0++;
						continue;
					}
					if (val[pos] < v1) {
						quanti[pos] = noBin;	i_0++;
					}
					else {
						noBin++;
						if (noBin >= histo->bins.size())
							throw "QuantiAtEDA noBin is XXX";
						//v1 = vThrsh[noBin + 1];
						v1 = histo->bins[noBin+1].split_F;
					}
				}
				int noNA = distri.histo->bins.size()-1;				
				HISTO_BIN* hNA=distri.histo->hBinNA();
				for (i = 0; i < nSamp_; i++) {
					if (quanti[i] == -1)
					{	quanti[i] = noNA;		hNA->nz++;		}
				}
				hNA->split_F = DBL_MAX;
				return;
			}
		}

		/*
			v0.2	 考虑有NAN的情况，参见QuantiAtEDA
		*/
		virtual void Split2Quanti_000(const LiteBOM_Config&config, const ExploreDA *eda, vector<double>& vThrsh, HistoGRAM *qHisto, tpDOWN *yDown, int nMostBin, int flag = 0x0) {
			assert(qHisto->quanti != nullptr);
			tpQUANTI *quanti = qHisto->quanti;
			size_t nSamp_ = val.size(), i, i_0 = 0, i_1, noBin = 0, min_step = MAX(nSamp_ / nMostBin, 50), nExSplit = 0;
			vector<tpSAMP_ID> idx;
			if (hDistri->sortedA.size() > 0)
				idx = hDistri->sortedA;
			else
				sort_indexes(val, idx);
			size_t nA = idx.size();
			Tx a0 = val[idx[0]], a1 = val[idx[nA - 1]], v0 = a0, v1;
			assert(a0 <= a1);

			//qHisto->a0 = a0;		qHisto->a1 = a1;
			if (a0 == a1) {
				//printf(" %s is const(%g)!!!", desc.c_str(), a0);
				return;
			}
			if (this->id == 16)
				id = 16;
			Tx step = (a1 - a0) / nMostBin, v1_last = a0;
			qHisto->bins.resize(nMostBin * 2 + 3);
			int histo_alg = 2;
			switch (histo_alg) {
			case 1:
				assert(0);	//HistoOnFrequncy(config, VECTOR2ARR(val), idx, vThrsh, nMostBin);
				break;
			case 2:
				while (i_0 < nA) {
					v0 = val[idx[i_0]];		v1 = v0 + step;	i_1 = i_0;
					noBin = vThrsh.size();
					HISTO_BIN& bin = qHisto->bins[noBin];
					bin.tic = noBin;
					//vThrsh.push_back((v0 + v1_last) / 2);
					vThrsh.push_back(v0);
					//while (i_1 < nA - 1 && val[idx[++i_1]] < v1){
					while (i_1 < nA && val[idx[i_1]] < v1) {
						assert(yDown == nullptr);
						/*if (yDown != nullptr) {		//how dividing the bins on the gradient statistics
							double Y1= yDown[idx[i_1]],Y2= Y1*Y1, impuri;
							tpDOWN y= yDown[idx[i_1 - 1]];
							Y2 +=y*y;		Y1+=y;
							impuri= Y2-	Y1*Y1/(i_1- i_0);
							if(impuri>Y2 && i_1 - i_0>min_step)	{
								v2 = val[idx[i_1-1]];
								if( v2>v0 )		//否则无法设置split
								{	nExSplit++;	break;		}
							}
						}*/
						i_1++;
					}
					v1_last = val[idx[i_1 - 1]];
					assert(i_1 == nA || v1_last < v1);
					for (i = i_0; i < i_1; i++)	quanti[idx[i]] = noBin;
					bin.nz = i_1 - i_0;
					i_0 = i_1;
				}
				assert(i_0 == nA);
				//assert(vThrsh.size() < nMostBin);
				if (vThrsh.size() > nMostBin) {
					printf("\n!!!FEAT_%d nBin=%ld > nMostBin!!!\n", id, vThrsh.size());
				}
				if (nExSplit > 0)	printf("\tnExSplit=%ld", nExSplit);
				qHisto->bins.resize(vThrsh.size());
				vThrsh.push_back(a1 + step / 10);
				break;
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
		HistoGRAM *qHisto = nullptr;
	public:
		/*union {
			vector<double> vThrsh;
			//vector<int> mapCategory;
		};*/
		FeatVec_Q(const FeatsOnFold *hData_, FeatVector *hFeat, int nMostBin, int flag = 0x0);
		virtual ~FeatVec_Q() {
			if (qHisto != nullptr)			delete qHisto;
			if (hFeatSource != nullptr) {
				delete hFeatSource;
				hDistri = nullptr;
			}
		}
		HistoGRAM *GetHisto() { return qHisto; }
		virtual void Observation_AtSamp(LiteBOM_Config config, SAMP_SET& samp, Distribution&distri, int flag = 0x0) { 
			hFeatSource->Observation_AtSamp(config, samp, distri,flag);
		}


		//static bin mapping	生成基于EDA的格子	参见Samp2Histo
		virtual void UpdateHisto(const FeatsOnFold *hData_, bool isOnY, bool isFirst, int flag = 0x0);
		//根据样本集，修正每个格子的内容(Y_sum,nz...)
		virtual void Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, int flag = 0x0);

		virtual void UpdateFruit(const FeatsOnFold *hData_, MT_BiSplit *hBlit, int flag = 0x0) {
			//double split = hBlit->fruit->thrshold;
			if (hBlit->fruit->isY) {
				vector<HISTO_BIN>& bins=hBlit->fruit->histo->bins;
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

		virtual void Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, int flag = 0x0);
	};


	

	class FeatVec_LOSS;

	
	



}

