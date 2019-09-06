#pragma once

#include "DataFold.hpp"
#include "../learn/DCRIMI_.hpp"

namespace Grusoft {
	class LambdaRank {
	protected:
		size_t nUnderFlow = 0, nOverFlow = 0;
		double a_0, a_1, grid;
		size_t nMost=-1;
		double sigma=20000;
		double *tables=nullptr;
	public:
		LambdaRank(int flag=0)	{}
		virtual ~LambdaRank() {
			if (tables != nullptr)
				delete[] tables;
		}
		virtual void Init(double sigma,double a_0,double a_1, size_t nMost, int flag = 0x0);
		virtual double At(double a);

	};
	/*
		v0.1	cys
			1/17/2019
	*/
	class FeatVec_LOSS {
	protected:
		const FeatsOnFold *hBaseData_=nullptr;
		LambdaRank HT_lambda_;
		DCRIMI_2 decrimi_2;
		IS_TYPE tpResi = is_XXX;
		FeatVector *y = nullptr, *predict = nullptr;
		vector<tpSAMP_ID> outliers;
		float *samp_weight = nullptr;
		//Average Precision (AP) 确实有问题
		template <typename Tx>
		void Down_AP() {
			Tx fuyi = -1, *y1 = ((FeatVec_T<Tx>*)predict)->arr(), *y0 = ((FeatVec_T<Tx>*)y)->arr();
			size_t dim = resi.size(), i, nOutlier = 0;
			tpDOWN *vResi = VECTOR2ARR(resi), *pDown = GetDownDirection();
			double eAll = 0, ePosi = 0, eNega = 0, *y_exp = new double[dim * 2], *grad_posi = y_exp + dim, alpha = 1000;
			for (i = 0; i < dim; i++) y_exp[i] = exp(y1[i]);
			//vEXP(dim, y_exp);
			//for (i = 0; i < dim; i++) y_exp[i] = y1[i]-0.9999;
			for (i = 0; i < dim; i++) {
				eAll += y_exp[i];
				if (y0[i] == 1) {	//outlier
					ePosi += y_exp[i];
				}
				else {
					eNega += y_exp[i];
				}
			}
			err_AP_outlier = ePosi / eAll;	//1-ePosi/eAll

			for (i = 0; i < dim; i++) y_exp[i] = y1[i];
			for (i = 0; i < dim; i++) {
				if (y0[i] == 1) {	//outlier
					pDown[i] = -y_exp[i] * eNega / eAll / eAll*alpha;
					grad_posi[nOutlier] = pDown[i];
					nOutlier++;
				}
				else {
					pDown[i] = y_exp[i] * (eAll - eNega) / eAll / eAll*alpha;
				}
			}
			delete[] y_exp;
		}

		/*
			难以理解，为何这个版本更好
			1 outlier只对比了很少一部分样本
			2 rou = HT_lambda_.At(y1[i] - y1[samp])	写反了
		*/
		template <typename Tx>
		void Lambda_0(int flag = 0x0) {
			size_t dim = resi.size(), i, nOutlier = outliers.size(), nMost = MIN(dim - nOutlier, 200);
			double sigma = dim, off, rou, grad, a;
			tpDOWN *vResi = VECTOR2ARR(resi), *pDown = GetDownDirection();
			tpDOWN *vHess = VECTOR2ARR(hessian);
			assert(hessian.size() == resi.size());
			Tx fuyi = -1, *y1 = ((FeatVec_T<Tx>*)predict)->arr(), *y0 = ((FeatVec_T<Tx>*)y)->arr();
			tpDOWN *mask = new tpDOWN[dim]();
			for (auto samp : outliers) {
				mask[samp] = 1;
			}
			for (auto samp : outliers) {
				for (i = 0; i < nMost; i++) {
					if (mask[i] == 1)
						continue;

					rou = HT_lambda_.At(y1[i] - y1[samp]);
					grad = -rou;
					pDown[samp] -= grad;		//down=-grad
					pDown[i] += grad;
					a = sigma*rou*(1 - rou);
					vHess[i] += a;		vHess[samp] += a;
				}
			}
			delete[] mask;
		}

		//优化版本
		template <typename Tx>
		void Lambda_1(int flag=0x0) {
			size_t dim = resi.size(), i, nOutlier = outliers.size(), nMost = MIN((dim - nOutlier)/ nOutlier, 1000),nz=0;
			double sigma = dim, off, rou,rou_1, grad, a;
			tpDOWN *vResi = VECTOR2ARR(resi), *pDown = GetDownDirection();
			tpDOWN *vHess = VECTOR2ARR(hessian);
			assert(hessian.size() == resi.size());
			Tx fuyi = -1, *y1 = ((FeatVec_T<Tx>*)predict)->arr(), *y0 = ((FeatVec_T<Tx>*)y)->arr();
			tpSAMP_ID id,*mask = new tpSAMP_ID[dim*2](),*ids=mask+dim;
			for (auto samp : outliers) {
				mask[samp] = 1;		ids[nz++] = samp;
			}
			for (i = 0; i < dim; i++) {
				if (mask[i] == 1)
					continue;
				ids[nz++] = i;
			}
			assert(nz == dim);
			std::random_shuffle(ids + nOutlier, ids + dim);
			nMost = MIN((dim - nOutlier) / nOutlier, 1000);		nz = nOutlier;
			for (auto samp : outliers) {
				for (i = nz; i < nz +nMost; i++) {
					id = ids[i];
					//rou = HT_lambda_.At(y1[samp] - y1[id]);
					rou = HT_lambda_.At(y1[id] - y1[samp]);		//效果明显更好，难以理解
					grad = -rou;
					pDown[samp] -= grad;		//down=-grad
					pDown[id] += grad;
					a = sigma*rou*(1 - rou);
					vHess[id] += a;		vHess[samp] += a;
				}
				nz += nMost;
			}
			delete[] mask;
		}

		//有问题，rank问题不好求解啊
		template <typename Tx>
		void UpdateResi_binary_CYS(FeatsOnFold *hData_, int round, int flag) {
			const string objective = hData_->config.objective, metric = hData_->config.eval_metric;
			bool isOutlier = objective == "outlier";
			Tx fuyi = -1, *y1 = ((FeatVec_T<Tx>*)predict)->arr(), *label = ((FeatVec_T<Tx>*)y)->arr(),a;
			tpDOWN *vResi = VECTOR2ARR(resi), *pDown = GetDownDirection();
			tpDOWN *vHess = VECTOR2ARR(hessian);
			size_t dim = resi.size(), nSamp = hData_->nSample(), step = dim, start, end;
			G_INT_64 i;
			double sum = 0, *y_exp = nullptr, a_logloss = 0;

			int num_threads = OMP_FOR_STATIC_1(dim, step);		
			bool isJonson = round<100;
			if (isJonson)	//应该渐进
				err_auc = decrimi_2.AUC_Jonson(dim, label, y1);
			else
				err_auc = decrimi_2.AUC_cys(dim, label, y1);
		
			if (pDown != nullptr) {
				Tx P_0 = decrimi_2.P_0, P_1 = decrimi_2.P_1, N_0= decrimi_2.N_0, N_1 = decrimi_2.N_1;
//#pragma omp parallel for schedule(static,1)
				for (int thread = 0; thread < num_threads; thread++) {
					size_t start = thread*step, end = min(start + step, dim), i;
					for (i = start; i < end; i++) {
						a = y1[i];
						if (label[i] == 0) {
							pDown[i] = -(a - N_0 );							vHess[i] = 1;
						}	else {
							pDown[i] = -(a - P_1 );							vHess[i] = 1;
						}
						/*if (label[i] == 0 && a > P_0) {
							pDown[i] = -(a - P_0+);							vHess[i] = 1;
						}else if (label[i] == 1 && a < N_1) {
							pDown[i] = -(a- N_1);							vHess[i] = 1;
						}	else {
							pDown[i] = 0;		vHess[i] = 1;
						}*/
					}
				}
			}
			if (y_exp != nullptr)
				delete[] y_exp;
		}

		template <typename Tx>
		void UpdateResi_binary(FeatsOnFold *hData_, int round, int flag) {
			const string objective = hData_->config.objective, metric = hData_->config.eval_metric;
			bool isOutlier = objective == "outlier";
			Tx fuyi = -1, *y1 = ((FeatVec_T<Tx>*)predict)->arr(), *label = ((FeatVec_T<Tx>*)y)->arr(),a;
			tpDOWN *vResi = VECTOR2ARR(resi), *pDown = GetDownDirection();
			tpDOWN *vHess = VECTOR2ARR(hessian);
			size_t dim = resi.size(), nSamp = hData_->nSample(), step = dim, start, end;
			G_INT_64 i;
			double a2 = 0, sum = 0,sumGH=0, label_sum=0,*y_exp = nullptr, a_logloss = 0;

			int num_threads = OMP_FOR_STATIC_1(dim, step);
			/*if (metric == "logloss" || pDown != nullptr) {		//get y_exp
				y_exp = new double[dim];
				//memcpy(y_exp, y1, sizeof(double)*dim);
#pragma omp parallel for schedule(static,1)
				for (int thread = 0; thread < num_threads; thread++) {
					size_t start = thread*step, end = min(start + step, dim), i;
					for (i = start; i < end; i++) {y_exp[i] = exp(y1[i]);}
				}
			}*/
			if (metric == "logloss") {
				err_logloss = 0;		//-np.mean(true_y*np.log(pred_h) + (1 - true_y)*np.log(1 - pred_h))
				//vEXP(dim, y_exp);
#pragma omp parallel for reduction( + : a_logloss )  schedule(static,1)
				for (int thread = 0; thread < num_threads; thread++) {
					size_t start = thread*step, end = min(start + step, dim), i;
					for (i = start; i < end; i++) {
						a_logloss += y1[i]<EXP_UNDERFLOW ? 0 : y1[i]>EXP_OVERFLOW ? y1[i] : log(1 + std::exp(y1[i]));
						//a_logloss += log(1 + y_exp[i]);
						//assert(!IS_NAN_INF(a_logloss));
						if (label[i] == 1)
							a_logloss -= y1[i];
					};
				}
				assert(!IS_NAN_INF(a_logloss));
				err_logloss = a_logloss / dim;
				
			}	else {	//'auc'
				bool isJonson = round < 50;
				if(isJonson)	//应该渐进
					err_auc = decrimi_2.AUC_Jonson(dim, label,y1);
				else
					err_auc = decrimi_2.AUC_cys(dim, label, y1);
			}
			if (pDown != nullptr) {
				Tx P_0 = decrimi_2.P_0, P_1 = decrimi_2.P_1, N_0 = decrimi_2.N_0, N_1 = decrimi_2.N_1;
#pragma omp parallel for schedule(static,1) reduction(+ : a2,sum,label_sum,sumGH)
				for (int thread = 0; thread < num_threads; thread++) {
					size_t start = thread*step, end = min(start + step, dim), i;
					for (i = start; i < end; i++) {				
						//double sig = y_exp[i] / (1 + y_exp[i]);
						double sig = y1[i]<EXP_UNDERFLOW ? 0 : y1[i]>EXP_OVERFLOW ? 1 : exp(y1[i]) / (1 + exp(y1[i]));
						//assert(!IS_NAN_INF(sig));
						pDown[i] = a = -(sig - label[i]);								vHess[i] = sig*(1 - sig);
						//pDown[i] *= samp_weight[i];		//思路有问题
						a2 += a*a;				sum += a;
						//a = pDown[i]* pDown[i] / vHess[i];
						//sumGH += a*a;		label_sum += label[i];
						/*a = y1[i];if (label[i] == 0 && a > P_0) {	//很奇怪，这样就是不行
							//pDown[i] *= 10;							
						}
						else if (label[i] == 1 && a < N_1) {
							//pDown[i] *= 10;							
						}*/
					}
				}
				DOWN_sum_2 = a2;	DOWN_sum_1 = sum;
				//DOWN_GH_2 = sumGH;
				//LABEL_mean = label_sum*1.0/dim;
			}
			//if(y_exp!=nullptr)	
			//	delete[] y_exp;
		}

		//vResi=predict-target		pDown=target-predict
		template <typename Tx>
		void UpdateResi(FeatsOnFold *hData_, int round, int flag = 0x0) {
			const string objective = hData_->config.objective,metric= hData_->config.eval_metric;
			bool isOutlier = objective == "outlier";	
			Tx fuyi = -1, *y1 = ((FeatVec_T<Tx>*)predict)->arr(), *y0 = ((FeatVec_T<Tx>*)y)->arr();
			tpDOWN *vResi = VECTOR2ARR(resi), *pDown = GetDownDirection();
			tpDOWN *vHess = VECTOR2ARR(hessian);
			size_t dim = resi.size(), nSamp=hData_->nSample(),step=dim,start,end;
			G_INT_64 i;
			//double sum = 0;
			//以后要集成各种loss function

			if (objective == "binary") {	//binary `log loss
				UpdateResi_binary<Tx>(hData_,round,flag);
				//UpdateResi_binary_CYS<Tx>(hData_, round, flag);
				
			}	else if (objective == "multiclass") {	//MSE loss
				throw "Loss::multiclass is ...";
			}	else if (objective == "regression" || objective == "outlier") {
				
				err_rmse = 0.0;		err_mae = 0;
				for (i = 0; i < dim; i++) {
					vResi[i] = y1[i] - y0[i];
					err_rmse += vResi[i] * vResi[i];
					err_mae += fabs(vResi[i]);
				}
				if (pDown != nullptr) {
					if (isOutlier) {		//Average Precision
						//Down_AP<Tx>();
						Lambda_0<Tx>();
						//Lambda_1<Tx>();
					}	else {	//MSE loss
						if (metric == "mse") {
							for (i = 0; i < dim; i++) {
								pDown[i] = -vResi[i];
							}
						}else if (metric == "mae") {
							for (i = 0; i < dim; i++) {
								pDown[i] = vResi[i]>0 ? -1 : 1;
							}
						}	else {
							throw "UpdateResi metric is XXX for regression!!!";
						}
					}					
				}
				//sum = NRM2(dim, vResi);
				//参见loss = PointWiseLossCalculator::AverageLoss(sum_loss, sum_weights_)及L2Metric设计
				err_l2 = err_rmse / dim;
				err_rmse = sqrt(err_rmse / dim);	
				err_mae = err_mae / dim;
			}
			else {
				throw "objective";
			}
			return;
		}
	public:
		std::vector<tpDOWN> down, resi, hessian,sample_down,sample_hessian;		//negative_gradient,是否下降由LOSS判定		
		//参见samp_set之相关定义
		double DOWN_sum_1 = 0, DOWN_sum_2 = 0, DOWN_GH_2 = 0, LABEL_mean = 0, DOWN_0 = DBL_MAX, DOWN_1 = -DBL_MAX;
		
		//https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d
		double err_rmse = DBL_MAX, err_mae = DBL_MAX, err_l2 = DBL_MAX,err_logloss= DBL_MAX, err_auc= DBL_MAX;
		double err_AP_outlier = DBL_MAX;		//average precision for outliers
		//pDown=target-predict
		tpDOWN *GetDownDirection() {
			return VECTOR2ARR(down);
		}
		tpDOWN *GetSampleDown() {
			return VECTOR2ARR(sample_down);
		}
		int *Tag() {//仅适用于分类问题
			CLASS_VEC *cls_vec = dynamic_cast<CLASS_VEC*>(y);			assert(cls_vec != nullptr);
			return cls_vec->arr();
		}

		size_t size()	const {
			return resi.size();
		}

		template<typename Ty>
		void Init_T(const FeatsOnFold *hData_, int _len, size_t x, int rnd_seed, int flag) {
			hBaseData_ = hData_;
			bool isTrain = BIT_TEST(flag, FeatsOnFold::DF_TRAIN);
			bool isEval = BIT_TEST(flag, FeatsOnFold::DF_EVAL);
			bool isPredict = BIT_TEST(flag, FeatsOnFold::DF_PREDIC);
			if (typeid(Ty) == typeid(float))
				tpResi = is_float;
			else if (typeid(Ty) == typeid(double))
				tpResi = is_double;
			else
				throw	"tpResi should be flow or double!!!";

			y = new FeatVec_T<Ty>(_len, 0, "loss");			predict = new FeatVec_T<Ty>(_len, 0, "predict");
			
			//predict.resize(_len, 0);
			if (isTrain || isEval ) {
				down.resize(_len, 0);		sample_down.resize(_len, 0);
				resi.clear();				
				hessian.clear();			sample_hessian.clear();
			}
			resi.resize(_len, 0);
			if (hData_->config.objective == "binary") {
				hessian.resize(_len, 0);	sample_hessian.resize(_len, 0);
				samp_weight = new float[_len]();
			}else	if (hData_->config.objective == "outlier") {
				if (isTrain) {
					hessian.resize(_len, 0);		sample_hessian.resize(_len, 0);
					HT_lambda_.Init(10000,-1, 1, 10000);
				}	else {
					hessian.clear();		sample_hessian.clear();
				}
			}
		}
		virtual void EDA(ExploreDA *edaX, int flag);
		virtual float* GetSampWeight(int flag) { return samp_weight; }

		FeatVec_LOSS() {
		}
		virtual ~FeatVec_LOSS() {
			down.clear();				sample_down.clear();
			resi.clear();
			hessian.clear();			sample_hessian.clear();

			if (samp_weight != nullptr)			delete samp_weight;
		}
		//virtual void Stat_Dump(const string&info, int flag=0x0);
		virtual FeatVector * GetY() { return y; }

		virtual void Update(FeatsOnFold *hData_,int round, int flag = 0x0);

		virtual double ERR(int flag = 0x0);

		virtual bool isOK(int typ, double thrsh, int flag = 0x0);

		friend class FeatsOnFold;
		friend class GBRT;
	};




}

