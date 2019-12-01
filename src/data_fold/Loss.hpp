#pragma once

#include "DataFold.hpp"
#include "../learn/DCRIMI_.hpp"
#include "Loss_binary.hpp"

namespace Grusoft {
	/*class LambdaRank {
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

	};*/
	/*
		v0.1	cys
			1/17/2019
	*/
	class FeatVec_LOSS {
	protected:
		const FeatsOnFold *hBaseData_=nullptr;
		LambdaRank HT_lambda_;
		DCRIMI_2 decrimi_2;
		Distribution dist_resi;
		IS_TYPE tpResi = is_XXX;
		FeatVector *y = nullptr, *predict = nullptr;// , *best_predict = nullptr;
		vector<tpSAMP_ID> outliers;
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
			size_t dim = resi.size(), i, nOutlier = outliers.size(), nMost = MIN2(dim - nOutlier, 200);
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
			size_t dim = resi.size(), i, nOutlier = outliers.size(), nMost = MIN2((dim - nOutlier)/ nOutlier, 1000),nz=0;
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
			nMost = MIN2((dim - nOutlier) / nOutlier, 1000);		nz = nOutlier;
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
					size_t start = thread*step, end = MIN2(start + step, dim), i;
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
		void WeightOnMarginal_2(FeatsOnFold *hData_, int round, int flag) {//初步测试，配合adaptive lr 还是有效果的
			size_t dim = resi.size(), nSamp = hData_->nSample(), step = dim;
			assert(dim == nSamp);
			Tx fuyi = -1, *y1 = ((FeatVec_T<Tx>*)predict)->arr(), *label = ((FeatVec_T<Tx>*)y)->arr();
			int num_threads = OMP_FOR_STATIC_1(dim, step);
			if (samp_weight == nullptr || round <10)
				return;
			double w0 = DBL_MAX, w1 = -DBL_MAX;
			decrimi_2.StatAtLabel(dim, label, y1, flag);
			double N_1= exp(decrimi_2.N_1);	 N_1 = N_1 / (1 + N_1);
			double P_0 = exp(decrimi_2.P_0);	 P_0 = P_0 / (1 + P_0);
//#pragma omp parallel for schedule(static,1)
			for (int thread = 0; thread < num_threads; thread++) {
				size_t start = thread*step, end = MIN2(start + step, dim), i;
				for (i = start; i < end; i++) {
					double sig, a;
					sig = exp(y1[i]);	 sig = sig / (1 + sig);					//[0-1]				
					a = (2 * sig - 1);						//a = max(-2, a);		a = min(2, a);
					samp_weight[i] = label[i] == 1 ? exp(-a) : exp(a);	
					/*,off= label[i] == 1? sig - N_1: P_0 - sig	
					if(off<0) {
						//off = min(1, off);		off = max(-1, off);
						//samp_weight[i] = exp(-off);
						//samp_weight[i] *= exp(-off);
					}*/
					w0 = MIN2(w0, samp_weight[i]);		w1 = MAX2(w1, samp_weight[i]);
				};
			}
			if (round % 100 == 0) {
				printf("w(%.4g,%.4g)\t", w0,w1);
			}
		}

		template <typename Tx>
		void UpdateResi_binary(FeatsOnFold *hData_, int round, int flag) {
			const string objective = hData_->config.objective, metric = hData_->config.eval_metric;
			bool isOutlier = objective == "outlier";
			Tx fuyi = -1, *y1 = ((FeatVec_T<Tx>*)predict)->arr(), *label = ((FeatVec_T<Tx>*)y)->arr();
			tpDOWN *vResi = VECTOR2ARR(resi), *pDown = GetDownDirection(),*delta = VECTOR2ARR(delta_step);
			tpDOWN *vHess = VECTOR2ARR(hessian);
			size_t dim = resi.size(), nSamp = hData_->nSample(), step = dim;
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
			if(samp_weight!=nullptr)
				WeightOnMarginal_2<Tx>(hData_,round,flag);
			
			if (metric == "logloss") {	//binary cross entropy
				err_logloss = 0;		//-np.mean(true_y*np.log(pred_h) + (1 - true_y)*np.log(1 - pred_h))
				//vEXP(dim, y_exp);
#pragma omp parallel for reduction( + : a_logloss )  schedule(static,1)
				for (int thread = 0; thread < num_threads; thread++) {
					size_t start = thread*step, end = MIN2(start + step, dim), i;
					for (i = start; i < end; i++) {
						//double a =  (2*y1[i] -1);
						//samp_weight[i] = label[i] == 1 ?exp(-a) : exp(a);
						double a = y1[i]<EXP_UNDERFLOW ? 0 : y1[i]>EXP_OVERFLOW ? y1[i] : log(1 + std::exp(y1[i]));
						if (label[i] == 1)
							a -= y1[i];
						a_logloss += a*samp_weight[i];
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
					size_t start = thread*step, end = MIN2(start + step, dim), i;
					for (i = start; i < end; i++) {				
						//double sig = y_exp[i] / (1 + y_exp[i]);
						double sig = y1[i]<EXP_UNDERFLOW ? 0 : y1[i]>EXP_OVERFLOW ? 1 : exp(y1[i]) / (1 + exp(y1[i])),a;
						//assert(!IS_NAN_INF(sig));
						pDown[i] = a = -(sig - label[i]);						vHess[i] = sig*(1 - sig);
						if (samp_weight != nullptr)
						{		pDown[i] *= samp_weight[i];			vHess[i] *= samp_weight[i];		}
						a2 += a*a;				sum += a;
						//a = pDown[i]* pDown[i] / vHess[i];
						//sumGH += a*a;		label_sum += label[i];						
					}
				}
				DOWN_sum_2 = a2;	DOWN_sum_1 = sum;
				DOWN_mean = sum / dim;
				double impuri = DOWN_sum_2 - dim*DOWN_mean*DOWN_mean;
				assert(impuri >= 0);
				DOWN_devia = sqrt(impuri / dim);
				//DOWN_GH_2 = sumGH;
				//LABEL_mean = label_sum*1.0/dim;
			}
			//if(y_exp!=nullptr)	
			//	delete[] y_exp;
		}

		//初步测试，无效
		template <typename Tx>
		void WeightOnMarginal_r(FeatsOnFold *hData_, int round, int flag) {
			size_t dim = resi.size(), nSamp = hData_->nSample(), step = dim,nzW=0;
			assert(dim == nSamp);
			tpDOWN *vResi = VECTOR2ARR(resi), *pDown = GetDownDirection();
			int num_threads = OMP_FOR_STATIC_1(dim, step);
			if (samp_weight == nullptr || round <10)
				return;
			dist_resi.STA_at(resi, false, 0x0);
			assert(dist_resi.nNA==0);
			double w0 = DBL_MAX, w1 = -DBL_MAX,devia= dist_resi.devia,a_1= (dist_resi.vMax- dist_resi.mean) / devia,T_w=2;
			// s = off_1 == off_0 ? 1 : 2.0 / (off_1 - off_0), T_off = off_0 + (off_1 - off_0)*0.9;
			//#pragma omp parallel for schedule(static,1)
			for (int thread = 0; thread < num_threads; thread++) {
				size_t start = thread*step, end = MIN2(start + step, dim), i;
				for (i = start; i < end; i++) {
					//double off =fabs(vResi[i]), a = off<T_off ? 1 : MIN2(2, exp(off- T_off));
					double a = fabs(vResi[i] - dist_resi.mean) / devia;
					//a = a<3 ? 1 : MIN2(2, exp(a-3));
					if (a <= T_w) {
						a = 1;
					}	else {
						a = MIN2(2, exp(a - T_w)); //a = MIN2(2, a - 2);
						nzW = nzW + 1;
					}
					samp_weight[i] = a;			pDown[i] *= sqrt(a);		//变通之举
					w0 = MIN2(w0, a);			w1 = MAX2(w1, a);
				};
			}
			if (round % 100 == 0) {
				printf("w(%.3g%%:%.4g,%.4g)\t", nzW*100.0/nSamp,w0, w1);
			}
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
			int num_threads = OMP_FOR_STATIC_1(dim, step);

			//double sum = 0;
			//以后要集成各种loss function

			if (objective == "binary") {	//binary `log loss
				UpdateResi_binary<Tx>(hData_,round,flag);
				//UpdateResi_binary_CYS<Tx>(hData_, round, flag);
				
			}	else if (objective == "multiclass") {	//MSE loss
				throw "Loss::multiclass is ...";
			}	else if (objective == "regression" || objective == "outlier") {
				//GST_TIC(t1);
				err_rmse = 0.0;		err_mae = 0;
#pragma omp parallel for schedule(static,1)
				OMP_FOR_func(vResi[i] = y1[i] - y0[i])
				/*for (i = 0; i < dim; i++) {
					vResi[i] = y1[i] - y0[i];
				}*/

				if (pDown != nullptr) {
					if (isOutlier) {		//Average Precision
						//Down_AP<Tx>();
						Lambda_0<Tx>();
						//Lambda_1<Tx>();
					}	else {	//MSE loss
						double a2 = 0, a1 = 0;
						if (metric == "mse" || metric == "rmse") {
							//for (i = 0; i < dim; i++) {		pDown[i] = -vResi[i];		}
#pragma omp parallel for schedule(static,1)
							OMP_FOR_func(pDown[i] = -vResi[i])
						}else if (metric == "mae") {
							//for (i = 0; i < dim; i++) {	pDown[i] = vResi[i]>0 ? -1 : 1;		}
#pragma omp parallel for schedule(static,1)
							OMP_FOR_func(pDown[i] = vResi[i]>0 ? -1 : 1)
						}	else {
							throw "UpdateResi metric is XXX for regression!!!";
						}
#pragma omp parallel for schedule(static,1) reduction(+ : a2,a1)
						for (int thread = 0; thread < num_threads; thread++) {
								size_t start = thread*step, end = MIN2(start + step, dim), i;	
								for (i = start; i < end; i++) { a2 += pDown[i] * pDown[i];		a1 += fabs(pDown[i]); }
						}
						//for (i = 0; i < dim; i++) {			a2 += pDown[i] * pDown[i];		a1 += fabs(pDown[i]);		}
						DOWN_sum_2 = a2;	DOWN_sum_1 = a1;
						if (samp_weight != nullptr && pDown != nullptr) {	//测试集无需加权
							WeightOnMarginal_r<Tx>(hData_, round, flag);
						}
					}					
				}
				double a2 = 0,a1 = 0;
#pragma omp parallel for schedule(static,1) reduction(+ : a2,a1)
				for (int thread = 0; thread < num_threads; thread++) {
					size_t start = thread*step, end = MIN2(start + step, dim), i;
					for (i = start; i < end; i++) {
						//for (i = 0; i < dim; i++) {
						double s = samp_weight != nullptr ? samp_weight[i] : 1;
						a2 += s*vResi[i] * vResi[i];
						a1 += s*fabs(vResi[i]);
					}
				}
				err_rmse = a2;		err_mae = a1;
				//sum = NRM2(dim, vResi);
				//参见loss = PointWiseLossCalculator::AverageLoss(sum_loss, sum_weights_)及L2Metric设计
				err_l2 = err_rmse / dim;
				err_rmse = sqrt(err_rmse / dim);	
				err_mae = err_mae / dim;
				DOWN_mean = DOWN_sum_1 / dim;
				double impuri = DOWN_sum_2 - dim*DOWN_mean*DOWN_mean;
				assert(impuri >= 0);
				DOWN_devia = sqrt(impuri / dim);
				//FeatsOnFold::stat.tX += GST_TOC(t1);
			}
			else {
				throw "objective";
			}
			return;
		}
	public:
		std::vector<tpDOWN> down, resi, hessian,sample_down,sample_hessian;		//negative_gradient,是否下降由LOSS判定		
		std::vector<tpDOWN> delta_step;
		float *samp_weight = nullptr;
		//参见samp_set之相关定义
		double DOWN_sum_1 = 0, DOWN_sum_2 = 0, DOWN_GH_2 = 0, LABEL_mean = 0, DOWN_0 = DBL_MAX, DOWN_1 = -DBL_MAX;
		double DOWN_mean, DOWN_devia;
		
		//https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d
		double err_rmse = DBL_MAX, err_mae = DBL_MAX, err_l2 = DBL_MAX,err_logloss= DBL_MAX, err_auc= DBL_MAX;
		double err_AP_outlier = DBL_MAX;		//average precision for outliers
		//pDown=target-predict
		tpDOWN *GetDownDirection() {
			return VECTOR2ARR(down);
		}
		tpDOWN *GetDeltaStep() {
			return VECTOR2ARR(delta_step);
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

		virtual void InitSampWeight(int flag=0x0);

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

			y = new FeatVec_T<Ty>(hData_,_len, 0, "loss");			predict = new FeatVec_T<Ty>(hData_,_len, 0, "predict");
			//best_predict = new FeatVec_T<Ty>(_len, 0, "best_predict");

			resi.clear();			resi.resize(_len, 0);
			//predict.resize(_len, 0);
			if (isTrain) {
				if (isTrain || isEval ) {
					down.resize(_len, 0);		sample_down.resize(_len, 0);
					delta_step.resize(_len, 0);
					hessian.clear();			sample_hessian.clear();
				}
				if (hData_->config.objective == "binary") {
					hessian.resize(_len, 0);	sample_hessian.resize(_len, 0);
				}else	if (hData_->config.objective == "outlier") {
					if (isTrain) {
						hessian.resize(_len, 0);		sample_hessian.resize(_len, 0);
						HT_lambda_.Init(10000,-1, 1, 10000);
					}	else {
						hessian.clear();		sample_hessian.clear();
					}
				}
				if(hData_->config.adaptive_sample_weight>0)
					InitSampWeight(flag);
			}
			if (isEval) {
				delta_step.resize(_len, 0);
			}
		}
		virtual void EDA(ExploreDA *edaX, int flag);
		virtual float* GetSampWeight(int flag) { return samp_weight; }

		FeatVec_LOSS() {
		}
		virtual ~FeatVec_LOSS(){
			Clear();
		}		
		virtual void Clear();
		//virtual void Stat_Dump(const string&info, int flag=0x0);
		virtual FeatVector * GetY() { return y; }

		virtual void Update(FeatsOnFold *hData_,int round, int flag = 0x0);

		template <typename Tx>
		double Adaptive_LR(MT_BiSplit *hBlit,bool isDelta_, int flag = 0x0) {
			assert(hBlit!=nullptr && hBaseData_->config.lr_adptive_leaf);
			//double s[] = {-0.01,0.1,0.5,1,2,5,10 };
			double s[] = { -0.01,0.1,0.5,1,2,5 };
			size_t i, loop, nLoop = sizeof(s) / sizeof(double), nSamp = delta_step.size();
			tpSAMP_ID *samps = nullptr, samp;
			if (hBlit == nullptr) {
			}	else {
				nSamp = hBlit->nSample();		samps = hBlit->samp_set.samps;
			}

			bool isTrain = BIT_TEST(hBaseData_->dType, FeatsOnFold::DF_TRAIN);
			bool isEval = BIT_TEST(hBaseData_->dType, FeatsOnFold::DF_EVAL);
			tpDOWN *delta_=VECTOR2ARR(this->delta_step);
			double err, min_err = DBL_MAX, eta_bst = 1.0, a;
			//min_err = -DBL_MAX;
			FeatVec_T<Tx> *hY = dynamic_cast<FeatVec_T<Tx>*>(y);		assert(hY != nullptr);
			const Tx *pred = ((FeatVec_T<Tx>*)predict)->arr(), *label = ((FeatVec_T<Tx>*)y)->arr();
			for (loop = 0; loop < nLoop; loop++) {
				if (isDelta_) {
					for (err = 0, i = 0; i<nSamp; i++) {
						samp = samps[i];
						a = pred[samp] + delta_[samp]* s[loop];
						err += a<EXP_UNDERFLOW ? 0 : a>EXP_OVERFLOW ? a : log(1 + std::exp(a));
						if (label[i] == 1)
							err -= a;
					}
				}	else {
					tpDOWN step_base = hBlit->GetDownStep();
					double step = step_base*s[loop];
					for (err = 0, i = 0; i<nSamp; i++) {
						samp = samps[i];
						a = pred[samp] + step;
						err += a<EXP_UNDERFLOW ? 0 : a>EXP_OVERFLOW ? a : log(1 + std::exp(a));
						if (label[samp] == 1)
							err -= a;
					}
				}
				assert(!IS_NAN_INF(err));
				if (err < min_err) {
				//if (err > min_err) {
					min_err = err;		eta_bst = s[loop];
				}
			}
			if (eta_bst < 0)
				;// printf("%.3g=>%.3g ", hBlit->lr_eta, eta_bst);
			if(hBlit!=nullptr)
				hBlit->lr_eta = eta_bst;
				
			return eta_bst;
		}

		virtual double ERR(int flag = 0x0);

		virtual bool isOK(int typ, double thrsh, int flag = 0x0);

		friend class FeatsOnFold;
		friend class GBRT;
	};




}

