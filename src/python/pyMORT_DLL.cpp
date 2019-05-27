/*
	Issue-Needed:
	1	如何处理gain越来越小 例如hBlit->gain=0.003
	2 bug-
		校验还是无法通过
		assert( hBest->left->nSample()==hBest->bst_blit.nLeft && hBest->right->nSample()==hBest->bst_blit.nRight);
	3 leaf_optimal = "taylor_2" 收敛太慢
	4 drop out无效，真奇怪

	数据预处理流程
	1 data imputation		---	修改数据
		eda_NA=-1	则跳过（保留数据中的NA 似乎能提高准确率）
	2 nomalization			---	修改数据
		1)如eda_Normal=N(0,1) 	则跳过EDA的histogram
	3 EDA					---	对传入数据不做任何修改
		由于EDA基于多个数据源，其分析结果(例如histogram)可用于训练

	3_1_2019
	1)	"Finding Influential Training Samples for Gradient Boosted Decision Trees"
	2)	"Pruning Decision Trees via Max-Heap Projection"
		
*/

#include "pyMORT_DLL.h"
#include <stdio.h>
#include "../tree/ManifoldTree.hpp"
#include "../tree/GBRT.hpp"
#include "../util/Object.hpp"
#include "../data_fold/EDA.hpp"
#include "../data_fold/Loss.hpp"
#include "../include/LiteBOM_config.h"

using namespace Grusoft;


struct MORT{
	LiteBOM_Config config;
	GBRT *hGBRT = nullptr;
	ExploreDA *hEDA = nullptr;

	MORT() {

	}

	static MORT *From(void *mort_0, int flag = 0x0) {
		MORT *mort = static_cast<MORT *>(mort_0);
		return mort;
	}
	virtual ~MORT() {
		if (hGBRT != nullptr)		
			delete hGBRT;
		if (hEDA != nullptr)
			delete hEDA;
	}
};

//LiteBOM_Config config;
//ExploreDA *g_hEDA=nullptr;

void OnUserParams(LiteBOM_Config&config, PY_ITEM* params, int nParam, int flag = 0x0) {
	char sERR[10000];
	int i,err=0;
	for (i = 0; i < nParam; ++i) {
		printf("\"%s\"=%f\t", params[i].Keys, params[i].Values);
		//printf("%d: Title = %s\n", i, params[i].Title);
		//printf("%d: Index = %d\n", i, params[i].Index);
		if (strcmp(params[i].Keys, "num_leaves") == 0) {
			config.num_leaves = params[i].Values;
			if (config.num_leaves <= 1) {
				sprintf(sERR,"\"num_leaves\"=%d", config.num_leaves );		
				err=-1;				goto END;
			}
		}
		if (strcmp(params[i].Keys, "verbose") == 0) {
			config.verbose = params[i].Values;
		}
		if (strcmp(params[i].Keys, "learning_rate") == 0) {
			config.learning_rate = params[i].Values;
		}
		if (strcmp(params[i].Keys, "n_estimators") == 0) {
			config.num_trees = params[i].Values;
		}
		if (strcmp(params[i].Keys, "subsample") == 0) {
			config.subsample = params[i].Values;
		}
		if (strcmp(params[i].Keys, "early_stopping_rounds") == 0) {
			config.early_stopping_round = params[i].Values;
		}
		if (strcmp(params[i].Keys, "min_child_samples") == 0) {
			config.min_data_in_leaf = params[i].Values;
		}
		/*if (strcmp(params[i].Keys, "histo_bins") == 0) {
			config.histo_bins = params[i].Values;
		}*/
		if (strcmp(params[i].Keys, "feature_quanti") == 0) {
			config.feat_quanti = params[i].Values;
		}
		if (strcmp(params[i].Keys, "feature_sample") == 0) {
			config.feature_fraction = params[i].Values;
		}
		/*if (strcmp(params[i].Keys, "histo_algorithm") == 0) {
			config.histo_algorithm = (LiteBOM_Config::HISTO_ALGORITHM)(int)(params[i].Values);
		}*/
		if (strcmp(params[i].Keys, "histo_bin_map") == 0) {
			config.histo_bin_map = (LiteBOM_Config::HISTO_BINS_MAP)(int)(params[i].Values);
		}
		if (strcmp(params[i].Keys, "NA") == 0) {
			config.eda_NA = (LiteBOM_Config::EDA_NA)(int)(params[i].Values);
		}
		if (strcmp(params[i].Keys, "node_task") == 0) {
			config.node_task = (LiteBOM_Config::NODAL_TASK)(int)(params[i].Values);
		}
		if (strcmp(params[i].Keys, "normal") == 0) {
			config.eda_Normal = (LiteBOM_Config::EDA_NORMAL)(int)(params[i].Values);
			if (config.eda_Normal == LiteBOM_Config::NORMAL_gaussian) {
				assert(0);
				//config.histo_algorithm = LiteBOM_Config::HISTO_ALGORITHM::on_subsample;
			}
		}
		if (strcmp(params[i].Keys, "objective") == 0) {
			config.objective = params[i].text;
		}
		if (strcmp(params[i].Keys, "iter_refine") == 0) {
			config.T_iterrefine = params[i].Values;
		}
		//兼容Metric Parameters in lightgbm.pdf
		if (strcmp(params[i].Keys, "metric") == 0 || strcmp(params[i].Keys, "binary") == 0) {
			if (strcmp(params[i].text,"binary_logloss")==0)
				config.eval_metric = "logloss";
			else {
				config.eval_metric = params[i].text;
				if (config.eval_metric!="auc" && config.eval_metric!="mse" && config.eval_metric != "mae") {
					sprintf(sERR, "This version does not support the following eval_metric {\"%s\"}", config.eval_metric.c_str());
					throw sERR;
				}

			}
		}
	}
	config.OnObjective();
	config.leaf_optimal = "lambda_0";
END:
	if (err < 0) {
		printf("\n\n!!!Invalid Parameters!!! %s err=%d \n********* OnUserParams ********* \n\n", sERR, err);
		throw sERR;
	}
	printf("********* OnUserParams ********* \n\n");
}

//GBRT *hGBRT = nullptr;
PYMORT_DLL_API void* LiteMORT_init(PY_ITEM* params, int nParam, int flag = 0x0) {
	try {
		MORT *mort = new MORT();
		OnUserParams(mort->config, params, nParam);
		printf("\n======LiteMORT_api init @%p(hEDA=%p,hGBRT=%p)...OK\n", mort,mort->hEDA,mort->hGBRT);
		return mort;
	}
	catch (char * sInfo) {
		printf("\n!!!!!! EXCEPTION@LiteMORT_init \n!!!!!! %s\n\n", sInfo);
		throw sInfo;
	}
	catch (...) {
		printf("\n======LiteMORT_init FAILED...");
		return nullptr;
	}

}
PYMORT_DLL_API void LiteMORT_clear(void *mort_0) {
	MORT *mort = MORT::From(mort_0);
	printf("\n======LiteMORT_api clear @%p(hEDA=%p,hGBRT=%p)...", mort_0,mort->hEDA, mort->hGBRT);
	delete mort;
	printf("\r======LiteMORT_api clear @%p...OK\n", mort_0);
}

PYMORT_DLL_API void LiteMORT_set_feat(PY_ITEM* params, int nParam, int flag = 0x0) {
	//OnUserParams(config, params, nParam);
	try {

	}
	catch (...) {
		printf("\n======LiteMORT_set_feat FAILED...");
	}
}

void FeatsOnFold::ExpandFeat(int flag) {
	return;
	bool isTrain = BIT_TEST(dType,  FeatsOnFold::DF_TRAIN);
	bool isPredict = BIT_TEST(dType,  FeatsOnFold::DF_PREDIC);
	size_t nFeat_0 = feats.size(),i,nSamp_=nSample(),feat;
	int flagF = flag, nQuant=0, nMostQ = config.feat_quanti;
	for (feat = 0; feat < 2; feat++) {
		FeatVector *hBase = feats[feat];
		tpQUANTI pos,*quanti = hBase->GetQuantiBins();
		if (quanti != nullptr) {
			//edaX->arrDistri.resize(feats.size()+1);
			vector<BIN_FEATA>& featas = hBase->hDistri->binFeatas;
			if (!hBase->hDistri->isValidFeatas() )
				continue;
			FeatVec_T<float> *hExp = new FeatVec_T<float>(nSamp_, feats.size(), "exp" + std::to_string(feat), flagF);
			hExp->hDistri = &(edaX->arrDistri[feats.size()]);
			float *val = hExp->arr();
			for (i = 0; i < nSamp_; i++) {
				pos = quanti[i];
				val[i] = featas[pos].density;
			}
			if (isTrain)
				hExp->EDA(config, true, 0x0);
			if (isQuanti) {
				FeatVec_Q *hFQ = new FeatVec_Q(this, hExp, nMostQ);
				hFQ->UpdateHisto(this, false, true);
				feats.push_back(hFQ);
				nQuant++;	//delete hFeat;
			}else
				feats.push_back(hExp);
		}
	}
}

FeatsOnFold *FeatsOnFold_InitInstance(LiteBOM_Config config, ExploreDA *edaX, string nam_, PY_COLUMN *cX_, PY_COLUMN *cY_, size_t nSamp_, size_t ldX_, size_t ldY_, int flag) {
	clock_t t0 = clock();
	assert(BIT_IS(flag, FeatsOnFold::DF_FLAG));
	if (config.eda_Normal == LiteBOM_Config::NORMAL_gaussian) {
		edaX = nullptr;
		printf("\n---- !!! Data normalization by gaussian So edaX is nullptr@FeatsOnFold_InitInstance !!!----\n");
	}
	//bool isQuanti = config.feat_quanti >0 && BIT_TEST(flag, FeatsOnFold::DF_TRAIN);	// BIT_TEST(flag, FAST_QUANTI);
	double sparse = 0, nana = 0;
	size_t nTrain = nSamp_, nMostQ = config.feat_quanti, nConstFeat = 0, nLocalConst = 0, nQuant = 0;
	FeatsOnFold *hFold = new FeatsOnFold(config, edaX, nam_, flag);
	srand(time(0));
	//x = rand();
	hFold->rander_samp.Init(31415927 * rand());
	hFold->rander_feat.Init(123456789 * rand());

	hFold->InitMost(nSamp_);
	//hFold->nMost = nSamp_;
	int rnd_seed = 0, nThread = config.num_threads, flagF= flag|FeatVector::VAL_REFER;
	for (size_t i = 0; i < ldX_; i++) {
		string desc = "feat_";
		PY_COLUMN *col = cX_ + i;//唯一的dtype处理
		if (i == 9)
			i = 9;
		if (col->isFloat()) {
			hFold->feats.push_back(new FeatVec_T<float>(nSamp_, i, desc + std::to_string(i),flagF));		
		}	else if (col->isInt()) {
			hFold->feats.push_back(new FeatVec_T<int32_t>(nSamp_, i, desc + std::to_string(i), flagF));
		}	else if (col->isInt16()) {
			hFold->feats.push_back(new FeatVec_T<int16_t>(nSamp_, i, desc + std::to_string(i), flagF));
		}	else if (col->isChar()) {
			hFold->feats.push_back(new FeatVec_T<int8_t>(nSamp_, i, desc + std::to_string(i), flagF));
		}	else if (col->isInt64()) {
			hFold->feats.push_back(new FeatVec_T<int64_t>(nSamp_, i, desc + std::to_string(i), flagF));
		}	else if (col->isDouble()) {
			hFold->feats.push_back(new FeatVec_T<double>(nSamp_, i, desc + std::to_string(i), flagF));
		}
		else 
			throw "FeatsOnFold_InitInstance col->dtype is XXX";
		FeatVector *hFeat = hFold->feats[hFold->feats.size() - 1];
		hFeat->nam = col->type_x;		hFeat->nam += col->name;
		hFeat->hDistri = &(hFold->edaX->arrDistri[i]);
	}
	//if (hFold->hMove != nullptr)
	//	hFold->hMove->Init_T<Tx, Ty>(nSamp_);
	hFold->importance = new Feat_Importance(hFold);

	if (cY_->isFloat()) {
		hFold->lossy->Init_T<float>(hFold, nSamp_, 0x0, rnd_seed, flag);
	}	else if (cY_->isInt()) {
		hFold->lossy->Init_T<int32_t>(hFold, nSamp_, 0x0, rnd_seed, flag);
	}	else if (cY_->isInt16()) {
		hFold->lossy->Init_T<int16_t>(hFold, nSamp_, 0x0, rnd_seed, flag);
	}	else if (cY_->isChar()) {
		hFold->lossy->Init_T<int8_t>(hFold, nSamp_, 0x0, rnd_seed, flag);
	}	else if (cY_->isInt64()) {
		hFold->lossy->Init_T<int64_t>(hFold, nSamp_, 0x0, rnd_seed, flag);
	}	else if (cY_->isDouble()) {
		hFold->lossy->Init_T<double>(hFold, nSamp_, 0x0, rnd_seed, flag);
	}	else
		throw "FeatsOnFold_InitInstance col->dtype is XXX";

	bool isTrain = BIT_TEST(flag, FeatsOnFold::DF_TRAIN);
	bool isPredict = BIT_TEST(flag, FeatsOnFold::DF_PREDIC);
	if (isPredict) {

	}
	else {
		FeatVector *Y = hFold->GetY();
		//Y->Set(nSamp_, (void*)(Y_));
		Y->Set(nSamp_, cY_->data);
	}
	hFold->lossy->EDA(hFold, nullptr, flag);

	GST_TIC(t1);
	//#pragma omp parallel for num_threads(nThread) schedule(dynamic) reduction(+ : sparse,nana,nConstFeat,nLocalConst,nQuant) 
	for (int feat = 0; feat < ldX_; feat++) {
		FeatVec_Q *hFQ = nullptr;
		FeatVector *hFeat = hFold->Feat(feat);
		//if(feat==116)
		//	feat = 116;
		PY_COLUMN *col = cX_ + feat;
		//printf("\r\tfeat=%d\t......",feat);
		hFeat->Set(nSamp_, col->data);
		if(isTrain)
			hFeat->EDA(config,true, 0x0);		//EDA基于全局分析，而这里的是局部分析。分布确实会不一样
		sparse += hFeat->hDistri->rSparse*nSamp_;
		nana += hFeat->hDistri->rNA*nSamp_;
		//if (BIT_TEST(hFeat->type, FeatVector::V_ZERO_DEVIA)) {
		if (hFeat->hDistri->isPass()) {
			//printf("%d\n", feat);
			nConstFeat++;			nLocalConst++;
			//hFeat->Clear();		//释放内存
		}
		else {
			if (hFold->isQuanti) {
				hFold->feats[feat] = hFQ = new FeatVec_Q(hFold, hFeat, nMostQ);
				hFQ->UpdateHisto(hFold, false, true);
				nQuant++;	//delete hFeat;
			}
		}
	}
	//FeatsOnFold::stat.tX += GST_TOC(t1);
	if (hFold->isQuanti) {
		hFold->Feature_Bundling();
	}
	/*if (hFold->isQuanti) {
	printf("\n********* FeatsOnFold::QUANTI nMostQ=%d\r\n", nMostQ);
	}*/
	sparse /= (nSamp_*ldX_);
	nana /= (nSamp_*ldX_);
	//assert(nana == 0.0);
	printf("\r********* Fold_[%s] nSamp=%lld nFeat=%lld(const=%lld) QUANT=%lld\n\tsparse=%g NAN=%g nLocalConst=%lld time=%g sec\r\n",
		hFold->nam.c_str(), nSamp_, ldX_, nConstFeat, nQuant, sparse, nana, nLocalConst, (clock() - t0) / 1000.0);
	//if(nLocalConst>0)
	//	printf("\t!!! [%s] nLocalConst=%lld !!! \n",hFold->nam.c_str(), nLocalConst);

	return hFold;
}
/*
	模板编程真麻烦
*/
template<typename Tx, typename Ty>
FeatsOnFold *FeatsOnFold_InitInstance(LiteBOM_Config config, ExploreDA *edaX, string nam_,Tx *X_, Ty *Y_, size_t nSamp_, size_t ldX_, size_t ldY_, int flag) {
	clock_t t0 = clock();
	assert(BIT_IS(flag, FeatsOnFold::DF_FLAG));
	if (config.eda_Normal == LiteBOM_Config::NORMAL_gaussian) {
		edaX = nullptr;
		printf("\n---- !!! Data normalization by gaussian So edaX is nullptr@FeatsOnFold_InitInstance !!!----\n");
	}
	//bool isQuanti = config.feat_quanti >0 && BIT_TEST(flag, FeatsOnFold::DF_TRAIN);	// BIT_TEST(flag, FAST_QUANTI);
	double sparse = 0, nana=0;
	size_t nTrain = nSamp_, nMostQ = config.feat_quanti, nConstFeat = 0,nLocalConst=0, nQuant = 0;
	FeatsOnFold *hFold = new FeatsOnFold(config, edaX, nam_, flag);
	srand(time(0));
	//x = rand();
	hFold->rander_samp.Init(31415927 * rand());
	hFold->rander_feat.Init(123456789 * rand());

	hFold->InitMost(nSamp_);
	//hFold->nMost = nSamp_;
	for (size_t i = 0; i < ldX_; i++) {
		string desc = "feat_";
		hFold->feats.push_back(new FeatVec_T<Tx>(nSamp_, i, desc + std::to_string(i)));
		FeatVector *hFeat = hFold->feats[hFold->feats.size() - 1];
		hFeat->hDistri = &(hFold->edaX->arrDistri[i]);
	}
	if (hFold->hMove != nullptr)
		hFold->hMove->Init_T<Tx, Ty>(nSamp_);
	hFold->importance = new Feat_Importance(hFold);

	int rnd_seed = 0,nThread = config.num_threads;
	hFold->lossy->Init_T<Ty>(hFold,nSamp_,  0x0, rnd_seed, flag);

	bool isPredict = BIT_TEST(flag, FeatsOnFold::DF_PREDIC);
	bool isTrain = BIT_TEST(flag, FeatsOnFold::DF_TRAIN);
	if (isPredict) {

	}	else {
		FeatVector *Y = hFold->GetY();
		Y->Set(nSamp_, (void*)(Y_));	
		//hFold->lossy->decrimi_2.InitAtLabel(nSamp_,  flag);
	}
	hFold->lossy->EDA(hFold, nullptr, flag);

	GST_TIC(t1);
//#pragma omp parallel for num_threads(nThread) schedule(dynamic) reduction(+ : sparse,nana,nConstFeat,nLocalConst,nQuant) 
	for (int feat = 0; feat < ldX_; feat++) {
		FeatVec_Q *hFQ=nullptr;
		FeatVector *hFeat = hFold->Feat(feat);
		//if(feat==116)
		//	feat = 116;
		
		//printf("\r\tfeat=%d\t......",feat);
		hFeat->Set(nSamp_, (void*)(X_ + feat*nSamp_));
		if(isTrain)
			hFeat->EDA(config,true,0x0);		//EDA基于全局分析，而这里的是局部分析。分布确实会不一样
		sparse += hFeat->hDistri->rSparse*nSamp_;
		nana += hFeat->hDistri->rNA*nSamp_;
		//if (BIT_TEST(hFeat->type, FeatVector::V_ZERO_DEVIA)) {
		if (hFeat->hDistri->isPass()) {
			//printf("%d\n", feat);
			nConstFeat++;			nLocalConst++;
			//hFeat->Clear();		//释放内存
		}	else {
			if (hFold->isQuanti) {
				hFold->feats[feat] = hFQ = new FeatVec_Q(hFold, hFeat, nMostQ);
				hFQ->UpdateHisto(hFold,false,true);
				nQuant++;	//delete hFeat;
			}
		}
	}
	//FeatsOnFold::stat.tX += GST_TOC(t1);
	if (hFold->isQuanti) {
		hFold->Feature_Bundling();
	}
	/*if (hFold->isQuanti) {
	printf("\n********* FeatsOnFold::QUANTI nMostQ=%d\r\n", nMostQ);
	}*/
	sparse /= (nSamp_*ldX_);
	nana /= (nSamp_*ldX_);
	//assert(nana == 0.0);
	printf("\r********* Fold_[%s] nSamp=%lld nFeat=%lld(const=%lld) QUANT=%lld\n\tsparse=%g NAN=%g nLocalConst=%lld time=%g sec\r\n",
		hFold->nam.c_str(), nSamp_, ldX_, nConstFeat, nQuant, sparse,nana, nLocalConst,(clock() - t0) / 1000.0);
	//if(nLocalConst>0)
	//	printf("\t!!! [%s] nLocalConst=%lld !!! \n",hFold->nam.c_str(), nLocalConst);

	return hFold;
}

/*
	非常奇怪，为啥a=0是最好的
*/
template<typename Tx, typename Ty>
void Imputer_At_(Tx *X_, Ty *y, size_t nFeat, size_t nSamp_, size_t flag) {
	//distri.Imputer_at(config, nSamp_, x, 0x0);
	LiteBOM_Config config;
	GST_TIC(tick);
	printf("********* Imputer_At Tx=%d Ty=%d nSamp=%lld nFeat=%d........\n", sizeof(Tx),sizeof(Ty), nSamp_, nFeat);
	size_t feat, i,nFill=0;
	for (feat = 0; feat < nFeat; feat++) {
		if (feat == 80)
			feat = 80;
		Distribution *dis_0=new Distribution(),*dis_1=new Distribution();
		dis_0->desc = std::to_string(feat);
		Tx *x = X_ + feat*nSamp_, a2, sum, x_0, x_1;
		dis_0->STA_at(nSamp_, x, true, 0x0);
		if (dis_0->rNA > 0 && dis_0->rNA <1 && config.eda_NA != LiteBOM_Config::NA_) {
			double a = config.eda_NA == LiteBOM_Config::NA_ZERO ? 0 :
				config.eda_NA == LiteBOM_Config::NA_MEAN ? dis_0->mean : dis_0->median;
			//a = -666;
			Imputer_Fill_(config, nSamp_, x, a);
			dis_1->STA_at(nSamp_, x, true, 0x0);
			if( nFill++%10==0)
				printf("\r---[%s]: mean=%.2g->%.2g\trNA=%.2g->%.2g\n", dis_0->desc.c_str(), dis_0->mean, dis_1->mean, dis_0->rNA, dis_1->rNA);
		}
		if (config.eda_NA != LiteBOM_Config::NA_  && dis_0->rNA==1 ) {
			//Imputer_Fill_(config, nSamp_, x, 0);
		}
		delete dis_0;		delete dis_1;
	}
	printf("********* Imputer_At nSamp=%lld nFeat=%d Time=%g\n", nSamp_, nFeat, GST_TOC(tick) );

}

/*
	v0.1
*/
PYMORT_DLL_API void LiteMORT_predict(void *mort_0,float *X, tpY *y, size_t nFeat_0, size_t nSamp, size_t flag) {
	MORT *mort = MORT::From(mort_0);
	ExploreDA *hEDA = mort->hEDA;
	LiteBOM_Config& config = mort->config;
	if (mort->hGBRT == nullptr) {
		printf("********* LiteMORT_predict model is NULL!!!\n" );
		return;
	}

	if (false) {	//晕，y是未知的
		Distribution disY;
		disY.STA_at(nSamp, y, true, 0x0);
		if (disY.nNA > 0) {
			printf("********* LiteMORT_predict Y has nans(%lld)!!! Please check the value of Y!!!\n", disY.nNA);
			return;
		}

	}

	//y应设为nullptr
	FeatsOnFold *hDat = FeatsOnFold_InitInstance<float, tpY>(config, hEDA, "predict",X, y, nSamp, nFeat_0, 1, flag| FeatsOnFold::DF_PREDIC);
	printf("\n********* LiteMORT_predict nSamp=%d,nFeat=%d hEDA=%p********* \n\n", nSamp, nFeat_0, hEDA);
	//hDat->nam = "predict";
	mort->hGBRT->Predict(hDat);
	FeatVector *pred = hDat->GetPrecict();
	FeatVec_T<tpY> *fY = dynamic_cast<FeatVec_T<tpY> *>(pred);	assert(fY != nullptr);
	tpY *p_val = fY->arr();
	for (size_t i = 0; i<nSamp; i++)
		y[i] = p_val[i];
	if (config.objective == "binary") {
		//vEXP(nSamp, y);
		for (size_t i = 0; i < nSamp; i++) {
			y[i] = exp(y[i]);
			y[i] = (y[i] / (1 + y[i]));
		}
	}
	delete hDat;

}

/*
	v0.1	cys
		10/26/2018
	数据会被修改！！！
*/
PYMORT_DLL_API void LiteMORT_Imputer_f(float *X, tpY *y, size_t nFeat, size_t nSamp_, size_t flag) {
	Imputer_At_(X,y,nFeat,nSamp_,flag);
}
PYMORT_DLL_API void LiteMORT_Imputer_d(double *X, tpY *y, size_t nFeat, size_t nSamp_, size_t flag) {
	Imputer_At_(X, y, nFeat, nSamp_, flag);
}

/*
	EDA只分析数据，不应做任何修改
*/
//PYMORT_DLL_API void LiteMORT_EDA(const float *X, const tpY *y, size_t nFeat_0, size_t nSamp, size_t flag) {

PYMORT_DLL_API void LiteMORT_EDA(void *mort_0, const float *dataX, const tpY *dataY, const size_t nFeat_0, const size_t nSamp_,
	const size_t nValid, PY_ITEM* descs, int nParam, const size_t flag)		{
	MORT *mort = MORT::From(mort_0);
	assert(nValid>=0 && nValid <= nSamp_);
	LiteBOM_Config& config = mort->config;
	//if (g_hEDA == nullptr)
	//	g_hEDA = new ExploreDA(config, nFeat_0, flag);
	mort->hEDA = new ExploreDA(config, nFeat_0, flag);
	int nDistr = mort->hEDA->arrDistri.size(),i;
	if (nParam > 0) {
		assert(nParam == nDistr);
		for (i = 0; i < nDistr; i++) {
			PY_ITEM* desc = descs + i;
			Distribution &distri = mort->hEDA->arrDistri[i];
			distri.nam = desc->Keys;
			//int type = (int)(desc->Values);
			//distri.type = type == 1 ? FeatVector::CATEGORY : 0x0;
			char *type = desc->text;	
			if (type == 0x0) {
				distri.type = 0x0;
			}else
				distri.type = strcmp(type, "category") == 0 ? Distribution::CATEGORY : 0x0;
		}
	}
	if (dataX != nullptr && dataY != nullptr) {
		mort->hEDA->Analysis(config, (float *)dataX, (tpY *)dataY, nSamp_, nFeat_0, 1, flag);
		mort->hEDA->CheckDuplicate(config, flag);
	}	else {

	}
	//g_hEDA->InitBundle(config, (float *)dataX, nSamp_, nFeat_0, flag);
	return ;
}

/*	增量式EDA很难设计
	10/31/2018
	size_t feat;
	if(g_hEDA==nullptr)
		g_hEDA = new ExploreDA(config, nFeat_0, flag);
	for (feat = 0; feat < nFeat_0; feat++) {
		Distribution& distri = g_hEDA->arrDistri[feat];
		distri.desc = "distri_" + std::to_string(feat);
		float *tmpX=new float[];
		distri.STA_at(nSamp_, x, true, 0x0);
		distri.X2Histo_(config, nSamp_, x);
	}
	distri.Dump( );
	g_hEDA->Analysis(config, (float *)trainX, (tpY *)trainY, nTrain, nFeat_0, 1, flag);
	if (validX != nullptr) {
		ExploreDA eda(config, nFeat_0, flag);
		eda.Analysis(config, (float *)validX, (tpY *)nullptr, nValid, nFeat_0, 1, flag);
		//g_hEDA->Merge(eda);
	}
	//g_hEDA->InitBundle(config, X, nSamp, nFeat_0, flag);
}*/



//some EDA functions
/*
	v0.2
*/
PYMORT_DLL_API void LiteMORT_fit(void *mort_0, float *train_data, tpY *train_target, size_t nFeat_0, size_t nSamp,
		float *eval_data, tpY *eval_target,size_t nEval,size_t flag) {
	try {
	GST_TIC(tick);
	MORT *mort = MORT::From(mort_0);
	LiteBOM_Config& config = mort->config;
	//if(hGBRT!=nullptr)
	//	LiteMORT_clear();
	bool isDelEDA = false;
	ExploreDA *hEDA = (ExploreDA *)(mort->hEDA);
	if (hEDA == nullptr) {
		printf("\n********* g_hEDA on train_data ********* \n");
		LiteMORT_EDA(mort,train_data, train_target, nFeat_0, nSamp,0,nullptr,0x0,flag );
		hEDA = mort->hEDA;		//isDelEDA = true;
	}
	size_t nFeat = nFeat_0,i,feat, nTrain= nSamp;
	printf( "\n********* LiteMORT_fit nSamp=%d,nFeat=%d hEDA=%p********* \n\n", nSamp, nFeat, hEDA);
	Distribution disY;	
	disY.STA_at(nSamp, train_target, true, 0x0);
	if (disY.nNA > 0) {
		printf("********* LiteMORT_fit Y has nans(%lld)!!! Please check the value of Y!!!\n", disY.nNA);
		return;
	}
	if (true) {	//需要输出 Y的分布
		disY.X2Histo_<tpY, tpY>(config, nSamp, train_target,nullptr);
		disY.Dump(-1, false, flag);
	}

	size_t f1= FeatsOnFold::DF_TRAIN ;	
	vector<FeatsOnFold*> folds;
	FeatsOnFold *hFold = FeatsOnFold_InitInstance<float, tpY>(config, hEDA, "train",train_data, train_target, nSamp, nFeat_0, 1, flag | f1),
		*hEval=nullptr;
	folds.push_back(hFold);
	//hFold->nam = "train";

	//int nTree = 501;		//出现过拟合
	int nTree = hFold->config.num_trees;
	if (nEval > 0) {
		ExploreDA *edaX_ = isDelEDA ? nullptr : hEDA;
		hEval= FeatsOnFold_InitInstance<float, tpY>(config, edaX_, "eval",eval_data,eval_target, nEval,nFeat_0, 1,flag | FeatsOnFold::DF_EVAL);
		//hEval->nam = "eval";
		folds.push_back(hEval);
	}
	mort->hGBRT = new GBRT(hFold, hEval, 0, flag==0 ? BoostingForest::REGRESSION : BoostingForest::CLASIFY, nTree);
	
	mort->hGBRT->Train("", 50, 0x0);
	//delete mort;		//仅用于测试 
	if (isDelEDA) {
		delete hEDA;			hEDA = nullptr;
	}

	//@%p(hEDA=%p,hGBRT=%p)	mort,mort->hEDA,mort->hGBRT,
	printf("\n********* LiteMORT_api fit  time=%.3g(%.3g)......OK\n\n", GST_TOC(tick), FeatsOnFold::stat.tX+ DCRIMI_2::tX );

	}
	catch (char * sInfo) {
		printf("\n!!!!!! EXCEPTION@LiteMORT_fit \n!!!!!!\"%s\"\n\n", sInfo);
		system("pause");
		throw sInfo;
	}
	catch (...) {
		printf("\n!!!!!! EXCEPTION@LiteMORT_fit %s!!!!!!\n\n", "...");
	}
	return ;
}

/*
	v0.2
*/
PYMORT_DLL_API void LiteMORT_fit_1(void *mort_0, PY_COLUMN *train_data, PY_COLUMN *train_target, size_t nFeat_0, size_t nSamp, PY_COLUMN *eval_data, PY_COLUMN *eval_target, size_t nEval, size_t flag) {
	try {
		GST_TIC(tick);
		MORT *mort = MORT::From(mort_0);
		LiteBOM_Config& config = mort->config;
		//if(hGBRT!=nullptr)
		//	LiteMORT_clear();
		bool isDelEDA = false;
		ExploreDA *hEDA = (ExploreDA *)(mort->hEDA);
		if (hEDA == nullptr) {
			printf("\n********* g_hEDA on train_data ********* \n");
			LiteMORT_EDA(mort, nullptr, nullptr, nFeat_0*4, nSamp, 0, nullptr, 0x0, flag);
			hEDA = mort->hEDA;		//isDelEDA = true;
		}
		size_t nFeat = nFeat_0, i, feat, nTrain = nSamp;
		printf("\n********* LiteMORT_fit nSamp=%d,nFeat=%d hEDA=%p********* \n\n", nSamp, nFeat, hEDA);
		/*Distribution disY;
		//disY.STA_at(nSamp, train_target, true, 0x0);
		if (disY.nNA > 0) {
			printf("********* LiteMORT_fit Y has nans(%lld)!!! Please check the value of Y!!!\n", disY.nNA);
			return;
		}
		if (true) {	//需要输出 Y的分布
			//disY.X2Histo_<tpY, tpY>(config, nSamp, train_target, nullptr);
			disY.Dump(-1, false, flag);
		}*/

		size_t f1 = FeatsOnFold::DF_TRAIN;
		vector<FeatsOnFold*> folds;
		FeatsOnFold *hFold = FeatsOnFold_InitInstance(config, hEDA, "train", train_data, train_target, nSamp, nFeat_0, 1, flag | f1);
		FeatsOnFold *hEval = nullptr;
		folds.push_back(hFold);
		hFold->ExpandFeat();

		//int nTree = 501;		//出现过拟合
		int nTree = hFold->config.num_trees;
		if (nEval > 0) {
			ExploreDA *edaX_ = isDelEDA ? nullptr : hEDA;
			hEval = FeatsOnFold_InitInstance(config, edaX_, "eval", eval_data, eval_target, nEval, nFeat_0, 1, flag | FeatsOnFold::DF_EVAL);
			hEval->ExpandFeat();
			folds.push_back(hEval);
		}
		mort->hGBRT = new GBRT(hFold, hEval, 0, flag == 0 ? BoostingForest::REGRESSION : BoostingForest::CLASIFY, nTree);

		mort->hGBRT->Train("", 50, 0x0);
		//delete mort;		//仅用于测试 
		if (isDelEDA) {
			delete hEDA;			hEDA = nullptr;
		}

		//@%p(hEDA=%p,hGBRT=%p)	mort,mort->hEDA,mort->hGBRT,
		printf("\n********* LiteMORT_api fit  time=%.3g(%.3g)......OK\n\n", GST_TOC(tick), FeatsOnFold::stat.tX + DCRIMI_2::tX);

	}
	catch (char * sInfo) {
		printf("\n!!!!!! EXCEPTION@LiteMORT_fit \n!!!!!!\"%s\"\n\n", sInfo);
		throw sInfo;
	}
	catch (...) {
		printf("\n!!!!!! EXCEPTION@LiteMORT_fit %s!!!!!!\n\n", "...");
	}
	return;
}
