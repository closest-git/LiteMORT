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
#include "../data_fold/FeatVec_Quanti.hpp"
#include "../include/LiteBOM_config.h"

using namespace Grusoft;

struct MORT{
	LiteBOM_Config config;
	GBRT *hGBRT = nullptr;
	ExploreDA *hEDA_train = nullptr;
	vector<FeatsOnFold *>merge_folds;

	MORT() {

	}

	static MORT *From(void *mort_0, int flag = 0x0) {
		MORT *mort = static_cast<MORT *>(mort_0);
		return mort;
	}
	virtual ~MORT() {
		if (hGBRT != nullptr)		
			delete hGBRT;
		if (hEDA_train != nullptr)
			delete hEDA_train;
	}
};

FeatsOnFold *FeatsOnFold_InitInstance(LiteBOM_Config config, ExploreDA *edaX, PY_DATASET *dataset_, MORT *mort, int flag);

//LiteBOM_Config config;
//ExploreDA *g_hEDA=nullptr;
void LiteBOM_Config::OnObjective() {
	if (objective == "outlier") {
		leaf_optimal = "lambda_0";

	}
	if (objective == "lambdaMART") {
	}
	if (objective == "binary") {
		//eval_metric = "logloss";
		//eval_metric = "auc";
		//eval_metric = "WMW";	//"Wilcoxon-Mann-Whitney";
	}
	//nMostSalp4bins = 64;
	if (num_threads <= 0) {
		int nMostThread = 0;
#pragma omp parallel
		nMostThread = omp_get_num_threads();
		num_threads = nMostThread;
	}
	//eta = 0.1;
	//nBlitThread = 8;	//并行导致结果不再可重复
}

void OnUserParams(LiteBOM_Config&config, PY_ITEM* params, int nParam, int flag = 0x0) {
	char sERR[10000];
	int i,err=0;
	for (i = 0; i < nParam; ++i) {
		//printf("\"%s\"=%f\t", params[i].Keys, params[i].Values);
		if (strcmp(params[i].Keys, "num_leaves") == 0) {
			config.num_leaves = params[i].Values;
			if (config.num_leaves <= 1) {
				sprintf(sERR,"\"num_leaves\"=%d", config.num_leaves );		
				err=-1;				goto END;
			}
		}
		if (strcmp(params[i].Keys, "representive") == 0) {
			//assert(params[i].arr != nullptr);
		}
		if (strcmp(params[i].Keys, "verbose") == 0) {
			config.verbose = params[i].Values;
		}
		if (strcmp(params[i].Keys, "verbose_eval") == 0) {
			config.verbose_eval = params[i].Values;
		}
		if (strcmp(params[i].Keys, "max_depth") == 0) {
			config.max_depth = params[i].Values;
		}
		if (strcmp(params[i].Keys, "elitism") == 0) {
			config.rElitism = params[i].Values;
		}
		if (strcmp(params[i].Keys, "learning_rate") == 0) {
			config.learning_rate = params[i].Values;
		}
		if (strcmp(params[i].Keys, "debug") == 0) {
			config.isDebug_1 = strcmp(params[i].text, "1") == 0;
		}
		if (strcmp(params[i].Keys, "learning_schedule") == 0) {
			config.lr_adptive_leaf = strcmp(params[i].text, "adaptive") == 0;
		}
		if (strcmp(params[i].Keys, "lambda_l2") == 0) {
			config.lambda_l2 = params[i].Values;
		}
		if (strcmp(params[i].Keys, "adaptive") == 0) {
			config.adaptive_sample_weight = strcmp(params[i].text, "weight") == 0;
		}
		if (strcmp(params[i].Keys, "feat_factor") == 0) {
			float *factor = (float*)params[i].arr;
			config.feat_selector = factor;
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
		if (strcmp(params[i].Keys, "n_threads") == 0) {
			config.num_threads = (int)params[i].Values;
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
		if (strcmp(params[i].Keys, "prune") == 0) {
			config.nMostPrune = params[i].Values;
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
		if (strcmp(params[i].Keys, "salp_bins") == 0) {
			config.nMostSalp4bins = params[i].Values;
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
				if (config.eval_metric!="auc" && config.eval_metric!="mse" && config.eval_metric != "rmse" && config.eval_metric != "mae") {
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

PYMORT_DLL_API void LiteMORT_set_mergesets(void *mort_0, PY_DATASET_LIST *merge_list, int64_t flag) {
	MORT *mort = MORT::From(mort_0);
	mort->merge_folds.clear();
	if (merge_list != nullptr) {
		for (int i = 0; i < merge_list->nSet; i++) {
			PY_DATASET *set = merge_list->list + i;
			printf("\n\t------MERGE@[\"%s\"](%lldx%d)......", set->name, set->nSamp, set->ldFeat);
			ExploreDA *hEDA = new ExploreDA(mort->config,set->name, flag);
			FeatsOnFold *hMerge = FeatsOnFold_InitInstance(mort->config, hEDA, set, nullptr, flag | FeatsOnFold::DF_MERGE);
			//hMerge->merge_right = set->merge_rigt;
			mort->merge_folds.push_back(hMerge);
		}/**/
	}
}


//GBRT *hGBRT = nullptr;
PYMORT_DLL_API void* LiteMORT_init(PY_ITEM* params, int nParam, PY_DATASET_LIST *null_list, int64_t flag = 0x0) {
	try {
		printf("\n======LiteMORT_api init......");		//大道至简

		MORT *mort = new MORT();
		OnUserParams(mort->config, params, nParam);
		mort->hEDA_train = new ExploreDA(mort->config,"MORT", flag);
		printf("======LiteMORT_api init @%p(hEDA=%p,hGBRT=%p)...OK\n", mort,mort->hEDA_train,mort->hGBRT);
		
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
	//printf("\n======LiteMORT_api clear @%p(hEDA=%p,hGBRT=%p)...", mort_0,mort->hEDA, mort->hGBRT);
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


namespace Grusoft {
	FeatVector *FeatVecQ_InitInstance(FeatsOnFold *hFold, FeatVector *hFeat, int x, int flag) {
		int nBins = hFold->config.feat_quanti;
		//assert(hFeat != nullptr && hFeat->hDistri != nullptr);
		Distribution *distri = hFold->histoDistri(hFeat);
		HistoGRAM *histo = distri->histo;
		if (histo != nullptr) {
			nBins = histo->nBins;
		}
		assert(nBins > 0);
		FeatVector *hFQ = nullptr;
		if (nBins <= 256) {
			hFQ = new FeatVec_Q<uint8_t>(hFold, hFeat, x);
			//hFQ = new FeatVec_Q<uint16_t>(hFold, hFeat, x);		printf("----%d\t \"%s\" nBins=%d\n", hFeat->id, hFeat->nam.c_str(), nBins);
		}
		else if (nBins <= SHRT_MAX) {
			hFQ = new FeatVec_Q<uint16_t>(hFold, hFeat, x);
			if(hFold->config.verbose>0)
				printf("\t----%d\t FeatVec_Q<uint16_t>@\"%s\" nBins=%d\n", hFeat->id, hFeat->nam.c_str(), nBins);
		}
		else {
			assert(0);
			hFQ = new FeatVec_Q<uint32_t>(hFold, hFeat, x);
		}
		if (hFQ != nullptr) {
			hFQ->UpdateHisto(hFold, false, true);
			if (!hFold->config.isDynamicHisto && !hFeat->AtFold_()->isMerge())
				hFeat->FreeVals();		//需要dynamic update histo
		}
		return hFQ;
	}
}

/*
	v0.1	cys
		11/11/2019
*/
FeatVector *PY_COL2FEAT(const FeatsOnFold *hFold, PY_COLUMN*col, size_t nSamp_,int id,bool isMergePivot,int flag) {
	string desc = "feat_" + std::to_string(id);
	desc = col->name;
	FeatVector *hFeat = nullptr;
	int flagF = flag | FeatVector::VAL_REFER;
	if (hFold->isTrain() && !isMergePivot) {
		hFold->edaX->AddDistri(col, id);		//col
	}

	if (col->isFloat()) {
		hFeat = new FeatVec_T<float>(hFold,nSamp_, id, desc, flagF);
	}
	else if (col->isFloat16()) {	//NO REFER!!!
		//if (config.verbose>666)
		//	printf("----%d\t \"%s\" is Float16\n", id, col->name);
		hFeat = new FeatVec_T<float>(hFold, nSamp_, id, desc, flag);
	}
	else if (col->isInt32()) {
		hFeat = new FeatVec_T<int32_t>(hFold, nSamp_, id, desc, flagF);
	}
	else if (col->isInt16()) {
		hFeat = new FeatVec_T<int16_t>(hFold, nSamp_, id, desc, flagF);
	}
	else if (col->isInt8()) {
		hFeat = new FeatVec_T<int8_t>(hFold, nSamp_, id, desc, flagF);
	}
	else if (col->isInt64()) {
		hFeat = new FeatVec_T<int64_t>(hFold, nSamp_, id, desc, flagF);
	}
	else if (col->isDouble()) {
		hFeat = new FeatVec_T<double>(hFold, nSamp_, id, desc, flagF);
	}
	else
		throw "FeatsOnFold_InitInstance col->dtype is XXX";


	hFeat->PY = col;
	hFeat->nam = col->type_x;		hFeat->nam += col->name;
	hFeat->UpdateType();

	hFeat->Set(nSamp_, col);
	if (hFold->isTrain() && !isMergePivot) {
		Distribution *tDistri = hFold->edaX->GetDistri(id);// hFold->edaX == nullptr ? nullptr : &(hFold->edaX->arrDistri[i]);
		hFeat->InitDistri(hFold, tDistri,nullptr,true, 0x0);		
	}
	else {
		hFeat->InitDistri(hFold, nullptr, nullptr,false, 0x0);		

	}

	return hFeat;
}

//FeatsOnFold *FeatsOnFold_InitInstance(LiteBOM_Config config, ExploreDA *edaX, string nam_, PY_COLUMN *cX_, PY_COLUMN *cY_, size_t nSamp_, size_t ldX_, size_t ldY_, int flag) {
FeatsOnFold *FeatsOnFold_InitInstance(LiteBOM_Config config, ExploreDA *edaX, PY_DATASET *dataset_, MORT *mort, int flag) {
		clock_t t0 = clock();
	assert(BIT_IS(flag, FeatsOnFold::DF_FLAG));
	if (config.eda_Normal == LiteBOM_Config::NORMAL_gaussian) {
		edaX = nullptr;
		printf("\n---- !!! Data normalization by gaussian So edaX is nullptr@FeatsOnFold_InitInstance !!!----\n");
	}
	//bool isQuanti = config.feat_quanti >0 && BIT_TEST(flag, FeatsOnFold::DF_TRAIN);	// BIT_TEST(flag, FAST_QUANTI);
	bool isTrain = BIT_TEST(flag, FeatsOnFold::DF_TRAIN);
	bool isPredict = BIT_TEST(flag, FeatsOnFold::DF_PREDIC);
	bool isMerge = BIT_TEST(flag, FeatsOnFold::DF_MERGE);	
	double sparse = 0, nana = 0;
	size_t nSamp_ = dataset_->nSamp, nMostQ = config.feat_quanti, nConstFeat = 0, nMergedFeat=0, nLocalConst = 0, nQuant = 0;
	FeatsOnFold *hFold = new FeatsOnFold(config, edaX, dataset_->name, flag);
	hFold->InitRanders();
	PY_COLUMN *cY_ = dataset_->columnY;
	hFold->InitMost(dataset_->nSamp);
	//hFold->nMost = nSamp_;
	int rnd_seed = 0, nThread = config.num_threads, flagF= flag|FeatVector::VAL_REFER;
	for (size_t i = 0; i < dataset_->ldFeat; i++) {
		string desc = "feat_";
		PY_COLUMN *col = dataset_->columnX + i;//唯一的dtype处理
		if (i != 18)
			;// continue;
		//Distribution *hD_ = hFold->edaX == nullptr ? nullptr : &(hFold->edaX->arrDistri[i]);
		//if (hD_ != nullptr && hD_->nam.size() > 0)
			;// assert(hD_->nam == string(col->name));	//需要验证
		hFold->feats.push_back(PY_COL2FEAT(hFold,col, nSamp_, i, false ,flag));
	}

	if (mort != nullptr && mort->merge_folds.size() > 0) {
		int i, nMerge = mort->merge_folds.size();
		for (i = 0; i < nMerge; i++) {
			PY_COLUMN *col = dataset_->merge_left + i;
			FeatVector *hFeat = PY_COL2FEAT(hFold, col,  nSamp_, hFold->nFeat() + i, true,flag);
			BIT_SET(hFeat->type, FeatVector::AGGREGATE);
			hFeat->map4set = new tpSAMP_ID[nSamp_ * 2];		//hFeat->map4feat = hFeat->map4set + nSamp_;
			hFold->merge_lefts.push_back(hFeat);
		}
		hFold->ExpandMerge(mort->merge_folds);
	}
	//if (hFold->hMove != nullptr)
	//	hFold->hMove->Init_T<Tx, Ty>(nSamp_);
	
	if (!isMerge) {		//lossy and importance
		hFold->importance = new Feat_Importance(hFold);
		hFold->lossy->Init_T<tpDOWN>(hFold, nSamp_, 0x0, rnd_seed, flag);
		if (isPredict) {

		}
		else {
			FeatVector *Y = hFold->GetY();
			//Y->Set(nSamp_, (void*)(Y_));
			Y->Set(nSamp_, cY_);		
		}
		hFold->lossy->EDA(nullptr, flag);
	}

	GST_TIC(t1);
	int nFeat = hFold->nFeat();
	if (true) {	//为了调试
		std::sort(hFold->feats.begin(), hFold->feats.end(), FeatVector::OrderByName );
		for (int i = 0; i < nFeat; i++) {
			//hFold->feats[i]->id = i;		指向eda->distribution，不能改
		}
	}

//#pragma omp parallel for num_threads(nThread) schedule(dynamic) reduction(+ : sparse,nana,nConstFeat,nLocalConst,nQuant) 
	for (int feat = 0; feat < nFeat; feat++) {
		FeatVector *hFQ = nullptr;
		FeatVector *hFeat = hFold->Feat(feat);		
		if(feat==77)
			feat = 77;
		Distribution *fDistri = hFeat->myDistri();
		sparse += fDistri->rSparse*nSamp_;
		nana += fDistri->rNA*nSamp_;
		//if (BIT_TEST(hFeat->type, FeatVector::V_ZERO_DEVIA)) {
		if (fDistri->isPass()) {
			//printf("%d\n", feat);
			nConstFeat++;			nLocalConst++;
			//hFeat->Clear();		//释放内存
		}
		else if (hFeat->isMerged() || hFold->isMerge() ) {		//(hFeat->isMerged() || isMerge)
			nMergedFeat++;

		}	else {
			if (hFold->isQuanti || hFeat->isCategory()) {
				hFold->feats[feat] = hFQ = FeatVecQ_InitInstance(hFold, hFeat, 0x0);	// new FeatVec_Q<short>(hFold, hFeat, nMostQ);
				nQuant++;	//delete hFeat;
			}
		}
	}
	//FeatsOnFold::stat.tX += GST_TOC(t1);
	if (hFold->isQuanti) {
		hFold->Feature_Bundling();
	}
	int nTotalBins = 0;
	for (int feat = 0; feat < hFold->nFeat(); feat++) {
		FeatVector *hFeat = hFold->Feat(feat);
		if (config.feat_selector != nullptr) {
			hFeat->select.user_rate = config.feat_selector[feat];
			printf("%d(%.3g)\t", feat, hFeat->select.user_rate);
		}
		if (BIT_TEST(hFeat->type, FeatVector::REPRESENT_)) {
			PY_COLUMN *col = hFeat->PY;			// dataset_->columnX + feat;
			assert(hFeat->select.isPick = true);
			hFeat->select.isPick = false;
			hFold->present.Append(hFeat, col->representive);
		}
		if (hFold->config.verbose>666 ) {
			hFeat->myDistri()->Dump(feat, false, flag);					//Train输出distribution信息
		}
		if (!isMerge) {
			Distribution *hDistri = hFold->histoDistri(hFeat);
			nTotalBins += hDistri->nHistoBin();
		}
	}
											/*if (hFold->isQuanti) {
	printf("\n********* FeatsOnFold::QUANTI nMostQ=%d\r\n", nMostQ);
	}*/
	sparse /= (nSamp_*hFold->nFeat());
	nana /= (nSamp_*hFold->nFeat());
	//assert(nana == 0.0);
	if (config.verbose > 0) {
		printf("\r********* Fold_[%s] nSamp=%lld nFeat=%lld(const=%lld) QUANT=%lld Total Bins=%d\n\tsparse=%g NAN=%g nLocalConst=%lld time=%g sec\r\n",
			hFold->nam.c_str(), nSamp_, hFold->nFeat(), nConstFeat, nQuant,nTotalBins, sparse, nana, nLocalConst, (clock() - t0) / 1000.0);
	}
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
	FeatVec_T<tpDOWN> *fPred = dynamic_cast<FeatVec_T<tpDOWN> *>(pred);	assert(fPred != nullptr);
	tpDOWN *p_val = fPred->arr();
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
}*/

/*
	v0.2
*/
PYMORT_DLL_API void LiteMORT_predict_1(void *mort_0, PY_DATASET_LIST*predict_list, size_t flag) {
	PY_DATASET* predict_set = PY_DATASET_LIST::GetSet(predict_list,0);
	PY_COLUMN *col_y = predict_set->columnY,*X= predict_set->columnX;
	size_t nSamp = predict_set->nSamp;
	tpY *y = (tpY *)col_y->data;
	MORT *mort = MORT::From(mort_0);
	ExploreDA *hEDA = mort->hEDA_train;
	LiteBOM_Config& config = mort->config;
	if (mort->hGBRT == nullptr) {
		printf("********* LiteMORT_predict model is NULL!!!\n");
		return;
	}

	//y应设为nullptr
	//FeatsOnFold *hDat = FeatsOnFold_InitInstance(config, hEDA, "predict", X, col_y, nSamp, predict_set->ldFeat, 1, flag | FeatsOnFold::DF_PREDIC);
	FeatsOnFold *hDat = FeatsOnFold_InitInstance(config, hEDA, PY_DATASET_LIST::GetSet(predict_list),mort, flag | FeatsOnFold::DF_PREDIC);

	if(config.verbose>0)
		printf("\n********* LiteMORT_predict nSamp=%d,nFeat=%d hEDA=%p********* \n\n", nSamp, hDat->nFeat(), hEDA);
	//hDat->nam = "predict";
	mort->hGBRT->Predict(hDat);
	FeatVector *pred = hDat->GetPrecict();
	FeatVec_T<tpDOWN> *fY = dynamic_cast<FeatVec_T<tpDOWN> *>(pred);	assert(fY != nullptr);
	tpDOWN *p_val = fY->arr();
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
	fflush(stdout);
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
	需要重新设计		11/12/2019	cys
	*/
//PYMORT_DLL_API void LiteMORT_EDA(const float *X, const tpY *y, size_t nFeat_0, size_t nSamp, size_t flag) {
/*
void LiteMORT_EDA(void *mort_0, const size_t nFeat_0, const size_t nSamp_,const size_t nValid, PY_ITEM* descs, int nParam, const size_t flag)		{
	MORT *mort = MORT::From(mort_0);
	assert(nValid>=0 && nValid <= nSamp_);
	LiteBOM_Config& config = mort->config;
	//if (g_hEDA == nullptr)
	//	g_hEDA = new ExploreDA(config, nFeat_0, flag);
	mort->hEDA_train = new ExploreDA(config, nFeat_0, flag);
	int nDistr = mort->hEDA_train->arrDistri.size(),i;
	if (nParam > 0) {
		assert(nParam == nDistr);
		for (i = 0; i < nDistr; i++) {
			PY_ITEM* desc = descs + i;
			Distribution &distri = mort->hEDA_train->arrDistri[i];
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
	
	//g_hEDA->InitBundle(config, (float *)dataX, nSamp_, nFeat_0, flag);
	return ;
}*/

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

PYMORT_DLL_API void cpp_test(void *mort_0, PY_DATASET*dat) {
	printf("%s",dat->name);
}

//some EDA functions
/*
	v0.2

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
	size_t f1= FeatsOnFold::DF_TRAIN ;	
	vector<FeatsOnFold*> folds;
	FeatsOnFold *hFold = FeatsOnFold_InitInstance<float, tpY>(config, hEDA, "train",train_data, train_target, nSamp, nFeat_0, 1, flag | f1),
		*hEval=nullptr;
	folds.push_back(hFold);
	//hFold->lossy->Stat_Dump("",0x0);	//需要输出 Y的分布

	//int nTree = 501;		//出现过拟合
	int nTree = hFold->config.num_trees;
	if (nEval > 0) {
		ExploreDA *edaX_ = isDelEDA ? nullptr : hEDA;
		hEval= FeatsOnFold_InitInstance<float, tpY>(config, edaX_, "eval",eval_data,eval_target, nEval,nFeat_0, 1,flag | FeatsOnFold::DF_EVAL);
		//hEval->nam = "eval";
		folds.push_back(hEval);
	}
	//FeatsOnFold::stat.tX += GST_TOC(tick);
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
}*/


/*
	v0.1	cys
		10/22/2019
*/
void Feats_one_by_one(FeatsOnFold *hTrain, FeatsOnFold *hEval, BoostingForest::MODEL mod_, int nTre_, int flag) {
	LiteBOM_Config config_0 = hTrain->config;
	GBRT *hGBRT_0 = new GBRT(hTrain, hEval, 0, flag == 0 ? BoostingForest::REGRESSION : BoostingForest::CLASIFY, nTre_), *hGBRT=nullptr;
	int nFeat = hTrain->nFeat(), i, nSelect = 0, nFix = 0, cand, x = 0,cand_best=-1;
	size_t nTrain = hTrain->nSample(), nEval = hEval->nSample();
	vector<int> cands;
	const FeatVector *T_y = hTrain->GetY(), *E_y = hEval->GetY();
	FeatVector*T_predict = hTrain->GetPrecict(), *E_predict = hEval->GetPrecict(),*hBestFeat=nullptr;
	FeatVector*T_best_predict = new FeatVec_T<tpDOWN>(nullptr,T_predict->size(), 0, "T_best"),
		*E_best_predict = new FeatVec_T<tpDOWN>(nullptr, E_predict->size(),0,"E_best");
	float *selector = hTrain->config.feat_selector;		//返回值
	for (i = 0; i < nFeat; i++) {
		FeatVector *hFeat = hTrain->Feat(i);
		Distribution *hDistri = hTrain->histoDistri(hFeat);
		//if (hFeat->hDistri != nullptr && hFeat->hDistri->isPass())
		if (hDistri->isPass())
			continue;
		if (selector[i] != 1) {
			hFeat->select.isPick = false;	nSelect++;
			cands.push_back(i);
		}
		else {
			hFeat->select.isPick = true;
			nFix++;
		}
	}	
	//hTrain->Feat(cands[0])->select_factor = 1;
	bool fromLastResi = true;

	hGBRT_0->Train("", x, flag);
	hGBRT_0->Predict(hTrain, false, true, false);		
	T_best_predict->CopyFrom(T_predict);
	hGBRT_0->Predict(hEval, false, true, false);		
	E_best_predict->CopyFrom(E_predict);
	double loss = hEval->lossy->ERR(), loss_best = loss,a,T_err= hTrain->lossy->ERR();
	//std::vector<tpDOWN> T_resi = hTrain->lossy->resi , E_resi = hEval->lossy->resi, *hY_best = nullptr;
	hTrain->init_score.fVec = T_best_predict;
	hEval->init_score.fVec = E_best_predict;
	hTrain->config.early_stopping_round /= 2;
	for (i = 0; i < nSelect; i++) {
		cand = cands[i];		FeatVector *hFeat = hTrain->Feat(cand);
		assert(hFeat->select.isPick == false);	 hFeat->select.isPick = true;
		hEval->nam = "eval_"+hFeat->nam;
		//hTrain->InitFeatSelector();
		hGBRT = new GBRT(hTrain, hEval, 0, flag == 0 ? BoostingForest::REGRESSION : BoostingForest::CLASIFY, nTre_);
		hGBRT->isRefData = true;
		if (fromLastResi) {
			//T_y->Set(nTrain, VECTOR2ARR(T_resi), 0x0);			E_y->Set(nEval, VECTOR2ARR(E_resi), 0x0);
			hTrain->config.init_scor = "0";			hEval->config.init_scor = "0";
		}
		else {
			hTrain->init_score.fVec = nullptr;						hEval->init_score.fVec = nullptr;
			hTrain->config.init_scor = config_0.init_scor;			hEval->config.init_scor = config_0.init_scor;
		}
		//update train and eval'y
		hGBRT->Train("", x, flag);
		assert(hGBRT->stat.nMaxFeat == nFix+1 && hGBRT->stat.nMinFeat == nFix+1);
		//assert(hGBRT->stopping.errors[0]== loss_best);
		loss = hGBRT->stopping.ERR_best();	// hEval->lossy->ERR();
		//double percent = fabs
		hFeat->select.vari_1 = (loss_best - loss) / loss_best*100.0;
		if (loss < loss_best && hFeat->select.vari_1>0.001) {
			printf("\n------[%s] is usefull. loss=%.3g%%[%.7g=>%.7g]------", hFeat->nam.c_str(), hFeat->select.vari_1,loss_best,loss );
			loss_best = loss;		cand_best = cand;
			/*hGBRT->Predict(hTrain, false, true, false);				
			hGBRT->Predict(hEval,false,true,false);
			a = hEval->lossy->ERR();	
			assert(a==loss);
			T_best_predict->CopyFrom(T_predict);		E_best_predict->CopyFrom(E_predict);
			//T_resi = hTrain->lossy->resi, E_resi = hEval->lossy->resi;
			nFix++;		*/
		}
		else {
		}
		hFeat->select.isPick = false;
		delete hGBRT;
	}

	for (i = 0; i < nFeat; i++) {	//update selector
		if (selector[i] == 1)
			continue;
		FeatVector *hFeat = hTrain->Feat(i);
		//selector[i] = hFeat->select.vari_1;
	}
	if (cand_best != -1) {
		hBestFeat = hTrain->Feat(cand_best);
		selector[cand_best] = hBestFeat->select.vari_1;
		printf("___MORT_feat_select___:\tUpdate selector[%d]=%.5g feat=%s", cand_best,selector[cand_best], hBestFeat->nam.c_str());
		hBestFeat = hTrain->Feat(cand_best);
		//assert();
	}
	hTrain->config = config_0;		hEval->config = config_0;

	delete hGBRT_0;
}


/*
	v0.2
*/
//PYMORT_DLL_API void LiteMORT_fit_1(void *mort_0, PY_COLUMN *train_data, PY_COLUMN *train_target, size_t nFeat_0, size_t nSamp, PY_COLUMN *eval_data, PY_COLUMN *eval_target, size_t nEval, size_t flag) {
PYMORT_DLL_API void LiteMORT_fit_1(void *mort_0, PY_DATASET_LIST *train_list, PY_DATASET_LIST *eval_list, size_t flag) {
	try {
		GST_TIC(tick);
		assert(train_list!=nullptr && train_list->nSet == 1);
		PY_DATASET* train_set = PY_DATASET_LIST::GetSet(train_list);
		PY_DATASET* eval_set = (eval_list==nullptr || eval_list->nSet==0) ? nullptr : eval_list->list;
		MORT *mort = MORT::From(mort_0);
		LiteBOM_Config& config = mort->config;
		size_t nSamp = train_set->nSamp, nEval = eval_set == nullptr ? 0 : eval_set->nSamp;
		int nFeat_0 = train_set->ldFeat;
		assert(nSamp>0 && nFeat_0>0);
		//if(hGBRT!=nullptr)
		//	LiteMORT_clear();
		bool isDelEDA = false;
		ExploreDA *hEDA = (ExploreDA *)(mort->hEDA_train);
		/*if (hEDA == nullptr) {
			printf("\n********* g_hEDA on train_data ********* \n");
			LiteMORT_EDA(mort, nFeat_0*4, nSamp, 0, nullptr, 0x0, flag);
			hEDA = mort->hEDA_train;		//isDelEDA = true;
		}*/
		size_t i, feat, nTrain = nSamp;
		printf("\n********* LiteMORT_fit nSamp=%d,nFeat_0=%d hEDA=%p********* \n\n", nSamp, nFeat_0, hEDA);

		size_t f1 = FeatsOnFold::DF_TRAIN;
		vector<FeatsOnFold*> folds;
		FeatsOnFold *hFold = FeatsOnFold_InitInstance(config, hEDA, train_set, mort,flag | f1);
		FeatsOnFold *hEval = nullptr;
		folds.push_back(hFold);		

		//int nTree = 501;		//出现过拟合
		int nTree = hFold->config.num_trees;
		if (nEval > 0) {
			ExploreDA *edaX_ = isDelEDA ? nullptr : hEDA;
			//hEval = FeatsOnFold_InitInstance(config, edaX_, "eval", eval_set->columnX, eval_set->columnY, nEval, nFeat_0, 1, flag | FeatsOnFold::DF_EVAL);
			hEval = FeatsOnFold_InitInstance(config, edaX_, PY_DATASET_LIST::GetSet(eval_list), mort,flag | FeatsOnFold::DF_EVAL);
			folds.push_back(hEval);
		}


		if (config.feat_selector != nullptr) {
			Feats_one_by_one(hFold, hEval, flag == 0 ? BoostingForest::REGRESSION : BoostingForest::CLASIFY, nTree,0x0);
		}
		else {
			mort->hGBRT = new GBRT(hFold, hEval, 0, flag == 0 ? BoostingForest::REGRESSION : BoostingForest::CLASIFY, nTree);
			mort->hGBRT->Train("", 50, 0x0);
		}
		//delete mort;		//仅用于测试 
		if (isDelEDA) {
			delete hEDA;			hEDA = nullptr;
		}
		//memory
		mort->hGBRT->ClearData();
		mort->hGBRT->ClearHisto();
		//delete hEDA;		//仅用于调试
		//@%p(hEDA=%p,hGBRT=%p)	mort,mort->hEDA,mort->hGBRT,
		//FeatsOnFold::stat.tX += GST_TOC(tick);
		printf("\n********* LiteMORT_fit_1  time=%.3g(%.3g)......OK\n\n", GST_TOC(tick), FeatsOnFold::stat.tX + DCRIMI_2::tX);
	}
	catch (char * sInfo) {
		printf("\n!!!!!! EXCEPTION@LiteMORT_fit \n!!!!!!\"%s\"\n\n", sInfo);
		system("pause");
		throw sInfo;
	}
	catch (...) {
		printf("\n!!!!!! EXCEPTION@LiteMORT_fit %s!!!!!!\n\n", "...");
	}
	fflush(stdout);
	return;
}


PYMORT_DLL_API void LiteMORT_feats_one_by_one(void *mort_0, PY_COLUMN *train_data, PY_COLUMN *train_target, size_t nFeat_0, size_t nSamp, PY_COLUMN *eval_data, PY_COLUMN *eval_target, size_t nEval, size_t flag) {

}

const char *GRUS_LITEMORT_APP_NAME = "LiteMORT-beta";

void GRUS_LITEMORT_VERSION(char * str) {
	char sName[80];
	int i, nLen = (int)strlen(GRUS_LITEMORT_APP_NAME), nFrame = 68, off;

	for (i = 1; i < nFrame - 1; i++)		sName[i] = ' ';
	sName[0] = sName[nFrame - 1] = '*';
	sName[nFrame] = '\n';
	off = (nFrame - 2 - nLen) / 2 + 1;
	for (i = 0; i < nLen; i++)
		sName[i + off] = GRUS_LITEMORT_APP_NAME[i];
	sName[nFrame + 1] = '\0';

	sprintf(str, "%s%s%s",
		"********************************************************************\n",
		sName,
		"*                   for personal, non-commercial use.              *\n"
		"*    Copyright (c) 2018-2019 by YingShiChen. All Rights Reserved.  *\n"
		"*                         gsp@grusoft.com                          *\n"
		"********************************************************************\n"
	);	
}

#if (defined _WINDOWS) || (defined WIN32)
	BOOL APIENTRY DllMain(HANDLE hModule,DWORD  ul_reason_for_call,	LPVOID lpReserved){
		char str_version[1000];
		switch (ul_reason_for_call) {
		case DLL_PROCESS_ATTACH:		
			GRUS_LITEMORT_VERSION(str_version);
			printf("%s", str_version);
			break;
		case DLL_THREAD_ATTACH:
			break;
		default:
			break;
		}

		return TRUE;
	}
#else
//https://stackoverflow.com/questions/22763945/dll-main-on-windows-vs-attribute-constructor-entry-points-on-linux
	__attribute__((constructor)) void dllLoad() {
		char str_version[1000];
		GRUS_LITEMORT_VERSION(str_version);
		printf("%s", str_version);
	}

	__attribute__((destructor)) void dllUnload() {

	}
#endif