#include <omp.h>
//#include <io.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#ifdef JSON_LIB
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#endif

#include "DataFold.hpp"
#include "FeatVec_Quanti.hpp"
#include "FeatVec_EXP.hpp"
#include "Loss.hpp"
#include "EDA.hpp"
#include "../tree/BoostingForest.hpp"
#include "../include/LiteBOM_config.h"
#include "../EDA/SA_salp.hpp"
using namespace Grusoft;

FeatsOnFold::STAT FeatsOnFold::stat;
/*
流形上的结构化数据
需求：
百万-几十亿个样本
每个样本的特征几十到几千
每个样本有1到N个标签
设计
高效（时间，内存）第一
优雅，易读		参考PYTHON,	没必要依赖C++的莫名语法

*/
FeatsOnFold::FeatsOnFold(LiteBOM_Config confi_, ExploreDA *eda_, string nam_, int dtype) :config(confi_), edaX(eda_), nam(nam_) {
	dType = dtype;
	isQuanti = config.feat_quanti >0 && (BIT_TEST(dType, FeatsOnFold::DF_TRAIN) || BIT_TEST(dType, FeatsOnFold::DF_MERGE));
	//isQuanti = config.feat_quanti > 0;
/*https://stackoverflow.com/questions/9878965/rand-between-0-and-1
	uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::seed_seq ss{ uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32) };
	rng.seed(ss);*/
	lossy = new FeatVec_LOSS();
	if (isTrain()) {

	}	else {
		config.subsample = 1;	config.feature_fraction = 1;

	}

}

void FeatsOnFold::InitRanders(int flag) {
	if (!isTrain())
		return;

	srand(time(0));
	//x = rand();
	if (config.useRandomSeed) {
		rander_samp.Init(31415927 * rand());
		rander_feat.Init(123456789 * rand());
		rander_bins.Init(123456789 * rand());
		rander_nodes.Init(123456789 * rand());

	}
	else {
		srand(42);
		rander_samp.Init(31415927);
		rander_feat.Init(123456789);
		rander_bins.Init(20190826);
		rander_nodes.Init(42);
	}
}

Feat_Importance::Feat_Importance(FeatsOnFold *hData_, int flag) {
	size_t nFeat = hData_->feats.size();
	split_sum.resize(nFeat);
	gain_sum.resize(nFeat);
}


FeatsOnFold* FeatsOnFold::read_json(const string &sPath, int flag) {
/*	LiteBOM_Config config;
	std::ifstream ifile(sPath);
	json js;
	ifile >> js;
	size_t i = 0, j, nSamp = js.size(), ldX = 8, ldY = 1, nCls = 2, nEle;
	FeatsOnFold *hFold = new FeatsOnFold(config,nullptr,"" );
	hFold->Init_T<float,float>(nSamp, ldX, ldY, flag);
	//hFold->lossy->Init_T<float>(nSamp, ldY, flag);

	vector<FeatVector*>& feats = hFold->feats;
	FeatVector *Y =hFold->GetY();
	for (json::iterator it = js.begin(); it != js.end(); ++it, ++i) {
		nEle = it->size();		//assert(nEle == ldX + ldY);
		for (j = 0; j < nEle - 1; j++) {
			FeatVector* feat= feats[j];
			double a=it->at(j);
			feat->Set(i,a);
		}
		float label = it->at(nEle - 1);		assert(label == 0 || label == 1);
		Y->Set(i,label);
	}
	
	//hData->Reshape();
	return hFold;*/
	return nullptr;
}

void FeatsOnFold:: Distri2Tag(int *mark, int nCls, int flag) {
	float *dtr = distri, d1 = 0;
	int *tag = Tag();
	size_t i, j, total = nSample(), cls;
	rOK = 0;
	for (i = 0; i<total; i++, dtr += nCls) {
		for (cls = -1, d1 = 0, j = 0; j<nCls; j++) {
			if (dtr[j]>d1) { d1 = dtr[j];	cls = j; }
		}
		mark[i] = cls;
		if (mark[i] == tag[i]) {
			rOK += 1;
		}
	}
	rOK /= total;
}

/*
void FeatsOnFold::nPickBySwarm()
//bool isSwarm = feat_salps != nullptr && hForest->stopping.nBraeStep>0;
bool isSwarm = feat_salps != nullptr && feat_salps->isFull();
if (isSwarm) {
vector<int> pick_1,pick_2;
isSwarm = feat_salps->PickOnStep(hForest->stopping.nBraeStep, pick_1,false);
for (auto x : pick_1) {
int no = x;	// picks[x];
if (mask[no] == 0)
{		continue;		}
pick_2.push_back(no);
//if (pick_2.size() > 32)
//	break;
}
picks = pick_2;
}
if(!isSwarm){
vector<int> no_k = rander.kSampleInN(nPick, picks.size()),pick_1;
for (auto x : no_k) {
int no = picks[x];				pick_1.push_back(no);
}
picks = pick_1;
}
if (feat_salps != nullptr) {
feat_salps->AddSalp(nFeat, picks, nTree);
}
*/

/*
	feature_fraction似乎能降低overfitting
	v0.1	cys
		8/30/2019
*/
void FeatsOnFold::nPick4Split(vector<int>&picks, GRander&rander, BoostingForest *hBooster, int flag) {
	int i, nFeat = feats.size(), nPick = (int)(sqrt(nFeat));
	int nTree = hBooster->forest.size();
	int *mask = new int[nFeat]();
	//picks.resize(nFeat);
	for (i = 0; i<nFeat; i++)	{
		FeatVector *hFeat = Feat(i);
		hFeat->select.hasCheckGain = false;
		if (i != 71) {		//仅用于调试
#ifdef _DEBUG
			;// hFeat->select.isPick = false;
#endif
		}
		if (hFeat->hDistri!=nullptr && hFeat->hDistri->isPass())
			continue;
		if (BIT_TEST(hFeat->type, FeatVector::IS_BUNDLE))
			continue;
		if (BIT_TEST(hFeat->type, Distribution::DISTRI_OUTSIDE))
			continue;
		if (BIT_TEST(hFeat->type, FeatVector::AGGREGATE)) {
			
		}
		if (hFeat->select.isPick == false)		{
			continue;
		}
		/*if (config.feat_selector != nullptr && hFeat->select_factor < 1) {
			float a = rander.Uniform_(0, 1);
			if (hFeat->select_factor<a)	{
				continue;
			}
		}*/
		//if(hFeat->id!=360)	continue;	//仅用于测试 
		mask[i] = 1;
		picks.push_back(i);
	}
	assert(picks.size()>0);
	if (config.feat_selector != nullptr) {	//feat select
		//printf("nPick4Split=%d @feat_factor\t", picks.size());
	}else if (config.feature_fraction<1) {	//for random forest
		nPick = MAX2(1,picks.size()*config.feature_fraction+0.5);
		hBooster->stopping.CheckBrae();
		vector<int> no_k = rander.kSampleInN(nPick, picks.size()),pick_1;
		double *w = new double[nFeat]();
		for (auto x : no_k) {
			int no = picks[x];				pick_1.push_back(no);
			assert(mask[no] == 1);
			FeatVector *hFeat = Feat(no);
			w[no] = -hFeat->wSplit;			//有意思
			w[no] = -hFeat->wGain;	
		}
		picks = pick_1;		
		nPick = picks.size();
		if (config.rElitism>0 && hBooster->forest.size()>16) {
			std::sort(picks.begin(), picks.end(), [&w](size_t i1, size_t i2) {return w[i1] < w[i2]; });
			for (i = 0; i < nPick - 1; i++) {
				assert(w[picks[i]] <= w[picks[i + 1]]);
				assert(mask[picks[i]] == 1);
			}
			int nElitism = MAX2(16, nFeat*config.rElitism);
			nElitism = MIN2(nPick, nElitism);
			int nExp = nElitism + MIN2(nElitism * 3,  nFeat / 3);
			std::random_shuffle(picks.begin()+ nElitism, picks.end());
			picks.resize(MIN2(nPick, nExp));
		}
		delete[] w;

		std::sort(picks.begin(), picks.end());
	}
	delete[] mask;
	nPickFeat = picks.size();
	hBooster->stat.nMaxFeat = MAX2(hBooster->stat.nMaxFeat, nPickFeat);
	hBooster->stat.nMinFeat = MIN2(hBooster->stat.nMinFeat, nPickFeat);
	assert(nPickFeat>0);
}

int *FeatsOnFold::Rank4Feat(int type, int flag) {
	int i, nFeat = feats.size();
	double *wFeat = new double[nFeat]();
	int *rank = new int[nFeat]();
	for (i = 0; i < nFeat; i++) {
		FeatVector *hFeat = Feat(i);
		wFeat[i] = hFeat->wSplit;
	}
	vector<tpSAMP_ID> idx;
	sort_indexes(nFeat, wFeat, idx);
	for (i = 0; i < nFeat; i++) {
		rank[idx[nFeat-1-i]] = i;
	}
	for (i = 0; i < nFeat-1; i++) {
		if(rank[i]<rank[i+1])
			assert(wFeat[i]>= wFeat[i+1]);
		else
			assert(wFeat[i] <= wFeat[i + 1]);
	}
	return rank;
}

void FeatsOnFold::Feature_Bundling(int flag) {
	size_t i,feat,next,nFeat=feats.size(),maxDup=0,dup;
	if(edaX ==nullptr || edaX->bundle.buns.size()==0)
		return;
	if(config.feat_quanti<=0)
		return;
	vector<vector<int>>& buns = edaX->bundle.buns;
	vector<int> dels;
	for  (auto bun:buns) {
	//r each (vector<int> bun in buns) {
		FeatVector *hBundle = new FeatVec_Bundle(this, feats.size(),bun, edaX->bundle.nMostDup);	//new FeatVector(feats,bun);
		feats.push_back(hBundle);
		for (auto no : bun) {
		//for each(int no in bun) {
			FeatVector *hFeat = Feat(no);
			BIT_SET(hFeat->type, FeatVector::IS_BUNDLE);
			//dels.push_back(no);
		}
	}
	
}

void FeatsOnFold::BeforePredict(int flag) {
	config.task = LiteBOM_Config::kPredict;
}

/*
	基于Y有两种做法 1(histo与X一样，并得到分割)	11(根据当前的预测值来生成histo，并得到分割)
*/
bool LiteBOM_Config::histo_bins_onY()	const {
	//return _histo_bins_Y != 0;
	if (node_task == histo_Y_split_Y)
		throw "!!!histogram_bins onY is ...!!!";
	return node_task == histo_Y_split_Y;
}

/*
*/
void FeatsOnFold::BeforeTrain(BoostingForest *hGBRT, int flag) {
	config.task = LiteBOM_Config::kTrain;
	size_t i,nFeat=feats.size();
	if (edaX != nullptr) {
	}
	bool isFirst = hGBRT->forest.size()==0;
	//if (config.histo_algorithm != LiteBOM_Config::HISTO_ALGORITHM::on_EDA) {
	//printf("\r\n");
	bool isByY = config.histo_bins_onY();
	bool isUpdate = isByY && hGBRT->skdu.noT % 50 == 0;
	isUpdate = hGBRT->skdu.noT>1 ;
	if (isUpdate) {

	}
	size_t nTotalBin0 = 0, nTotalBin1 = 0,nValidFeat=0;
	for (i = 0; i < nFeat; i++) {
		//printf("\rFeatsOnFold::BeforeTrain\tfeat=%d\t......", i);
		FeatVector *hFeat = Feat(i);
		if (hFeat->hDistri != nullptr) {
			if (hFeat->hDistri->histo == nullptr)
			{			continue;			}
			nValidFeat++;
		}
		//FeatVec_Q *hFQ = dynamic_cast<FeatVec_Q *>(hFeat);
		if (hFeat->GetHisto() != nullptr) {
			nTotalBin0 += hFeat->GetHisto()->nBins;
			if (isUpdate) {		//很多原因导致update
				if (isByY) {
					assert(0);
					throw "!!!histogram_bins onY is ...!!!";
					//hFeat->distri.X2Histo_(config, nSamp_, x, Y_);
					hFeat->UpdateHisto(this, true, isFirst, 0x0);
				}
				//else if (hGBRT->stopping.isOscillate && hFeat->wSplit_last>64 && !hFeat->hDistri->isUnique) {
				else if(config.isDynamicHisto) {
					//if (/*hFeat->wSplit_last>1024 &&*/ !hFeat->hDistri->isUnique && !BIT_TEST(hFeat->hDistri->type, Distribution::CATEGORY)) {					
					bool isDiscrete = hFeat->hDistri->isUnique || BIT_TEST(hFeat->hDistri->type, Distribution::CATEGORY);
					bool isUpdate = hFeat->wSplit_last > 1024;//future-sales,geotab等比赛验证，确实有效诶，但是。。。
					if (isUpdate && !isDiscrete) {
						hFeat->hDistri->UpdateHistoByW(this->config, hGBRT->forest.size(), hFeat->wBins);
						//GST_TIC(t1);
						hFeat->UpdateHisto(this, false, isFirst, 0x0);
						//FeatsOnFold::stat.tX += GST_TOC(t1);
					}/**/
				}
			}
		}
		nTotalBin1 += hFeat->GetHisto()->nBins;
		//hFeat->XY2Histo_(config, this->samp_set, x);
	}
	if(hGBRT->skdu.noT%50==0 && nTotalBin1!=nTotalBin0)
		printf("Total Bins=[%d,%d,%.4g]\r\n", nTotalBin0, nTotalBin1, nTotalBin1*1.0/ nValidFeat);

	
}

void FeatsOnFold::PickSample_GH( MT_BiSplit*hBlit,int flag ) {
	const SAMP_SET&samp_set = hBlit->samp_set;
	size_t nSamp = samp_set.nSamp;
	G_INT_64 i;
	tpSAMP_ID samp, *samps = samp_set.samps;
	tpDOWN *hessian = GetHessian();
	tpDOWN *down = GetDownDirection();
	tpDOWN *s_hessian = GetSampleHessian();
	tpDOWN *s_down = GetSampleDown();
#pragma omp parallel for schedule(static)
	for (i = 0; i < nSamp; i++) {
		s_down[i] = down[samps[i]];
	}
	if (hessian != nullptr) {
#pragma omp parallel for schedule(static)
		for (i = 0; i < nSamp; i++) {
			s_hessian[i] = hessian[samps[i]];
		}
	}
}

/*
	alpha-快速测试

void FeatsOnFold::Compress(int flag) {
	size_t i,j, nFeat = feats.size(), nSamp=this->nSample(),nSame=0,cur;
	double *weight_1 = new double[nSamp](), *weight_2 = new double[nSamp]();
	for (auto hFeat : feats) {
		FeatVec_Q *hFQ = dynamic_cast<FeatVec_Q *>(hFeat);
		if (hFQ == nullptr)		continue;
		tpQUANTI* quanti = hFQ->arr();
		for (i = 0; i < nSamp; i++) {
			weight_1[i] += quanti[i];
			weight_2[i] += quanti[i]*(i+1);
		}
	}
	//order by weight_1
	vector<tpSAMP_ID> idx;
	sort_indexes(nSamp,weight_1, idx);
	i = 0;
	while (i < nSamp-1 ) {
		j = i;		cur = idx[i];
		while (++j < nSamp) {
			if (weight_1[idx[j]] == weight_1[cur] && weight_2[idx[j]] == weight_2[cur]) {
				nSame++;
			}
			else
				break;
		}
		i = j;
	}
	delete[] weight_1;		delete[] weight_2;
}*/

void FeatsOnFold::AfterTrain(int flag) {
	//lossy.predict->BinaryOperate(lossy.y, FeatVector::COPY_MEAN, 0x0);
}

/* https://www.codeproject.com/Articles/75423/Loop-Unrolling-over-Template-Arguments  LOOP_unroll might get slower (program code doesn't fit into the L1 cache).
template <int DIM>
struct LOOP_unroll {
	template<typename Operation>
	inline void operator(Operation& op) { op(); ToHisto_unroll<N - 1>(op); }
};
template <>
struct LOOP_unroll<0> {
	template<typename Operation>
	inline void operator(Operation& op) { op();  }
};*/

//https://stackoverflow.com/questions/18971401/sparse-array-compression-using-simd-avx2
#define TO_BIN_0(pBins,quanti,samps,down,i)	{	HISTO_BIN *pB0 = pBins + (quanti[samps[i]]);	pB0->G_sum -= down[i];	pB0->nz++;	}
#define TO_BIN_01(pBins,quanti,samps,down)	{	HISTO_BIN *pB0 = pBins + (quanti[*(samps)]);	pB0->G_sum -= *(down);	pB0->nz++;	}
/*
void FeatVec_Q::Samp2Histo_null_hessian(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, int flag0) {
	HistoGRAM *qHisto = GetHisto();
	size_t nSamp = samp_set.nSamp, i,step;
	int nBin = qHisto->nBins, *nzs = new int[nBin](),pos, num_threads = OMP_FOR_STATIC_1(nSamp, step);;
	double *G_sums = new double[nBin]();

	tpDOWN *down = hData_->GetSampleDown();
	if (nSamp == hData_->nSample()) {
		down = hData_->GetDownDirection();
	}
	const tpSAMP_ID *samps = samp_set.samps;
	tpQUANTI *quanti = arr(), no;
	histo->CopyBins(*qHisto, true, 0x0);
	HISTO_BIN *pBins = histo->bins;
	GST_TIC(t1);


#pragma omp parallel for schedule(static,1) 
	//for (i = 0; i < nSamp; i++, samps++, down++) {
	for (int thread = 0; thread < num_threads; thread++) {
		size_t start = thread*step, end = MIN2(start + step, nSamp), i;
		const tpSAMP_ID *samps_T = samp_set.samps + start;
		tpDOWN *down_T = down + start;
		for (i = start; i < end; i++, samps_T++, down_T++) {
			pos = quanti[*(samps_T)];
			G_sums[pos] -= *(down_T);	nzs[pos]++;
		}
	}

	FeatsOnFold::stat.tX += GST_TOC(t1);
	for (i = 0; i < nBin; i++) {
		pBins[i].G_sum = G_sums[i];
		pBins[i].nz = nzs[i];
		pBins[i].H_sum = pBins[i].nz;
	}	
	delete[] nzs;		delete[] G_sums;
}*/

/*
void FeatVec_Q::Samp2Histo_null_hessian_sparse(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, int flag0) {
	HistoGRAM *qHisto = GetHisto();

	tpDOWN *down = hData_->GetSampleDown();
	string optimal = hData_->config.leaf_optimal;
	bool isLambda = optimal == "lambda_0";
	size_t nSamp = samp_set.nSamp, i,nz=0;
	if (nSamp == hData_->nSample()) {
		down = hData_->GetDownDirection();
	}
	const tpSAMP_ID *samps = samp_set.samps;
	tpSAMP_ID samp;
	tpDOWN a=0;
	tpQUANTI *quanti = arr(), no,cur=0,next=-1;
	histo->CopyBins(*qHisto, true, 0x0);
	int nBin = histo->nBins;	// bins.size();
	HISTO_BIN *pBins = histo->bins, *pBin;	//https://stackoverflow.com/questions/7377773/how-can-i-get-a-pointer-to-the-first-element-in-an-stdvector
	i = 0;		next = quanti[samps[0]];
	while (i<nSamp) {
		cur = next;
		a = down[i];			nz = 1;
		while (++i<nSamp && (next = quanti[samps[i]])==cur) {
			a += down[i];		nz++;
		}
		//assert(pos >= 0 && pos < nBin);
		pBin = pBins + cur;	//HISTO_BIN& bin = histo->bins[no];
		pBin->G_sum += -a;				pBin->nz += nz;			
	}
	//FeatsOnFold::stat.tX += GST_TOC(t1);

	for (i = 0; i < nBin; i++) {
		pBins[i].H_sum = pBins[i].nz;
	}
}*/

/*
	测试数据也量化之后，在测试集上已无意义.		需要重新设计

void FeatVec_Q::PerturbeHisto(const FeatsOnFold *hData_, int flag) {
	if (qHisto_1 != nullptr) {
		delete qHisto_1;
	}
	qHisto_1 = new HistoGRAM(this, qHisto_0->nSamp);
	qHisto_1->CopyBins(*qHisto_0, true, 0x0);
	//qHisto_1->RandomCompress();
	int nBin = qHisto_0->nBins - 1,i;
	HISTO_BIN *cur = nullptr;

	for(i=1;i<nBin;i++){
		//double T0 = qHisto_0->bins[i - 1].split_F, T1 = qHisto_0->bins[i].split_F, T2 = qHisto_0->bins[i + 1].split_F;
		//cur = &(qHisto_1->bins[i]);
		//测试数据也量化之后，在测试集上已无意义
		//cur->split_F = kk == 0 ? T1 - (T1 - T0) *30 : T1 + (T2 - T1) *30;
	}
}*/

/*
void FeatVec_Q::InitSampHisto(HistoGRAM* histo, bool isRandom, int flag) {
	if (qHisto_0->nBins == 0) {
		histo->ReSet(0);	return;
	}	else {
		histo->CopyBins(*qHisto_0, true, 0x0);
	}
	if (false) {
		//assert(qHisto_0->bins.size() > 0);
		//histo->CopyBins(*qHisto_1, true, 0x0);		//变化2 
		//histo->CompressBins();
		histo->RandomCompress(this,false);					//变化1 
	}
}*/

/*
size_t FeatVec_Q::UniqueCount(const SAMP_SET&samp_set, int flag) {
	size_t i, nSamp = samp_set.nSamp, nUnique;
	tpQUANTI *quanti = arr(),no;
	const tpSAMP_ID *samps = samp_set.samps;
	set<int>mapu; 
	for (i = 0; i < nSamp; i += 4) {
		no = quanti[samps[i]];
		mapu.insert(no);
	}
	nUnique = mapu.size();
	return nUnique;
}*/


/*
	参见InitBundle之符号分析
	v0.1	cys
		10/19/2013
*/
FeatVec_Bundle::FeatVec_Bundle(FeatsOnFold *hData_,int id_, const vector<int>&bun, size_t nMostDup, int flag) {
	/*id=id_;
	//const SAMP_SET&samp_set = hData_->samp_set;
	size_t nSamp = hData_->nSample(),nnz = 0, i, nMerge = 0,stp=0,off=0, nDup=0;
	string optimal = hData_->config.leaf_optimal;
	//qHisto = optimal == "grad_variance" ? new HistoGRAM(nSamp) : new Histo_CTQ(nSamp);
	qHisto = new HistoGRAM(this,nSamp);
	vector<HISTO_BIN>& bins = qHisto->bins;
	assert(val == nullptr);
	//val.resize(nSamp);
	tpQUANTI *quanti = arr(), no,*quanti_0=nullptr;
	for (i = 0; i < nSamp; i++) {	quanti[i]=-1;	}
	for (auto no : bun) {
	//for each (int no in bun) {
		off = bins.size();
		FeatVector *hF = hData_->Feat(no);
		if (hF->hDistri->isPass())
			continue;
		FeatVec_Q *hFQ = dynamic_cast<FeatVec_Q *>(hF);
		if (hFQ == nullptr)
			throw "FeatVec_Bundle hFQ is 0!!!";
		//assert(hF->s);
		quanti_0 = hFQ->arr();
		for (i = 0; i < nSamp; i++) {
			if (quanti_0[i] < 0) {
				continue;
			}
			if(quanti[i]>=0)
				nDup++;
			quanti[i] = quanti_0[i] + off;
		}
		HistoGRAM *nextHisto = hFQ->GetHisto();
		size_t nNormalBin=0;
		for (auto bin : nextHisto->bins) {
		//for each(HISTO_BIN bin in nextHisto->bins) {
			//if(bin.nz==0)		continue;
			if (bin.nz > 0)
				nNormalBin++;
			bin.tic += off;
			bins.push_back(bin);
			feat_ids.push_back(hFQ->id);
		}
		//assert(nNormalBin>0);
		//bins.insert(bins.end(), nextHisto->bins.begin(), nextHisto->bins.end());
		//vThrsh.insert(vThrsh.end(), hFQ->vThrsh.begin(), hFQ->vThrsh.end());
		//for( i= 0;i<hFQ->vThrsh.size();i++)			feat_ids.push_back(hFQ->id);
		nMerge++;
	}
	assert(nDup<=nMostDup*nMerge);
	printf( "\n-------- Feat(Bundle)_%d nMerge=%d[%d-%d] dup=%.3g \n", this->id, nMerge, feat_ids[0], feat_ids[bins.size()-1], nDup*1.0/(nMostDup*nMerge) );
	if(nDup > nMostDup*nMerge)
		throw("!!!FeatVec_Bundle nDup > nMostDup*nMerge!!!");*/
}

/*
void FeatVec_Bundle::Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* hParent, HistoGRAM* histo, int nMostBin,  int flag0) {
	if (qHisto->nBins == 0) {
		histo->ReSet(0);
		return;
	}
	tpDOWN *err = nullptr;
	size_t nSamp = samp_set.nSamp, i;
	const tpSAMP_ID *samps = samp_set.samps;
	tpSAMP_ID samp;
	tpDOWN a;
	tpQUANTI *quanti = arr(), no;
	histo->CopyBins(*qHisto, true,0x0);
	//histo->nSamp = nSamp;
	HISTO_BIN *pBins = histo->bins, *pBin;	//https://stackoverflow.com/questions/7377773/how-can-i-get-a-pointer-to-the-first-element-in-an-stdvector
	for (i = 0; i<nSamp; i++) {
		samp = samps[i];
		//no = quanti[samp];
		a = err[samp];
		{
			if( a!=0 )
				samp = samps[i];
			pBin = pBins + quanti[samp];	//HISTO_BIN& bin = histo->bins[no];

		}
		pBin->G_sum += a;
		pBin->nz++;
	}
}*/


/*
FeatVec_Q::FeatVec_Q(const FeatsOnFold *hData_, FeatVector *hFeat,int nMostBin, int flag) : hFeatSource(hFeat){
	id = hFeat->id;
	desc = hFeat->desc;	
	nam = hFeat->nam;
	type = hFeat->type;
	hDistri = hFeatSource->hDistri;
}*/


/*
	参见lightGBM::BoostFromScore
	v0.1	cys
		11/8/2018
*/
void INIT_SCORE::Init(FeatsOnFold *hData_, int flag) {
	LiteBOM_Config& config = hData_->config;
	Distribution *yDis = hData_->GetY()->hDistri;
	assert(yDis !=nullptr);
	double mean = yDis->mean;
	//hLoss->predict->Set(mean);
	//size_t nSamp=down.size(),i;
	if(config.init_scor=="mean")
		step = mean;
	else //config.init_scor == "0")
		step = 0;
	printf("----Start training from score %g",step);
	//if(hData_->config.eval_metric == "auc"){
	if (hData_->config.objective == "binary") {
		double sigmoid_ = 1, a = step;
		double score = std::log(a / (1.0f - a)) / sigmoid_;
		step = score;
		printf("---->%g", step);
	}
	printf("\n");
}



FeatVector* FeatsOnFold::GetPrecict() {
	assert(lossy!=nullptr);
	return lossy->predict;
}
FeatVector* FeatsOnFold::GetY() {
	assert(lossy != nullptr);
	return lossy->GetY();
}

//pDown=target-predict
tpDOWN *FeatsOnFold::GetDownDirection() const {
	assert(lossy != nullptr);
	return lossy->GetDownDirection();
}
tpDOWN *FeatsOnFold::GetDeltaStep() const {
	assert(lossy != nullptr);
	return lossy->GetDeltaStep();
}
tpDOWN *FeatsOnFold::GetHessian() const {
	assert(lossy != nullptr);
	if (lossy->hessian.size() == 0)
		return nullptr;
	else
		return VECTOR2ARR(lossy->hessian);
}

tpDOWN *FeatsOnFold::GetSampleDown() const {
	assert(lossy != nullptr);
	return lossy->GetSampleDown();
}
tpDOWN *FeatsOnFold::GetSampleHessian() const {
	assert(lossy != nullptr);
	if (lossy->sample_hessian.size() == 0)
		return nullptr;
	else
		return VECTOR2ARR(lossy->sample_hessian);
}

int *FeatsOnFold::Tag() { return lossy->Tag(); }


/*
	v0.1 cys
		11/15/2019
*/
void FeatsOnFold::ExpandMerge(const vector<FeatsOnFold *>&merge_folds, int flag) {
	int i,nFeat,nMerge=0,nExFeat=0;
	assert(merge_lefts.size() == merge_folds.size());
	for (auto fold : merge_folds) {
		assert( BIT_TEST(fold->dType, FeatsOnFold::DF_MERGE) );
		FeatVector *hLeft = this->merge_lefts[nMerge++];
		//FeatVector *hRight = fold->Feat(fold->merge_right);		assert(hRight != nullptr);
		hLeft->Merge4Quanti(nullptr, 0x0);
		SAMP_SET samp1(hLeft->size(), hLeft->map4set);
		for (auto hFeat : fold->feats) {
			if (nExFeat == 0)
				i = 0;
			FeatVector *hRight = hFeat;
			if (isEval()) {
				assert(hRight->hDistri!=nullptr);		//already in ExpandMerge@train
			}else
				hFeat->EDA(config, true, &samp1, 0x0);

			if (fold->isQuanti || hFeat->isCategory()) {
				//assert(isTrain());
				FeatVector *hFQ = FeatVecQ_InitInstance(fold, hFeat, 0x0);	// new FeatVec_Q<short>(hFold, hFeat, nMostQ);
				hRight = hFQ;	//delete hFeat;
			}

			FeatVector *hEXP = nullptr;
			if(hLeft->PY->isInt8())
				hEXP = new FeatVec_EXP<uint8_t>(this, hRight->nam + "@" + hLeft->nam,hLeft, hRight);
			else if(hLeft->PY->isInt16())
				hEXP = new FeatVec_EXP<uint16_t>(this, hRight->nam + "@" + hLeft->nam, hLeft, hRight);
			else {
				assert(hLeft->PY->isInt32());
				hEXP = new FeatVec_EXP<int32_t>(this, hRight->nam + "@" + hLeft->nam, hLeft, hRight);
			}

			hEXP->EDA(this->config, false, nullptr, 0x0);
			feats.push_back(hEXP);
			nExFeat++;
		}
	}	
}

//核心函数 
void FeatsOnFold::SplitOn(MT_BiSplit *hBlit, int flag) {
	FeatVector *hF_ = Feat(hBlit->feat_id);
	assert(hBlit->samp_set.nSamp <= hF_->size());
	//if (hBlit->samp_set.nSamp == 139)	//仅用于调试
	//	int i = 0;
	hF_->Value_AtSamp(&hBlit->samp_set, GetSampleValues());
	hF_->SplitOn(this, hBlit);
}

void FeatsOnFold::ExpandFeat(int flag) {
	return;
	/*	bool isTrain = BIT_TEST(dType,  FeatsOnFold::DF_TRAIN);
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
	}*/
}

/*
void FeatVec_Q::UpdateFruit(const FeatsOnFold *hData_, MT_BiSplit *hBlit, int flag) {
	//double split = hBlit->fruit->thrshold;
	if (hBlit->fruit->isY) {
		//vector<HISTO_BIN>& bins=hBlit->fruit->histo->bins;
		if (this->isCategory()) {
			const HistoGRAM *histo = hBlit->fruit->histo_refer;
			assert(histo!=nullptr && hBlit->fruit->mapFolds!=nullptr);
			//hBlit->fruit->mapFold = this->hDistri->mapCategory;
			for (int i = 0; i < histo->nBins; i++) {
				int pos = histo->bins[i].tic,fold= histo->bins[i].fold;
				hBlit->fruit->mapFolds[pos] = fold;
				//hBlit->fruit->mapFold.insert(pair<int, int>(pos, fold));
			}
		}
		else {
					}
		//hBlit->fruit->T_quanti = -13;
	}
	else {
		if (hData_->config.split_refine != LiteBOM_Config::REFINE_SPLIT::REFINE_NONE)
			hFeatSource->RefineThrsh(hData_, hBlit);
	}
}*/

/*
https://ask.julyedu.com/question/7603

	- 首先对原数据排序；
	- 然后统计出distinct value 以及对应的 count；
	- 如果distinct value数目小于max bin数，则正好每个value放入一个bin；
	- 如果distinct value大于max bin，计算bin大小的均值，按一定规则将sample放入bin，保证相同value的放入同一bin，bin包含的sample数尽量平均。
	- 注：max bin的默认值是256。

	对于category类型的feature，则是每一种取值放入一个bin，且当取值的个数大于max bin数时，会忽略那些很少出现的category值。
	在求split时，对于category类型的feature，算的是"按是否属于某个category值划分"的gain，这和数值型feature是完全不同的，它的实际效果就是类似one-hot的编码方法。
*/