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
优雅，易读		参考PYTHON,	没必要以来C++的莫名语法

*/
FeatsOnFold::FeatsOnFold(LiteBOM_Config confi_, ExploreDA *eda_, string nam_, int dtype) :config(confi_), edaX(eda_), nam(nam_) {
	dType = dtype;
	isQuanti = config.feat_quanti >0 && BIT_TEST(dType, FeatsOnFold::DF_TRAIN);	//BIT_TEST(dType, FAST_QUANTI);
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
void FeatsOnFold::nPick4Split(vector<int>&picks, GRander&rander, BoostingForest *hForest, int flag) {
	int i, nFeat = feats.size(), nPick = (int)(sqrt(nFeat));
	int nTree = hForest->forest.size();
	int *mask = new int[nFeat]();
	//picks.resize(nFeat);
	for (i = 0; i<nFeat; i++)	{
		FeatVector *hFeat = Feat(i);
		if(i==116)
			i = 116;
		if (hFeat->hDistri!=nullptr && hFeat->hDistri->isPass())
			continue;
		if (BIT_TEST(hFeat->type, FeatVector::IS_BUNDLE))
			continue;
		if (BIT_TEST(hFeat->type, Distribution::DISTRI_OUTSIDE))
			continue;
		if (BIT_TEST(hFeat->type, FeatVector::AGGREGATE)) {
			
		}
		//if(hFeat->id!=360)	continue;	//仅用于测试 
		mask[i] = 1;
		picks.push_back(i);
	}
	assert(picks.size()>0);
	if (config.feature_fraction<1) {	//for random forest
		nPick = MAX(1,picks.size()*config.feature_fraction);
		hForest->stopping.CheckBrae();
		vector<int> no_k = rander.kSampleInN(nPick, picks.size()),pick_1;
		double *w = new double[nFeat]();
		for (auto x : no_k) {
			int no = picks[x];				pick_1.push_back(no);
			assert(mask[no] == 1);
			FeatVector *hFeat = Feat(no);
			w[no] = -hFeat->wSplit;
		}
		picks = pick_1;		
		nPick = picks.size();
		if (config.nElitism>0 && hForest->forest.size()>16) {
			std::sort(picks.begin(), picks.end(), [&w](size_t i1, size_t i2) {return w[i1] < w[i2]; });
			for (i = 0; i < nPick - 1; i++) {
				assert(w[picks[i]] <= w[picks[i + 1]]);
				assert(mask[picks[i]] == 1);
			}
			//int nElitism = min(nPick, 16);
			int nElitism = min(nPick, config.nElitism);
			std::random_shuffle(picks.begin()+ nElitism, picks.end());
			picks.resize(min(nPick, min(nElitism*6, nElitism+nFeat/3)));
		}
		delete[] w;

		std::sort(picks.begin(), picks.end());
	}
	delete[] mask;
	nPickFeat = picks.size();
	assert(nPickFeat>0);
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
			nTotalBin1 += hFeat->hDistri->histo->nBins;
			nValidFeat++;
		}
		FeatVec_Q *hFQ = dynamic_cast<FeatVec_Q *>(hFeat);
		if (hFQ != nullptr) {
			nTotalBin0 += hFQ->GetHisto()->nBins;
			if (isUpdate) {		//很多原因导致update
				if (isByY) {
					assert(0);
					throw "!!!histogram_bins onY is ...!!!";
					//hFeat->distri.X2Histo_(config, nSamp_, x, Y_);
					hFeat->UpdateHisto(this, true, isFirst, 0x0);
				}
				else if(hFeat->wSplit_last>1024 && !hFeat->hDistri->isUnique){
					hFeat->hDistri->UpdateHistoByW(this->config,hFeat->wBins);
					//GST_TIC(t1);
					hFeat->UpdateHisto(this, false,isFirst, 0x0);
					//FeatsOnFold::stat.tX += GST_TOC(t1);
				}
			}
		}
		//hFeat->XY2Histo_(config, this->samp_set, x);
	}
	if(hGBRT->skdu.noT==0)
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
*/
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
}

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

#define TO_BIN_0(pBins,quanti,samps,down,i)	{	HISTO_BIN *pB0 = pBins + (quanti[samps[i]]);	pB0->G_sum -= down[i];	pB0->nz++;	}
void FeatVec_Q::Samp2Histo_null_hessian(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, int flag0) {
	HistoGRAM *qHisto = GetHisto();

	tpDOWN *down = hData_->GetSampleDown();	
	string optimal = hData_->config.leaf_optimal;
	bool isLambda = optimal == "lambda_0";
	size_t nSamp = samp_set.nSamp, i, nSamp_LD= 0,LD=4;
	if (nSamp == hData_->nSample()) {
		down = hData_->GetDownDirection();
	}
	const tpSAMP_ID *samps = samp_set.samps;
	tpSAMP_ID samp;
	tpDOWN a;
	tpQUANTI *quanti = arr(), no;
	histo->CopyBins(*qHisto, true, 0x0);
	int nBin = histo->nBins;// bins.size();
	HISTO_BIN *pBins = histo->bins, *pBin;	//https://stackoverflow.com/questions/7377773/how-can-i-get-a-pointer-to-the-first-element-in-an-stdvector

	nSamp_LD = LD==0 ? 0 : LD * (int)(nSamp / LD);
	for (i = 0; i < nSamp_LD; i += LD) {
		TO_BIN_0(pBins, quanti,samps, down,i);
		TO_BIN_0(pBins, quanti, samps, down, i+1);
		TO_BIN_0(pBins, quanti, samps, down, i+2);
		TO_BIN_0(pBins, quanti, samps, down, i+3);
		/*TO_BIN_0(pBins, quanti, samps, down, i+4);
		TO_BIN_0(pBins, quanti, samps, down, i + 5);
		TO_BIN_0(pBins, quanti, samps, down, i + 6);
		TO_BIN_0(pBins, quanti, samps, down, i + 7);
		TO_BIN_0(pBins, quanti, samps, down, i + 8);
		TO_BIN_0(pBins, quanti, samps, down, i + 9);
		TO_BIN_0(pBins, quanti, samps, down, i + 10);
		TO_BIN_0(pBins, quanti, samps, down, i + 11);*/
	}
	 //if(nSamp<10000)
	for (i = nSamp_LD; i<nSamp; i++) {
		TO_BIN_0(pBins, quanti, samps, down, i);
		/*tpQUANTI pos = quanti[samps[i]];
		assert(pos >= 0 && pos < nBin);
		a = down[i];
		pBin = pBins + pos;	//HISTO_BIN& bin = histo->bins[no];
		pBin->G_sum += -a;
		pBin->nz++;*/
	}
	//FeatsOnFold::stat.tX += GST_TOC(t1);

	for (i = 0; i < nBin; i++) {
		pBins[i].H_sum = pBins[i].nz;
	}
}

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
}

/*
	测试数据也量化之后，在测试集上已无意义.		需要重新设计
*/
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
		double T0 = qHisto_0->bins[i - 1].split_F, T1 = qHisto_0->bins[i].split_F, T2 = qHisto_0->bins[i + 1].split_F;
		cur = &(qHisto_1->bins[i]);
		//测试数据也量化之后，在测试集上已无意义
		//cur->split_F = kk == 0 ? T1 - (T1 - T0) *30 : T1 + (T2 - T1) *30;
	}
}


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
}

/*
	v0.2
*/
void FeatVec_Q::Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* hParent, HistoGRAM* histo, int nMostBin,  int flag0) {
	tpDOWN *hessian = hData_->GetSampleHessian();
	if (hessian == nullptr) {
		Samp2Histo_null_hessian(hData_, samp_set, histo, nMostBin, flag0);
		//Samp2Histo_null_hessian_sparse(hData_, samp_set, histo, nMostBin, flag0);
	}
	else {
		//histo->nSamp = samp_set.nSamp;
		tpQUANTI *quanti = arr(), no, *map = nullptr;
		/*if (hParent != nullptr) {
			histo->CopyBins(*hParent, true, 0x0);
		}else*/
			InitSampHisto(histo, false);
		if (histo->nBins == 0) {
			return;
		}
		tpDOWN *down = hData_->GetSampleDown();	
		string optimal = hData_->config.leaf_optimal;
		bool isLambda = optimal == "lambda_0";
		size_t nSamp = samp_set.nSamp, i, nSamp4 = 0;
		if (nSamp == hData_->nSample()) {
			hessian = hData_->GetHessian();
			down = hData_->GetDownDirection();
		}
		const tpSAMP_ID *samps = samp_set.samps;
		tpSAMP_ID samp;
		tpDOWN a;
		//histo->CopyBins(*qHisto, true, 0x0);
		int nBin = histo->nBins;//bins.size();
		HISTO_BIN *pBins = histo->bins,*pBin;	//https://stackoverflow.com/questions/7377773/how-can-i-get-a-pointer-to-the-first-element-in-an-stdvector
		/*if (hParent != nullptr && histo->bins.size()<qHisto_0->bins.size()) {
			map = new tpQUANTI[qHisto_0->bins.size()];		//晕，无效的尝试
			hParent->TicMap(map, 0x0);
			for (i = 0; i<nSamp; i++) {
				tpQUANTI pos = map[quanti[samps[i]]];
				assert(pos >= 0 && pos < nBin);
				a = down[i];
				pBin = pBins + pos;	//HISTO_BIN& bin = histo->bins[no];
				pBin->G_sum += -a;			pBin->H_sum += hessian[i];
				//pBin->H_sum += hessian == nullptr ? 1 : hessian[samp];
				pBin->nz++;
			}
			delete[] map;
		}
		else*/ {		//主要的时间瓶颈
			nSamp4 =  4 * (int)(nSamp / 4);
			for (i=0; i < nSamp4; i += 4) {
				HISTO_BIN *pB0 = pBins + quanti[samps[i]];
				HISTO_BIN *pB1 = pBins + quanti[samps[i+1]];
				HISTO_BIN *pB2 = pBins + quanti[samps[i+2]];
				HISTO_BIN *pB3 = pBins + quanti[samps[i+3]];
				tpDOWN a0 = down[i], a1 = down[i+1], a2 = down[i+2], a3 = down[i+3];
				pB0->G_sum -= a0;			pB1->G_sum -= a1;			pB2->G_sum -= a2;			pB3->G_sum -= a3;
				pB0->H_sum += hessian[i];			pB1->H_sum += hessian[i+1];
				pB2->H_sum += hessian[i+2];			pB3->H_sum += hessian[i+3];
				pB0->nz++;	pB1->nz++;	pB2->nz++;	pB3->nz++;
			}/**/
			//if(nSamp<10000)
			for (i = nSamp4; i<nSamp; i++) {
				tpQUANTI pos = quanti[samps[i]];
				assert(pos >= 0 && pos < nBin);
		
				//a = down[samp];
				a = down[i];
				pBin= pBins+ pos;	//HISTO_BIN& bin = histo->bins[no];
				pBin->G_sum += -a;		
				pBin->H_sum += hessian[i];
				//pBin->H_sum += hessian == nullptr ? 1 : hessian[samp];
				pBin->nz++;
			}	
			//if(hParent==nullptr)
		}
	}
	histo->CheckValid(hData_->config);

	#ifdef _DEBUG
	if (true /* && !isRandomDrop*/) {
		double G_sum = 0;	// histo->hBinNA()->G_sum;
		for (int i = 0; i < histo->nBins; i++) {
		//for (auto item : histo->bins) {
			G_sum += histo->bins[i].G_sum;
		}
		assert(fabs(G_sum + samp_set.Y_sum_1)<1.e-7*fabs(G_sum) || fabs(samp_set.Y_sum_1)<0.001);
	}
	#endif

}



/*
	static bin mapping

	v0.1	cys
		10/22/2018
*/
void FeatVec_Q::UpdateHisto(const FeatsOnFold *hData_, bool isOnY, bool isFirst, int flag) {
	if(qHisto_0!=nullptr)
		delete qHisto_0;
	//vThrsh.clear( );
	//const SAMP_SET&samp_set = hData_->samp_set;
	size_t nSamp = hData_->nSample(), i, samp,nValid=0;
	string optimal = hData_->config.leaf_optimal;
	//qHisto = optimal == "grad_variance" ? new HistoGRAM(nSamp) : new Histo_CTQ(nSamp);
	qHisto_0 = new HistoGRAM(this,nSamp);
	FeatVector *hFeat = hFeatSource;
	size_t nMostBin = hData_->config.feat_quanti;
	//bool isOnY = hData_->config.histo_bins_onY();	//hData_->config.histo_algorithm == LiteBOM_Config::HISTO_ALGORITHM::on_Y;
	tpDOWN *yDown = nullptr;		//明显不合理	3/11/2019
	if (isOnY) {
		if (isFirst) {
			//((FeatsOnFold *)hData_)->lossy->Update((FeatsOnFold *)hData_);
		}	
		yDown = ((FeatsOnFold *)hData_)->GetDownDirection();
	}/**/
	//assert(val == nullptr);
	if (val == nullptr) {		
		val = new tpQUANTI[nSamp];
	}
	//val.resize(nSamp);
	tpQUANTI *quanti = arr(), no;
	//qHisto->quanti = quanti;
	for( i=0;i<nSamp;i++ )	quanti[i]=-111;		//-1 for NAN
	
	ExploreDA *edaX = hData_->edaX;
	if (edaX != nullptr /*&& hData_->config.histo_algorithm == LiteBOM_Config::HISTO_ALGORITHM::on_EDA*/) {
		Distribution& distri = edaX->arrDistri[id];
		if (isOnY) {
			throw "!!!histogram_bins onY is ...!!!";
			FeatVec_T<float> *hFeatFloat = dynamic_cast<FeatVec_T<float>*>(hFeatSource);
			float *x = hFeatFloat->arr();
			distri.ClearHisto();
			distri.X2Histo_(hData_->config, nSamp, x, yDown);
		}

		if(distri.histo!=nullptr)
			qHisto_0->CopyBins(*(distri.histo), true, 0x0);
		hFeat->QuantiAtEDA(edaX, quanti, nMostBin, hData_->isTrain(),0x0);
	}	else{
		//hFeat->Split2Quanti(hData_->config,edaX, vThrsh, qHisto, yDown, nMostBin);
		printf( "\n!!! FeatVec_Q::Update_Histo edaX=nullptr !!!\n" );		
		throw "\n!!! FeatVec_Q::Update_Histo edaX=nullptr !!!\n";
	}
	for (nValid=0,i = 0; i < nSamp; i++) {
		if(quanti[i] == -111)
		{	printf("\n!!! FeatVec_Q::Update_Histo quanti[%d] is -111 !!!\n",i);		throw "\n!!! FeatVec_Q::Update_Histo quanti is -1 !!!\n";	}
		if (quanti[i] >=0 )
			nValid++;
	}
	if (nValid == 0) {
		printf("\n FeatVec_Q(%s) nBin=%d a0=%g a1=%g", desc.c_str(), qHisto_0->nBins, 0, -1);
		BIT_SET(this->type, Distribution::DISTRI_OUTSIDE);
	}
	if(hData_->config.nMostSalp4bins>0 && hData_->isTrain())
		select_bins = new FS_gene_(this->nam,hData_->config.nMostSalp4bins, qHisto_0->nBins, 0x0);
	if (wBins != nullptr)
		delete[] wBins;
	wBins = new float[qHisto_0->nBins]();
	wSplit_last = 0;
		//printf("\n FeatVec_Q(%s) nBin=%d a0=%g a1=%g", desc.c_str(),qHisto->bins.size(),qHisto->a0, qHisto->a1 );	
}

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
		/*if (quanti[samp]<0)	
			pBin = &(histo->binNA);
		else*/ {
			if( a!=0 )
				samp = samps[i];
			pBin = pBins + quanti[samp];	//HISTO_BIN& bin = histo->bins[no];

		}
		pBin->G_sum += a;
		pBin->nz++;
	}
}

void FeatVec_Bundle::UpdateFruit(MT_BiSplit *hBlit, int flag) {
	/*double split = hBlit->fruit->thrshold;
	tpQUANTI q_split = split;		assert(q_split == split);
	hBlit->fruit->T_quanti = q_split;
	hBlit->feat_id = feat_ids[q_split];
	//assert(split>a0 && split <= a1);
	float thrsh = vThrsh[q_split];		//严重的BUG之源啊
	hBlit->fruit->thrshold = thrsh;
	printf("\nFeatVec_Bundle::pick bins=%d split=%g thrsh=%g feat=%d\t", feat_ids.size(),split, thrsh,hBlit->feat_id);*/
}

/*
	
*/
FeatVec_Q::FeatVec_Q(const FeatsOnFold *hData_, FeatVector *hFeat,int nMostBin, int flag) : hFeatSource(hFeat){
	id = hFeat->id;
	desc = hFeat->desc;	
	nam = hFeat->nam;
	type = hFeat->type;
	hDistri = hFeatSource->hDistri;
}

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
}

void INIT_SCORE::ToDownStep(int flag) {


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
int  FeatsOnFold::OMP_FOR_STATIC_1(const size_t nSamp, size_t& step, int flag) {
	int num_threads = 1;
	step = nSamp;
#pragma omp parallel	
#pragma omp master											
	{	num_threads = omp_get_num_threads();	}
	step = (nSamp + num_threads - 1) / num_threads;
	return num_threads;
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