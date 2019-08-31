#pragma once

#include "./DataFold.hpp"
#include "./Binfold.hpp"
using namespace Grusoft;

/*
	参见 HistoGRAM::GreedySplit_X
*/
void BinFold::GreedySplit(const FeatsOnFold *hData_, int flag) {
	int i, jj;
	for (i = 0; i < nFeat; i++) {
		for (jj = 0; jj < nQuanti; jj++) {
		}	
	}
}

/*
	参见FeatVec_Q::Samp2Histo
*/
BinFold::BinFold(const FeatsOnFold *hData_, const vector<int> &pick_feats, const SAMP_SET&samp_set, int flag) {
	size_t nSamp = samp_set.nSamp, i, nFeat= pick_feats.size(),t,pos;
	ldQ = hData_->config.feat_quanti + 2;	//预留nan的位置
	bGrad = new BINFOLD_FLOAT[nFeat*ldQ]();
	bHess = new BINFOLD_FLOAT[nFeat*ldQ]();
	NZ = new size_t[nFeat*ldQ]();
	size_t *curNZ;
	const tpDOWN *hessian = hData_->GetHessian();
	const tpDOWN *down = hData_->GetDownDirection();
	const tpSAMP_ID *samps = samp_set.samps;
	tpSAMP_ID samp;
	double g,hess,*curGrad,*curHess;
	tpQUANTI **arrFeat = new tpQUANTI*[nFeat],*feat, *val = nullptr,bin;
	for (t = 0; t < nFeat; t++) {
		FeatVec_T<tpQUANTI>*hFT = dynamic_cast<FeatVec_T<tpQUANTI>*>(hData_->feats[pick_feats[t]]);
		if (hFT == nullptr)		{
			delete[] arrFeat;	return ;
		}
		arrFeat[t] = hFT->arr();
	}
	for (t = 0; t < nFeat; t++) {		//实测t外层要快很多
		feat = arrFeat[t];
		curNZ = NZ + t*ldQ;
		curGrad = bGrad + t*ldQ;		curHess = bHess + t*ldQ;
		for (i = 0; i<nSamp; i++) {
			samp = samps[i];		
			g = down[samp],hess= hessian==nullptr ? 1 : hessian[samp];
			bin = feat[samp];
			//pos = t*ldQ + bin;
			/*if (quanti[samp]<0)	//Nan
				pBin = &(histo->binNA);
			else
				pBin = pBins + quanti[samp];	//HISTO_BIN& bin = histo->bins[no];*/
			curGrad[bin] += -g;			curHess[bin] += hess;
			curNZ[bin]++;
		}
	}
	delete[] arrFeat;
}