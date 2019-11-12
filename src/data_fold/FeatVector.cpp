#pragma once

#include "./FeatVec_EXP.hpp"
using namespace Grusoft;

void FeatVec_EXP::EDA(const LiteBOM_Config&config, bool genHisto, int flag) {
	size_t i,nSamp_= nSamp();
	assert(hDistri == nullptr);
	hDistri = new Distribution();
	*hDistri = *hRight->hDistri;
	hDistri->nam = nam;

	HistoGRAM *histo_right = hRight->hDistri->histo;
	assert(histo_right!=nullptr);
	hDistri->histo = new HistoGRAM(this,nSamp_,0x0);
	hDistri->histo->CopyBins(*histo_right,true,0x0);
	//hDistri->STA_at(nSamp_, val, true, 0x0);
	if (ZERO_DEVIA(hDistri->vMin, hDistri->vMax))
		BIT_SET(this->type, Distribution::V_ZERO_DEVIA);	
	
}

void FeatVec_EXP::Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_0, HistoGRAM* histo, int nMostBin, const tpSAMP_ID *samps4quanti, int flag0)	const {
	//SAMP_SET samp_1;
	//size_t nRightSamp = hRight->nSamp();
	//samp_1.Alloc(nRightSamp);
	hRight->Samp2Histo(hData_, samp_0, histo, nMostBin, hLeft->samp4quanti, flag0);
}

void FeatVec_EXP::SplitOn(FeatsOnFold *hData_, MT_BiSplit *hBest, int flag) {
	//Expand 
	hRight->Merge4Quanti(nullptr, 0x0);
	//void *hOldVal = hRight->ExpandOnSamp(hLeft->samp4quanti);
	hRight->SplitOn(hData_, hBest, flag);
	//hRight->CloseOnSamp(hOldVal,hLeft->samp4quanti);
}