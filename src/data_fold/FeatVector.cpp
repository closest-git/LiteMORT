#pragma once

#include "./FeatVec_EXP.hpp"
using namespace Grusoft;

void FeatVector::UpdateType(int flag){
	assert(PY != nullptr);
	if (PY->isCategory()) {
		if(hDistri!=nullptr)	//expanded feat has no hDistri
			BIT_SET(hDistri->type, Distribution::CATEGORY);
		BIT_SET(type, Distribution::CATEGORY);
	}
	if (PY->isDiscrete()) {
		if (hDistri != nullptr)
			BIT_SET(hDistri->type, Distribution::DISCRETE);
		BIT_SET(type, Distribution::DISCRETE);
	}
	if (PY->representive > 0) {
		if (hDistri != nullptr)
			BIT_SET(hDistri->type, Distribution::DISCRETE);
		BIT_SET(type, Distribution::DISCRETE);
		BIT_SET(type, FeatVector::REPRESENT_);
	}
}
/*
void FeatVec_EXP::EDA(const LiteBOM_Config&config, bool genHisto, int flag) {
	size_t i,nSamp_= size();
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

void FeatVec_EXP::Value_AtSamp(const SAMP_SET*samp_set, void *samp_values, int flag){
	hLeft->Merge4Quanti(samp_set,0x0);
	SAMP_SET samp1(samp_set->nSamp,hLeft->samp4quanti);	
	hRight->Value_AtSamp(&samp1, samp_values);
	//hRight->SplitOn(hData_, hBest, flag);
}	


void FeatVec_EXP::SplitOn(FeatsOnFold *hData_, MT_BiSplit *hBest, int flag) {
//hRight->Merge4Quanti(nullptr, 0x0);
	//void *hOldVal = hRight->ExpandOnSamp(hLeft->samp4quanti);
	//hRight->SplitOn(hData_, hBest, flag);
	//hRight->CloseOnSamp(hOldVal,hLeft->samp4quanti);	
	hRight->SplitOn(hData_, hBest, flag);
}*/
