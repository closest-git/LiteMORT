#pragma once

#include "./DataFold.hpp"

namespace Grusoft {
	//FeatVec_EXP的设计不是很合理
	template<typename tpLEFT>
	class FeatVec_EXP : public FeatVector {
		const tpLEFT *left_maps=nullptr;		//仅仅指向	
	protected:
		//FeatVec_T<short> *featX = nullptr, *featY = nullptr;
		FeatVec_T<tpLEFT> *hLeft = nullptr;		//仅指向
		FeatVector *hRight = nullptr;		
	public:
		FeatVec_EXP(FeatsOnFold *hData_,string nam_, FeatVector *hL_, FeatVector *hR_, int flag = 0x0): hRight(hR_){
			this->nam = nam_;
			this->id = hData_->nFeat();
			hLeft = dynamic_cast<FeatVec_T<tpLEFT>*>(hL_);
			assert(hLeft != nullptr);
			left_maps = hLeft->arr();

			PY = hRight->PY;
			UpdateType();
		}

		virtual ~FeatVec_EXP() {
		}
		virtual bool isMerged()	const { return	true; }

		virtual inline size_t size() const	{ return hLeft->size(); }
		/*virtual HistoGRAM *GetHisto(int flag = 0x0) { 
			assert(hDistri!=nullptr && hDistri->histo != nullptr);
			return hDistri->histo; 
		}*/

		virtual void Value_AtSamp(const SAMP_SET*samp_set, void *samp_values, int flag = 0x0) {
			hLeft->Merge4Quanti(samp_set, 0x0);
			size_t nSamp = samp_set == nullptr ? hLeft->size(): samp_set->nSamp;
			SAMP_SET samp1(nSamp, hLeft->map4set);
			hRight->Value_AtSamp(&samp1, samp_values);
		}

		virtual inline void* pValue_AtSamp(const size_t&samp_0, int flag = 0x0) {
			tpLEFT pos_R = left_maps[samp_0];
			if (IS_NAN_INF(pos_R)) {
				return nullptr;
			}	else {
				return hRight->pValue_AtSamp((size_t)pos_R);
			}
		}

		/*
		virtual void EDA(const LiteBOM_Config&config, bool genHisto, const SAMP_SET *samp_set, int flag) {
			assert(samp_set==nullptr);
			size_t i, nSamp_ = size();
			assert(hDistri == nullptr);
			hDistri = new Distribution();
			*hDistri = *hRight->hDistri;
			hDistri->nam = nam;

			HistoGRAM *histo_right = hRight->hDistri->histo;
			assert(histo_right != nullptr);
			hDistri->histo = new HistoGRAM(this, nSamp_, 0x0);
			hDistri->histo->CopyBins(*histo_right, true, 0x0);
			//hDistri->STA_at(nSamp_, val, true, 0x0);
			if (ZERO_DEVIA(hDistri->vMin, hDistri->vMax))
				BIT_SET(this->type, Distribution::V_ZERO_DEVIA);
		}*/

		virtual void SplitOn(FeatsOnFold *hData_, MT_BiSplit *hBest, int flag = 0x0) {
			hRight->SplitOn(hData_, hBest, flag);
		}

		virtual inline int left_rigt(const void *pVal, const ARR_TREE*arr_tree, int no, int flag = 0x0) {
			return hRight->left_rigt(pVal, arr_tree,no, flag);
		}

		//每个feat的binNA都不一样
		virtual void Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_0, HistoGRAM* histo, int nMostBin, const tpSAMP_ID *samps4quanti = nullptr, int flag0 = 0x0)	const {
			assert(samps4quanti==nullptr);
			size_t nSamp = samp_0.nSamp, i,nRight=hRight->size(),nNA=0;
			Distribution *rhDistri = hData_->histoDistri(hRight);
			const HistoGRAM *qHisto = rhDistri->histo;		assert(qHisto!=nullptr);
			int nBin = qHisto->nBins;
			/*tpSAMP_ID *map4feat = hLeft->map4feat;
			memcpy(map4feat, hLeft->map4set,sizeof(tpSAMP_ID)*nSamp);
			for (i = 0; i < nSamp; i++) {
				if (map4feat[i] == tpSAMP_ID_NAN) {
					map4feat[i] = nRight - 1;		//always last bin for NAN
					assert(0);
				}
				assert(map4feat[i] >= 0 && map4feat[i]<nRight);
				if (map4feat[i] == nRight - 1)
					nNA++;
			}*/
			hRight->Samp2Histo(hData_, samp_0, histo, nMostBin, hLeft->map4set, flag0);
		}

	};

}

