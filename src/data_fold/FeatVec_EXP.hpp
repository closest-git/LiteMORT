#pragma once

#include "./DataFold.hpp"

namespace Grusoft {
	class FeatVec_EXP : public FeatVector {
	protected:
		//FeatVec_T<short> *featX = nullptr, *featY = nullptr;
		const FeatVector *hLeft = nullptr;		//½öÖ¸Ïò
		FeatVector *hRight = nullptr;		
	public:
		FeatVec_EXP(FeatsOnFold *hData_,string nam_, const FeatVector *hL_, FeatVector *hR_, int flag = 0x0):hLeft(hL_), hRight(hR_){
			this->nam = nam_;
			this->id = hData_->nFeat();
		}
		virtual ~FeatVec_EXP() {
		}
		virtual bool isMerged()	const { return	true; }

		virtual size_t nSamp() const	{ return hLeft->nSamp(); }
		virtual HistoGRAM *GetHisto(int flag = 0x0) { 
			assert(hDistri!=nullptr && hDistri->histo != nullptr);
			return hDistri->histo; 
		}

		virtual void EDA(const LiteBOM_Config&config, bool genHisto, int flag);
		virtual void SplitOn(FeatsOnFold *hData_, MT_BiSplit *hBest, int flag = 0x0);
		virtual void Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, const tpSAMP_ID *samps4quanti = nullptr, int flag0 = 0x0)		const;

	};

}

