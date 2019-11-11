#pragma once

#include "./DataFold.hpp"

namespace Grusoft {
	class FeatVec_EXP : public FeatVector {
	protected:
		//FeatVec_T<short> *featX = nullptr, *featY = nullptr;
		const FeatVector *hLeft = nullptr, *hRight = nullptr;		//½öÖ¸Ïò
	public:
		FeatVec_EXP(FeatsOnFold *hData_,string nam_, const FeatVector *hL_, const FeatVector *hR_, int flag = 0x0):hLeft(hL_), hRight(hR_){
			this->nam = nam_;
			this->id = hData_->nFeat();
		}
		virtual ~FeatVec_EXP() {
		}
		bool isMerged()	const { return	true; }

		virtual size_t nSamp() const	{ return hLeft->nSamp(); }

		virtual void EDA(const LiteBOM_Config&config, bool genHisto, int flag);

		virtual void Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* hParent, HistoGRAM* histo, int nMostBin, int flag0 = 0x0) {
			throw "......";
		}


	};

}

