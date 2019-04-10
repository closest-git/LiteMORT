#pragma once

#include "./DataFold.hpp"

namespace Grusoft {
	class FeatVec_EXP : public FeatVec_T<tpQUANTI> {
	protected:
		FeatVec_T<tpQUANTI> *featX = nullptr, *featY = nullptr;
	public:
		FeatVec_EXP(FeatsOnFold *hData_, int id_, const FeatVec_T<tpQUANTI> *fX, const FeatVec_T<tpQUANTI> *fY, size_t nMostDup, int flag = 0x0);
		virtual ~FeatVec_EXP() {
		}

		virtual void UpdateFruit(MT_BiSplit *hBlit, int flag = 0x0);

		virtual void Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, int flag = 0x0) {
			;
		}
	};

}

