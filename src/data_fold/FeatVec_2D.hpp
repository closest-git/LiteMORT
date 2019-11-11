#pragma once

#include "./DataFold.hpp"

namespace Grusoft {
	class HistoGRAM_2D : public HistoGRAM {
	protected:
		HistoGRAM *histoX = nullptr, *histoY = nullptr;
	public:
		HistoGRAM_2D(FeatVector*hFeat_, size_t nMost, int flag = 0x0) : HistoGRAM(hFeat_,nMost, flag) {
		}
		virtual ~HistoGRAM_2D() {
			;
		}

		virtual void GreedySplit_X(const FeatsOnFold *hData_, const SAMP_SET& samp_set, int flag = 0x0);
	};

	class FeatVec_2D : public FeatVec_T<short> {
	protected:
		FeatVec_T<short> *featX = nullptr, *featY = nullptr;
	public:
		FeatVec_2D(FeatsOnFold *hData_, int id_, const FeatVec_T<short> *fX, const FeatVec_T<short> *fY, size_t nMostDup, int flag = 0x0);
		virtual ~FeatVec_2D() {
		}


		
	};

}

