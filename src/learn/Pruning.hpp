#pragma once

#include <string>
#include <vector>
#include <time.h>
#include <omp.h>
#include "../util/samp_set.hpp"


namespace Grusoft{
	typedef float tpMetricU;
	class FeatsOnFold;
	class EnsemblePruning{
	protected:
		size_t nSamp=0, nWeak=0,nMostWeak=0;
		FeatsOnFold *hFold = nullptr;
		int nSparsified() {
			int nPick, i;
			for (nPick = 0, i = 0; i < nWeak; i++) {
				if (w[i] > 0)
					nPick++;
			}
			return nPick;
		}
		void ToCSV(const string& sPath, int flag);
		void LoadCSV(const string& sPath, int flag);

	public:
		tpMetricU *U = nullptr, *w_0 = nullptr, *w = nullptr;
		short *plus_minus = nullptr;

		EnsemblePruning(FeatsOnFold *hFold, int nWeak_,int flag=0x0);
		virtual ~EnsemblePruning() {
			if (U != nullptr)		
				delete[] U;
			if (w_0 != nullptr)
				delete[] w_0;
			if (w != nullptr)
				delete[] w;
			if (plus_minus != nullptr)
				delete[] plus_minus;
		}
		virtual bool isValid() { return true; }

		virtual void Pick(int nWeak_,int T, int flag);

		virtual void OnStep(int noT, tpDOWN*down, int flag = 0x0);
	};

};


