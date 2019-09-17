#pragma once

#include <string>
#include <vector>
#include <time.h>
#include <omp.h>
#include "../util/samp_set.hpp"


namespace Grusoft{
	class EnsemblePruning{
	protected:
		size_t nSamp=0, nWeak=0;

		int nSparsified() {
			int nPick, i;
			for (nPick = 0, i = 0; i < nWeak; i++) {
				if (w[i] > 0)
					nPick++;
			}
			return nPick;
		}

	public:
		float *U = nullptr, *w_0 = nullptr, *w = nullptr;
		short *plus_minus = nullptr;

		EnsemblePruning(int n,int m,int flag=0x0);
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

		virtual void Pick(int T, int flag);
	};

};


