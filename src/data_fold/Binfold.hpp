#pragma once
#include <vector>
#include <map>  
#include <algorithm>
#include <cmath>
#include "../util/samp_set.hpp"

typedef double BINFOLD_FLOAT;

namespace Grusoft {
	class FeatsOnFold;
	class FRUIT;

	class BinFold {
	protected:
		FRUIT *fruit = nullptr;		//仅仅指向

		size_t nFeat=0,nQuanti=0, ldQ=0;
		BINFOLD_FLOAT *bGrad=nullptr,*bHess=nullptr;
		size_t *NZ = nullptr;
	public:
		BinFold(const FeatsOnFold *hData_, const vector<int> &pick_feats, const SAMP_SET&samp_set, int flag = 0x0);

		BinFold(size_t nF_,size_t nQ_, int flag = 0x0) : nFeat(nF_), nQuanti(nQ_){
			bGrad = new BINFOLD_FLOAT[nF_*nQ_]();
			bHess = new BINFOLD_FLOAT[nF_*nQ_]();
			NZ = new size_t[nF_*nQ_]();
		}
		virtual ~BinFold() {
			if (bGrad != nullptr)		delete[] bGrad;
			if (bHess != nullptr)		delete[] bHess;
			if (NZ != nullptr)			delete[] NZ;
		}

		virtual void GreedySplit(const FeatsOnFold *hData_, int flag = 0x0);
		
	};

	
}