#include <memory>
#include <iostream>
#include <algorithm>
#include <tchar.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <fstream>
#include "../util/GST_def.h"
#include "Pruning.hpp"
#include "../data_fold/DataFold.hpp"

using namespace Grusoft;
using namespace std;

EnsemblePruning::EnsemblePruning(FeatsOnFold *hFold_, int mWeak_, int flag) : hFold(hFold_),nMostWeak(mWeak_) {
	nSamp = hFold->nSample();
	U = new float[nSamp*nMostWeak];
	w_0 = new float[nMostWeak];
	w = new float[nMostWeak];
	nWeak = 0;
	if (1) {
		LoadCSV("e:/EnsemblePruning_1250_18_.csv",0x0);
		Pick(0, 0, 0);
	}
}

/*
*/
void EnsemblePruning::OnStep(int noT_, tpDOWN*hWeak, int flag) {
	tpMetricU *U_t = U + nWeak*nSamp;		//Construct margin matrix
	assert(nWeak >= 0 && nWeak < nMostWeak);
	for (size_t i = 0; i < nSamp; i++) {
		U_t[i] = hWeak[i];
	}
	nWeak = nWeak + 1;
}

void EnsemblePruning::LoadCSV(const string& sPath, int flag) {
	FILE *fp = fopen(sPath.c_str(), "rt");
	assert(fp != NULL);
	FeatVector *hY = hFold->GetY();
	FeatVec_T<double> *hYd = dynamic_cast<FeatVec_T<double>*>(hY);
	double *y = hYd == nullptr ? nullptr : hYd->arr(),a;
	size_t samp, h;
	for (samp = 0; samp < nSamp; samp++) {
		tpMetricU *U_ = U+samp;
		for (h = 0; h < nWeak; h++) {
			fscanf(fp, "%g\t", U_[h*nSamp]);
		}
		if (y != nullptr) {
			fscanf(fp, "%g\t", &a);
			assert(a == y[samp]);
		}
		fscanf(fp, "\n");
	}
	for (h = 0; h < nWeak; h++) {
		fscanf(fp, "%g\t", w_0[h]);
	}
	fclose(fp);
	printf("<<<<<< Load from %s ...  OK", sPath.c_str() );
}

void EnsemblePruning::ToCSV(const string& sPath, int flag) {
	FILE *fp = fopen(sPath.c_str(), "wt");
	assert(fp != NULL);
	size_t samp, h;
	FeatVector *hY = hFold->GetY();
	FeatVec_T<double> *hYd = dynamic_cast<FeatVec_T<double>*>(hY);
	double *y = hYd == nullptr ? nullptr : hYd->arr();
	for (samp = 0; samp < nSamp; samp++) {
		tpMetricU *U_ = U + samp;
		for (h = 0; h < nWeak; h++) {
			fprintf(fp, "%g\t", U_[h*nSamp]);
		}
		if (y != nullptr) {
			fprintf(fp, "%g\t", y[samp]);
		}
		fprintf(fp, "\n");
	}
	for (h = 0; h < nWeak; h++) {
		fprintf(fp, "%g\t", w_0[h]);
	}
	if (y != nullptr) { fprintf(fp, "%g\t", -666666.0); }

	fprintf(fp, "\n");
	fclose(fp);
	printf(">>>>>> Dump to %s ...  OK", sPath.c_str());
}

void EnsemblePruning::Pick(int tt, int T,int flag){
	//nWeak = nWeak_;
	int nPick = nWeak,nLarge=nSamp/3,i,no,k, nZero;
	double sum = 0;
	short sigma = 0;
	plus_minus = new short[nWeak];
	for (sum = 0, i = 0; i < nWeak; i++) {	sum += fabs(w_0[i]);	}
	for (i = 0; i < nWeak; i++) { w_0[i] /= sum; }
	ToCSV("E:\\EnsemblePruning_"+std::to_string(nSamp) + "_"+std::to_string(nWeak) +"_.csv",0x0);
	return;

	memcpy(w, w_0, sizeof(tpMetricU)*nWeak);
	nPick = nSparsified();
	while (nPick > T) {
		vector<tpSAMP_ID> idx;
		sort_indexes(nWeak, w, idx);
		nZero = 0;
		k = nWeak-nLarge-nZero;		//non-zero entries in w	that are not in R.
		float omiga = w_0[nLarge];
		//Aij
		//Spencer¡¯s Theorem

		for (sum = 0, i = 0; i < nWeak; i++) {
			sum += plus_minus[i - nLarge];
		}
		sigma = sum>=0 ? -1 : 1;
		for (i = 0; i < nWeak; i++) {
			no = idx[i];
			if (no < nLarge) {

			}	else {
				if (plus_minus[i- nLarge]==sigma )
					w[i] *= 2;
				else  {
					w[i] = 0;
				}
			}
			
		}
		nPick = nSparsified();
		
	}
	delete[] plus_minus;
};