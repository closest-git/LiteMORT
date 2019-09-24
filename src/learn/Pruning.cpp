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

template<typename T>
double norm_2(size_t nX, T *x, int flag=0x0) {
	double sum_2 = 0;
	for (size_t i = 0; i < nX; i++) {
		sum_2 += x[i] * x[i];
	}
	return sqrt(sum_2);
}

template<typename T>
double dot_(size_t nX, T *x, T *y, int flag = 0x0) {
	double dot = 0;
	for (size_t i = 0; i < nX; i++) {
		dot += x[i] * y[i];
	}
	return dot;
}

template<typename T>
void scale_(size_t nX, T *x, const T& s,int flag=0x0) {
	for (size_t i = 0; i < nX; i++) {
		x[i] *= s;
	}
}


/*Assume the first r rows of a forms an orthonormal basis	Returns the projection of x onto the orthogonal complement of the rows of a scaled to unit length
*/
template<typename Tx>
bool orthogonal_(double *orth, int num_orth, Tx *x,size_t nX,int flag = 0x0) {
	double nrm2 = norm_2(nX, x, flag);
	tpMetricU *u=nullptr;
	if (nrm2< 1e-8)
		return false;
	if (num_orth == 0) {
		for (size_t i = 0; i < nX; i++) {
			x[i] /= nrm2;
		}
		return true;
	}
	//u = u - np.inner(u,a[j]) * a[j]
	nrm2 = norm_2(nX, u, flag);
	if (nrm2< 1e-8)
		return false;

	for (size_t i = 0; i < nX; i++) {
		x[i] = u[i]/nrm2;
	}
	//orth[num_orth] = p
	return true;
}

EnsemblePruning::EnsemblePruning(FeatsOnFold *hFold_, int mWeak_, int flag) : hFold(hFold_),nMostWeak(mWeak_) {
	nSamp = hFold->nSample();
	U = new tpMetricU[nSamp*nMostWeak];
	ax = new tpMetricU[nSamp];
	w_0 = new float[nMostWeak];
	w = new float[nMostWeak];
	nWeak = 0;
	if (1) {
		LoadCSV("e:/EnsemblePruning_1250_18_.csv",0x0);
		Pick(0, 0, 1);
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
	nWeak = 18;

	FILE *fp = fopen(sPath.c_str(), "rt");
	assert(fp != NULL);
	FeatVector *hY = hFold->GetY();
	FeatVec_T<double> *hYd = dynamic_cast<FeatVec_T<double>*>(hY);
	double *y = hYd == nullptr ? nullptr : hYd->arr();
	float a;
	size_t samp, h,nz=0;
	for (samp = 0; samp < nSamp; samp++) {
		tpMetricU *U_ = U+samp;
		for (h = 0; h < nWeak; h++) {
			//fscanf(fp, "%f\t", U_+h*nSamp);
			fscanf(fp, "%f\t", U + nz);		nz++;
		}
		if (y != nullptr) {
			fscanf(fp, "%f", &a);
			assert(a == y[samp]);
		}
		fscanf(fp, "\n");
	}
	for (h = 0; h < nWeak; h++) {
		fscanf(fp, "%f\t", w_0+h);
	}
	fclose(fp);
	printf("<<<<<< Load from %s ...  OK", sPath.c_str() );
}

void EnsemblePruning::Ax_(tpMetricU *A, tpMetricU*x, tpMetricU*ax, int flag) {

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
			fprintf(fp, "%lf\t", U_[h*nSamp]);
		}
		if (y != nullptr) {
			fprintf(fp, "%lf\t", y[samp]);
		}
		fprintf(fp, "\n");
	}
	for (h = 0; h < nWeak; h++) {
		fprintf(fp, "%lf\t", w_0[h]);
	}
	if (y != nullptr) { fprintf(fp, "%lf\t", -666666.0); }

	fprintf(fp, "\n");
	fclose(fp);
	printf(">>>>>> Dump to %s ...  OK", sPath.c_str());
}

void EnsemblePruning::make_orthogonal(tpMetricU *B,int ldB,int &nRun,int nMost,int nLive_0,int *isSmall, int flag ) {
	size_t i;
	for (i = 0; i < nSamp; i++) {
		if (isSmall[sorted_indices[i]]) {
			if (orthogonal_(orth, num_orth, B+sorted_indices[i]*ldB, ldB)) {
				num_orth = num_orth + 1;
				if (num_orth >= nLive_0)
					break;
			}
			nRun = nRun + 1;
			isSmall[sorted_indices[i]] = 0;
			if (nRun >= nMost / 4)
				break;
		}
	}
}

void EnsemblePruning::sorted_ax(int flag) {
	int ldU = nWeak;
	double a;
	for (size_t i = 0; i < nSamp; i++) {
		a = dot_(ldU,U+i*nWeak,x);
		ax[i] = -fabs(a);
	}
	sort_indexes(nSamp, ax, sorted_indices);
}
/*
*/
void EnsemblePruning::partial_infty_color(int nX,bool balance,int flag) {
	tpMetricU *A = U;
	int nY = nX, ldB = nY;		assert(ldB == nWeak);
	double delta = 1.0e-5,*e=new double[ldB];
	int nLive = 0, nLive_0, i, nIter = 0, num_orth=0, num_big=0, num_initial=0, num_diff=0, num_iters = 0, num_frozen=0;
	int *isLive = new int[nY], *wasLive = new int[nY], *wasSmall = new int[nSamp];
	double *orth = new double[nY*nY](),*y=new double[nY];
	float *ax;
	for (i = 0; i < nX; i++) {
		isLive[i] = 1;			wasSmall[i] = 1;
		if (fabs(x[i]) < 1 - delta) {
			nLive++;
		}
		y[i] = x[i];
	}
//y = x[initial_is_live]	b = a[:, initial_is_live]
	assert(nLive == nX);
	if (nLive < 8)
		return;
	memcpy(wasLive, isLive, sizeof(int)*nY);
	for (i = 0; i < nSamp; i++) {
		wasSmall[i] = 1;
	}
	nLive_0 = nLive;
	if (balance) {
		/*ones = np.ones(initial_live)
			ones = ones / np.sqrt(initial_live)
			orth[0] = ones
			num_orth = num_orth + 1*/
	}
	while ((int)(nLive*5/4)>nLive_0) {
		for (i = 0; i < nY; i++) {
			isLive[i] = fabs(y[i]) < 1 - delta;
			if (isLive[i] != wasLive[i]) {
				num_diff++;				
			}
		}
		if (num_diff > 0) {
			for (i = 0; i < nY; i++) {
				isLive[i] = fabs(y[i]) < 1 - delta;
				if (isLive[i] != wasLive[i]) {
					memset(e, 0x0, sizeof(double)*nY);
					e[i] = 1;
					if (orthogonal_(orth, num_orth, e,nY)) {
						num_orth = num_orth + 1;
						if (num_orth >= nLive_0)
							break;
					}
					num_frozen = num_frozen + 1;
				}
			}
			memcpy(wasLive, isLive, sizeof(int)*nY);
		}
		if (num_orth > nLive_0)
			break;
		num_iters = num_iters + 1;
		sorted_ax();
		//Ax_(A, x, ax, 0x0);		//ax = a @ x
		//	abs_ax = np.absolute(ax)
		if (num_initial < nLive_0 / 4) {
			make_orthogonal(A, ldB, num_initial, nLive_0 / 4, nLive_0, wasSmall, flag);
		}
		if (num_orth >= nLive_0)
			break;
		if (num_big < num_frozen) {
			make_orthogonal(A, ldB, num_big, num_frozen, nLive_0, wasSmall, flag);
		}
		if (num_orth >= nLive_0)
			break;
		/*double *g = np.random.randn(nLive);
		if (!orthogonal_(orth, num_orth, g,18))
			break;
		for (i = 0; i < nLive; i++) {
			if (isLive[i] == 0)
				gamma[i] = 0;
		}
				if not is_live[i] :
					gamma[i] = 0*/
		memcpy(wasLive, isLive, sizeof(int)*nY);
	}

	delete[] isLive;		delete[] wasLive;
	delete[] wasSmall;		delete[] orth;
	delete[] y;

}

void EnsemblePruning::Pick(int tt, int T,int flag){
	//nWeak = nWeak_;
	int nPick = nWeak,nLarge=nSamp/3,i,no,k, nZero;
	double sum = 0;
	short sigma = 0;
	plus_minus = new short[nWeak];
	for (sum = 0, i = 0; i < nWeak; i++) {	sum += fabs(w_0[i]);	}
	for (i = 0; i < nWeak; i++) { 
		w_0[i] /= sum; 
		scale_(nWeak,U+i*nWeak, w_0[i]);
	}
	if (flag == 0) {
		ToCSV("E:\\EnsemblePruning_"+std::to_string(nSamp) + "_"+std::to_string(nWeak) +"_.csv",0x0);
		return;
	}
	x=new tpMetricU[nWeak]();
	partial_infty_color(nWeak,false, 0x0);
	delete[] x;

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