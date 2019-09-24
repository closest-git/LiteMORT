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

template <typename T>
void RAND_normal(size_t nX, T *x, int flag = 0x0) {
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0, 1.0);		//param_type(_Ty _Mean0 = 0.0, _Ty _Sigma0 = 1.0)

	for (size_t i = 0; i<nX; ++i) {
		double number = distribution(generator);
		x[i] = number;
	}
}

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
void scale_(size_t nX, T *x,int step, const T& s,int flag=0x0) {
	for (size_t i = 0; i < nX; i++) {
		x[i*step] *= s;
	}
}

/*Assume the first r rows of a forms an orthonormal basis	Returns the projection of x onto the orthogonal complement of the rows of a scaled to unit length
*/
template<typename Tx>
bool orthogonal_(double *orth, int ldO,int num_orth, Tx *x,size_t nX,int flag = 0x0) {
	double nrmX= norm_2(nX, x, flag);
	double *orth_row = orth + num_orth*ldO,*row_a;
	if (nrmX< 1e-8)
		return false;
	if (num_orth == 0 && flag==0) {
		for (size_t i = 0; i < nX; i++) {
			orth_row[i] = x[i]/ nrmX;
		}
		return true;
	}
	Tx *u=new Tx[nX];
	memcpy(u, x, sizeof(Tx)*nX);
	for(int j = 0; j < num_orth; j++){
		row_a = orth + j*ldO;
		double dot = dot_(nX, u, row_a);
		for (size_t i = 0; i < nX; i++) {
			u[i] -= dot*row_a[i];
		}
	}
	//u = u - np.inner(u,a[j]) * a[j]
	double nrmU = norm_2(nX, u, flag);
	if (nrmU< nrmX/100)
		return false;
	
	for (size_t i = 0; i < nX; i++) {
		if (flag == 1) {
			x[i] = u[i] / nrmU;
		}else
			orth_row[i] = u[i]/ nrmU;
	}
	delete[] u;
	//orth[num_orth] = p
	return true;
}

EnsemblePruning::EnsemblePruning(FeatsOnFold *hFold_, int mWeak_, int flag) : hFold(hFold_),nMostWeak(mWeak_) {
	nSamp = hFold->nSample();
	//ldA = nMostWeak;	//统一为row_major
	mA = new tpMetricU[nSamp*nMostWeak];
	mB = new tpMetricU[nSamp*nMostWeak];
	ax_ = new tpMetricU[nSamp];
	w_0 = new tpMetricU[nMostWeak];
	wx = new tpMetricU[nMostWeak];
	wy = new tpMetricU[nMostWeak];
	gamma = new double[nMostWeak]();
	wasSmall = new int[nSamp];
	y2x = new int[nSamp];
	for (size_t i = 0; i < nSamp; i++)
		wasSmall[i] = 1;

	isLive = new int[nSamp]();
	wasLive = new int[nSamp]();
	nWeak = 0;
	if (1) {
		LoadCSV("e:/EnsemblePruning_1250_18_.csv",0x0);
		Pick(0, 0, 1);
	}
}

EnsemblePruning::~EnsemblePruning() {
	FREE_a(mA);			FREE_a(mB);
	FREE_a(wx);			FREE_a(wy);
	FREE_a(ax_);
	FREE_a(isLive);		 FREE_a(wasLive);
	FREE_a(y2x);

	FREE_a(w_0);
	FREE_a(wasSmall);

	if (plus_minus != nullptr)
		delete[] plus_minus;
	if (gamma != nullptr)
		delete[] gamma;
}

/*
*/
void EnsemblePruning::OnStep(int noT_, tpDOWN*hWeak, int flag) {
	tpMetricU *U_t = mA + nWeak*nSamp;		//Construct margin matrix
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
		//tpMetricU *U_ = mA+samp;
		for (h = 0; h < nWeak; h++) {
			//fscanf(fp, "%f\t", U_+h*nSamp);
			fscanf(fp, "%f\t", &a);		mA[nz++]=a;
		}
		if (y != nullptr) {
			fscanf(fp, "%f", &a);
			assert(a == y[samp]);
		}
		fscanf(fp, "\n");
	}
	for (h = 0; h < nWeak; h++) {
		fscanf(fp, "%f\t", &a);			w_0[h] = a;
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
		tpMetricU *U_ = mA + samp;
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
			if (orthogonal_(orth, ldOrth, num_orth, B+sorted_indices[i]*ldB, ldB)) {
				num_orth = num_orth + 1;
				if (num_orth >= nLive_0)
					break;
			}
			nRun = nRun + 1;
			isSmall[sorted_indices[i]] = 0;
			if (nRun >= nMost)
				break;
		}
	}
}

void EnsemblePruning::sorted_ax(int flag) {
	int ldA = nWeak;
	double a;
	for (size_t i = 0; i < nSamp; i++) {
		a = dot_(ldA,mA+i*nWeak,wx);
		ax_[i] = -fabs(a);
	}
	sort_indexes(nSamp, ax_, sorted_indices);
}

double EnsemblePruning::UpateGamma(int *isLive,int nY,int flag) {
	double g_Max = 0;// , *gamma = new double[nY]();
					 //RAND_normal(nY, gamma);		// np.random.randn(nLive);
	if (true) {//仅用于调试
		double gamma_0[] = { -1.95880275, 0.03199851, 2.31866016, 1.13076601,-0.47713595, 1.10638398,-0.16743563, 0.016846, -0.63711211,-0.77417962,-0.46160502, 0.24628335,
		-0.00904039,-0.39995661, 0.69108282,-1.26731069,-2.25347049, 1.29984439 };	
		memcpy(gamma, gamma_0, sizeof(double)*nY);
	}
	if (!orthogonal_(orth, ldOrth, num_orth, gamma, nY, 1))
		return 0;
	int i;
	for (i = 0; i < nY; i++) {
		if (isLive[i] == 0)
			gamma[i] = 0;
		g_Max = max(g_Max, fabs(gamma[i]));
	}
	if (g_Max == 0)
		return g_Max;
	double bg,axbg=0;
	for (i = 0; i < nSamp; i++) {
		bg = dot_(nY, mB + i*nY, gamma);
		axbg += ax_[i] * bg;
	}
	if (axbg>0)	//if np.inner(ax,b @ gamma) > 0:
		scale_(nY, gamma,1, -1.0);

	return g_Max;
}

int EnsemblePruning::SubOnLive(int flag) {
	double delta = 1.0e-5;
	int nY = 0, i, nLive=0;
	for (i = 0; i < nWeak; i++) {
		if (fabs(wx[i]) < 1 - delta) {
			nLive++;
			isLive[nY] = 1;
			y2x[nY] = i;
			wy[nY++] = wx[i];
		}	else
			continue;
	}
	if (nY == nWeak) {
		memcpy(mB, mA, sizeof(tpMetricU)*nWeak*nSamp);
	}	else {
		for (i = 0; i < nY; i++) {
			tpMetricU *hA=mA+ y2x[i], *hB = mB+i;
			for (size_t j = 0; j < nSamp; j++, hA+= nWeak,hB+=nY) {
				*hB = *hA;
			}
		}
	}
	
	FREE_a(orth);
	ldOrth = nY;	num_orth = 0, orth = new double[ldOrth*ldOrth]();
	return nY;
}

/*
*/
bool EnsemblePruning::partial_infty_color(int nX,bool balance,int flag) {
	int nY = SubOnLive(flag), ldB = nY;		
	double delta = 1.0e-5,*e=new double[nY];
	int nLive = nY, nLive_0=nY, i, nIter = 0,  num_big=0, num_initial=0, num_diff=0, num_iters = 0, num_frozen=0;
//y = x[initial_is_live]	b = a[:, initial_is_live]
	if (nY < 8)
		return true;
	memcpy(wasLive, isLive, sizeof(int)*nY);
	for (i = 0; i < nSamp; i++) {
		wasSmall[i] = 1;
	}
	if (balance) {
		/*ones = np.ones(initial_live)
			ones = ones / np.sqrt(initial_live)
			orth[0] = ones
			num_orth = num_orth + 1*/
	}
	while ((int)(nLive*5/4)>nLive_0) {
		for (i = 0; i < nY; i++) {
			isLive[i] = fabs(wy[i]) < 1 - delta;
			if (isLive[i] != wasLive[i]) {
				num_diff++;				
				memset(e, 0x0, sizeof(double)*nY);				e[i] = 1;
				if (orthogonal_(orth, ldOrth, num_orth, e,nY)) {
					num_orth = num_orth + 1;
					if (num_orth >= nLive_0)
						break;
				}
				num_frozen = num_frozen + 1;			
			}
		}
		if (num_diff > 0) {			
			memcpy(wasLive, isLive, sizeof(int)*nY);
		}
		if (num_orth > nLive_0)
			break;
		num_iters = num_iters + 1;
		sorted_ax();
		//Ax_(A, x, ax, 0x0);		//ax = a @ x
		//	abs_ax = np.absolute(ax)
		if (num_initial < nLive_0 / 4) {
			make_orthogonal(mB, ldB, num_initial, (int)(ceil(nLive_0/4.0)), nLive_0, wasSmall, flag);
		}
		if (num_orth >= nLive_0)
			break;
		if (num_big < num_frozen) {
			make_orthogonal(mB, ldB, num_big, num_frozen, nLive_0, wasSmall, flag);
		}
		if (num_orth >= nLive_0)
			break;
		double gn=UpateGamma(isLive,nY,flag);
		if (gn == 0)
			break;
		double *val, coord_mult,a,z=DBL_MAX,s;
		for (i = 0; i < nY; i++) {
			coord_mult = gamma[i] * wy[i];
			s = (1e-27 + fabs(gamma[i]));
			a = coord_mult < 0 ? (1 + fabs(wy[i])) / s : (1 - fabs(wy[i])) / s;
			a = isLive[i] ? a : 1e27;
			z = min(z, a);
		}
		//memcpy(wasLive, isLive, sizeof(int)*nY);
		memset(isLive, 0x0, sizeof(int)*nY);
		for (nLive=0,i = 0; i < nY; i++) {
			wy[i] = wy[i] + z*gamma[i];
			wx[y2x[i]] = wy[i];
			if (fabs(wy[i]) < 1.0 - delta) {
				isLive[i] = 1;	nLive++;
			}
		}
		printf("");
	}

	return false;

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
		scale_(nSamp,mA+i,nWeak, w_0[i]);
	}
	if (flag == 0) {
		ToCSV("E:\\EnsemblePruning_"+std::to_string(nSamp) + "_"+std::to_string(nWeak) +"_.csv",0x0);
		return;
	}
	memset(wx, 0x0, sizeof(tpMetricU)*nWeak);
	while(!partial_infty_color(nWeak,false, 0x0))	;

	nPick = nSparsified();
	while (nPick > T) {
		vector<tpSAMP_ID> idx;
		sort_indexes(nWeak, wx, idx);
		nZero = 0;
		k = nWeak-nLarge-nZero;		//non-zero entries in w	that are not in R.
		float omiga = w_0[nLarge];
		//Aij
		//Spencer’s Theorem

		for (sum = 0, i = 0; i < nWeak; i++) {
			sum += plus_minus[i - nLarge];
		}
		sigma = sum>=0 ? -1 : 1;
		for (i = 0; i < nWeak; i++) {
			no = idx[i];
			if (no < nLarge) {

			}	else {
				if (plus_minus[i- nLarge]==sigma )
					wx[i] *= 2;
				else  {
					wx[i] = 0;
				}
			}
			
		}
		nPick = nSparsified();
		
	}
	delete[] plus_minus;
};