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
#include "../tree/BoostingForest.hpp"

using namespace Grusoft;
using namespace std;

bool EnsemblePruning::isDebug = false;
bool EnsemblePruning::isRand = false;

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
double norm_2(size_t nX, const T *x, int flag=0x0) {
	double sum_2 = 0;
	for (size_t i = 0; i < nX; i++) {
		sum_2 += x[i] * x[i];
	}
	return sqrt(sum_2);
}

template<typename Tx>
double norm_(size_t nX, const Tx* x, int flag = 0x0) {
	double nrm = 0;
	for (size_t i = 0; i < nX; i++) {
		nrm = max(nrm, fabs(x[i]));
	}
	return nrm;
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

EnsemblePruning::EnsemblePruning(BoostingForest *hBoost_, FeatsOnFold *hFold_, int mWeak_, int flag) : hBoost(hBoost_),hFold(hFold_),nMostWeak(mWeak_) {
	nSamp = hFold->nSample();
	if (isDebug) {
		nMostWeak = 98;
	}
	//ldA = nMostWeak;	//统一为row_major
	mA = new tpMetricU[nSamp*nMostWeak];
	mB = new tpMetricU[nSamp*nMostWeak];
	ax_ = new tpMetricU[nSamp];
	cc_0 = new tpMetricU[nMostWeak];
	cc_1 = new tpMetricU[nMostWeak];
	plus_minus = new double[nMostWeak];
	wx = new tpMetricU[nMostWeak];
	wy = new tpMetricU[nMostWeak];
	gamma = new double[nMostWeak]();
	wasSmall = new int[nSamp];
	y2x = new int[nSamp];
	//init_score = new double[nSamp]();
	for (size_t i = 0; i < nSamp; i++)
		wasSmall[i] = 1;

	isLive = new int[nSamp]();
	wasLive = new int[nSamp]();
	nWeak = 0;
	if (isDebug) {
		nWeak = 98;		 LoadCSV("e:/EnsemblePruning_625_98_.csv", 0x0);
		Pick(0, 0, 1);
	}
}

/*
	测试集（"e:/EnsemblePruning_1250_18_.csv"）	score_1=0.8564785350613284，score_2=0.8487257000694284
*/
bool EnsemblePruning::Compare(int flag) {
	//printf("\n======EnsemblePruning::nWeak=%d=>%d err_0=%.5g score=%.4g=>=%.4g", 0, 0, 1., 1., 1.);
	EARLY_STOPPING& stop = hBoost->stopping;
	double err_0 = 0;
	int nz_0 = nWeak, nz_1 = nWeak;
	if (isDebug) {
	}	else {
		assert(nWeak + 1 < stop.errors.size());
		err_0 = stop.errors[nWeak + 1];
	}

	double err_1 = 0, err_2 = 0, *pred_1 = new double[nSamp], *pred_2 = new double[nSamp],s;
	//memcpy()
	double *y = hFold->GetY_<double>();
	DCRIMI_2 decrimi_2;
	size_t i;
	for (i = 0; i < nWeak; i++) {
		if (fabs(cc_0[i]) == 0)	nz_0--;
		if (fabs(cc_1[i]) == 0)	nz_1--;
	}
	for (i = 0; i < nSamp; i++) {
		s = init_score[i];
		pred_1[i] = dot_(nWeak, mA + i*nWeak, cc_0) + s;
		pred_2[i] = dot_(nWeak, mA + i*nWeak, cc_1) + s;
	}
	double score_1 = decrimi_2.AUC_Jonson(nSamp, y, pred_1);
	double score_2 = decrimi_2.AUC_Jonson(nSamp, y, pred_2);
	err_1 = 1 - score_1;
	err_2 = 1 - score_2;
	printf("\n======EnsemblePruning::nWeak=%d=>%d err_0=%.5g score=%.4g=>=%.4g\n", nz_0, nz_1, err_0, err_1, err_2);
	//assert(err_0== err_1);
	delete[] pred_1;		delete[] pred_2;
	return err_1 < err_2;
}

EnsemblePruning::~EnsemblePruning() {
	FREE_a(mA);			FREE_a(mB);
	FREE_a(wx);			FREE_a(wy);
	FREE_a(ax_);
	FREE_a(isLive);		 FREE_a(wasLive);
	FREE_a(y2x);

	FREE_a(init_score);

	FREE_a(cc_0);		FREE_a(cc_1);
	FREE_a(wasSmall);

	if (plus_minus != nullptr)
		delete[] plus_minus;
	if (gamma != nullptr)
		delete[] gamma;/**/
}

/*
*/
void EnsemblePruning::OnStep(ManifoldTree *hTree, tpDOWN*hWeak, int flag) {
	if (init_score == nullptr) {	//很妙的解释	https://towardsdatascience.com/demystifying-maths-of-gradient-boosting-bd5715e82b7c
		assert(hWeak!=nullptr);
		init_score = new double[nSamp];
		for (size_t i = 0; i < nSamp; i++) {
			init_score[i] = hWeak[i];
		}
	}	else {
		forest.push_back(hTree);
		tpMetricU *U_t = mA + nWeak*nSamp;		//Construct margin matrix
		assert(nWeak >= 0 && nWeak < nMostWeak);
		for (size_t i = 0; i < nSamp; i++) {
			U_t[i] = hWeak[i];
		}
		nWeak = nWeak + 1;
	}
}

void EnsemblePruning::LoadCSV(const string& sPath, int flag) {
	FILE *fp = fopen(sPath.c_str(), "rt");
	assert(fp != NULL);
	FeatVector *hY = hFold->GetY();
	FeatVec_T<double> *hYd = dynamic_cast<FeatVec_T<double>*>(hY);
	double *y = hYd == nullptr ? nullptr : hYd->arr();
	float a;
	size_t samp, h,nz=0;
	for (samp = 0; samp < nSamp; samp++) {
		for (h = 0; h < nWeak; h++) {
			//fscanf(fp, "%f\t", U_+h*nSamp);
			fscanf(fp, "%f\t", &a);		
			mA[h*nSamp+samp] = a;//mA[nz++]=a;
		}
		if (y != nullptr) {
			fscanf(fp, "%f", &a);
			assert(a == y[samp]);
		}
		fscanf(fp, "\n");
	}
	for (h = 0; h < nWeak; h++) {
		fscanf(fp, "%f\t", &a);			cc_0[h] = a;
	}
	fscanf(fp, "%f\t", &a);			
	assert( nWeak == (int)a);

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
		fprintf(fp, "%lf\t", init_score[samp]);
		fprintf(fp, "\n");
	}
	for (h = 0; h < nWeak; h++) {
		assert(!IS_NAN_INF(cc_0[h]));
		fprintf(fp, "%lf\t", cc_0[h]);
	}
	if (y != nullptr) 	{ 
		fprintf(fp, "%lf\t", nWeak*1.0);	fprintf(fp, "%lf\t",-6666.0);
	}

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
		ax_[i] = a;		// -fabs(a);
	}
	//sort_indexes(nSamp, ax_, sorted_indices);
	size_t i;
	sorted_indices.resize(nSamp);// initialize original index locations
	iota(sorted_indices.begin(), sorted_indices.end(), 0);
	// sort indexes based on comparing values in v
	const tpMetricU*v = ax_;
	std::sort(sorted_indices.begin(), sorted_indices.end(), [&v](size_t i1, size_t i2) {return -fabs(v[i1]) < -fabs(v[i2]); });
	
}

double EnsemblePruning::UpateGamma(int *isLive,int nY,int flag) {
	double g_Max = 0;// , *gamma = new double[nY]();
	RAND_normal(nY, gamma);		// np.random.randn(nLive);
	if (!isRand) {//仅用于调试
		double gamma_0[] = { -1.95880275, 0.03199851, 2.31866016, 1.13076601,-0.47713595, 1.10638398,-0.16743563, 0.016846, -0.63711211,-0.77417962,-0.46160502, 0.24628335,
		-0.00904039,-0.39995661, 0.69108282,-1.26731069,-2.25347049, 1.29984439 };	
		for (int i = 0; i < nY; i++)		gamma[i] = 1;	//memcpy(gamma, gamma_0, sizeof(double)*nY);
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

//is_live=abs_x < 1.0-delta;	 mB = mA[:,is_live];		wy = wx[is_live]
int EnsemblePruning::SubOnLive(double delta, bool update_orth, double *v_0, double *v_sub, int flag) {
	//double delta = 1.0e-5;
	int nY = 0, i, nLive=0;
	for (i = 0; i < nWeak; i++) {
		if (fabs(wx[i]) < 1 - delta) {
			nLive++;
			isLive[nY] = 1;
			y2x[nY] = i;
			v_sub[nY++] = v_0[i];
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
	
	if (update_orth) {
		FREE_a(orth);
		ldOrth = nY;	num_orth = 0, orth = new double[ldOrth*ldOrth]();

	}
	return nY;
}

/*
*/
bool EnsemblePruning::partial_infty_color(int nX,bool balanced,int flag) {
	double delta = 1.0e-5;
	int nY = SubOnLive(delta,true,wx,wy,flag), ldB = nY, i, nIter = 0,  num_big=0, num_initial=0, num_diff=0, num_iters = 0, num_frozen=0;
	double  *e = new double[nY];
	nLive = nY, nLive_0=nY;
//y = x[initial_is_live]	b = a[:, initial_is_live]
	if (nY < 8)
		return true;
	memcpy(wasLive, isLive, sizeof(int)*nY);
	for (i = 0; i < nSamp; i++) {
		wasSmall[i] = 1;
	}
	if (balanced) {
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

void EnsemblePruning::basic_local_search(double *x_,  bool balanced, int flag){
	size_t i,j;
	for (i = 0; i < nSamp; i++) {
		ax_[i] = dot_(nWeak,mA+i*nWeak, x_);
	}
	double best_norm = norm_(nSamp,ax_),nrm_,*flipped=new double[nSamp];
	bool improved = true;
	while (improved) {
		improved = false;
		for(i = 0; i < nWeak; i++ ){
			double s = x_[i];
			tpMetricU*col = mA + i;
			for (j = 0; j < nSamp; j++, col+=nWeak) {
				flipped[j] = ax_[j]-2*s*(*col);
			}
			nrm_ = norm_(nSamp, flipped);
			if (nrm_ < best_norm) {
				if (balanced) {
					/*
					for j in range(0,a.shape[1]):
					if x[i]==x[j]:
					continue
					final_flipped = flipped - 2*x[j]*a[:,j]
					if np.linalg.norm(final_flipped,ord=norm) < best_norm:
					ax = final_flipped
					x[i] = -x[i]
					x[j] = -x[j]
					best_norm = np.linalg.norm(final_flipped,ord=norm)
					improved=True
					break
					*/
				}
				else {
					memcpy(ax_, flipped, sizeof(tpMetricU)*nSamp);
					x_[i] = -x_[i];
					best_norm = norm_(nSamp,flipped);
					improved = true;
				}
			}
			
		}
	}
	delete[] flipped;
}

/*
*/
void EnsemblePruning::local_improvements(double *x_, bool balanced, int flag) {
	assert(0);
	size_t i, j;
	for (i = 0; i < nSamp; i++) {
		ax_[i] = dot_(nWeak, mA + i*nWeak, x_);
	}
	double best_norm = norm_(nSamp, ax_), nrm_, *flipped = new double[nSamp];
	int num_flip = min(7, nWeak), iters = 0,iter=0;
	if (balanced && num_flip % 2 == 1)
		num_flip = num_flip + 1;

	while (true) {
		iters = iters + 1;
		for (iter = 0; iter < nWeak; iter++) {
			int sampled_coords[] = { 0, 1, 5, 6, 9, 15, 16 };
			bool allDistinct = true;
			for (i = 0; i < num_flip - 1; i++) {
				if (sampled_coords[i] == sampled_coords[i + 1])
					allDistinct = false;
			}
			if (!allDistinct)		continue;
			if (balanced) {
				/*sum_is=0
                for i in range(0,num_flip):
                    sum_is = sum_is + x[sampled_coords[i]]
                if sum_is!=0:
                    continue*/
			}


		}
	}
	delete[] flipped;
}

void EnsemblePruning::greedy(double *grad,bool balanced, int flag) {
	double *x= grad,*so_far=new double[nSamp],*col;
	size_t i,j;
	if (balanced) {
		/*
		so_far = np.zeros(a.shape[0])
		for i in range(0,a.shape[1]):
		if i%2==1:
		continue
		test = so_far + a[:,i]
		test_minus = so_far - a[:,i]
		if i<a.shape[1]-1:
		test = test - a[:,i+1]
		test_minus = test_minus + a[:,i+1]
		if np.linalg.norm(test,ord=norm) < np.linalg.norm(test_minus,ord=norm):
		x[i] = 1
		if i<a.shape[1]-1:
		x[i+1]=-1
		else:
		x[i] = -1
		if i<a.shape[1]-1:
		x[i+1] = 1
		*/
	}
	else {
		x[0] = 1;
		double norm_1=0,norm_2=0,a1,a2,a;
		for (col=mA,j = 0; j < nSamp; j++,col+=nWeak) {
			so_far[j] = *col;
		}
		for (i = 1; i < nWeak; i++) {
			norm_1 = 0, norm_2 = 0;
			for (col = mA+i,j = 0; j < nSamp; j++, col += nWeak) {
				a = *col;
				a1 = so_far[j] + a;		a2 = so_far[j] - a;
				norm_1 = max(norm_1, fabs(a1));
				norm_2 = max(norm_2, fabs(a2));
			}
			x[i] = (norm_1 < norm_2) ? 1 : -1;
			for (col = mA + i, j = 0; j < nSamp; j++, col += nWeak) {
				so_far[j] += x[i] * (*col);
			}
		}
	}
	delete[] so_far;
}

void EnsemblePruning::round_coloring(bool balanced,int flag) {
	double *samps = new double[nWeak*6],*flips=samps+nWeak,*init_y=flips+nWeak,*sub_y= init_y+nWeak, *best_sub_y= sub_y+nWeak,*new_sub_y= best_sub_y +nWeak;
	RAND_normal(nWeak, samps);
	if (!isRand) {		//仅用于调试
		//double samples_0[] = { 0.04892675, 0.85041767, 0.0882261, 0.00201122, 0.64102732, 0.26890527, 0.1720314, 0.76263232, 0.54824072, 0.14100026, 0.17752911, 0.09100698
		//	,0.16327615, 0.34498547, 0.49066404, 0.44659881, 0.04286212, 0.94289195 };
		for (int i = 0; i < nWeak; i++)		samps[i] = 1;	//	memcpy(samps, samples_0, sizeof(double)*nWeak);
	}
	int *sign_flips = new int[nWeak],i,at=0,sign;
	double a, new_norm, best_norm=0,*a_outside=new double[nSamp], ay, sub_ay, b_y,new_ay;
	nLive = 0;
	for (i = 0; i < nWeak; i++) {
		sign_flips[i] = 1;
		a = fabs(wx[i]);
		if (a < 1.0 - 1e-4)	//live = (abs < 1.0-1e-4)
			nLive++;
		sign = samps[i] < (1 - a) / 2 ? -1 : 1;
		flips[i] = wx[i] * sign;
		init_y[i] = flips[i] < 0 ? -1 : 1;
	}
	if (nLive <= 10) {
		int nY = SubOnLive(1e-4, false,init_y, sub_y,flag);
		assert(nY == nLive);
		for (best_norm=0,i = 0; i < nSamp; i++){
			ay = dot_(nWeak,mA+i*nWeak, init_y);
			best_norm = max(best_norm, fabs(ay));
			sub_ay = dot_(nY,mB + i*nY, sub_y);
			a_outside[i] = ay - sub_ay;
		}
		memcpy(best_sub_y, sub_y, sizeof(double)*nY);

		while (true) {
			at = 0;
			while (at < nLive && sign_flips[at] == -1) {
				sign_flips[at] = 1;		at++;
			}
			if (at == nLive)
				break;
			sign_flips[at] = -1;
			for (i = 0; i < nY; i++) {
				new_sub_y[i] = sub_y[i]*sign_flips[i];		//new_sub_y = np.multiply(sub_y, sign_flips)
			}		
			for (new_norm=0,i = 0; i < nSamp; i++) {
				b_y = dot_(nY, mB + i*nY, new_sub_y);//new_sub_ay = sub_a @ new_sub_y
				new_ay = a_outside[i] + b_y;//new_ay = a_outside + new_sub_ay
				new_norm = max(new_norm, fabs(new_ay));
			}	
			if (new_norm < best_norm) {
				best_norm = new_norm;
				memcpy(best_sub_y, new_sub_y, sizeof(double)*nY);
			}
		}
		for (i = 0; i < nY; i++) {		//y[live_indices] = best_sub_y
			init_y[y2x[i]] = best_sub_y[i];
		}
	}
	if (balanced) {
/*
		while True:
		toFlip = 1
		if sum(y==1) < n/2:
		toFlip = -1
		ofToFlip = sum(y==toFlip)
		needToFlip = int(ofToFlip-n/2)
		if needToFlip==0:
		break
		listOfToFlip = [i for i, x in enumerate(y) if x==toFlip]
		ay = a @ y
		best_norm = 1e27
		#try all single flips
		best_to_flip = 0
		for i in listOfToFlip:
		col = a[:,i]
		new_ay = ay - (2*col*y[i])
		new_norm = np.linalg.norm(new_ay,ord=norm)
		if new_norm < best_norm:
		best_norm = new_norm
		best_to_flip = i
		y[best_to_flip] = -y[best_to_flip]
*/
	}
	for (i = 0; i < nWeak; i++) {
		assert(init_y[i]==1 || init_y[i]==-1);
		plus_minus[i] = init_y[i];
	}
	delete[] sign_flips;		delete[] samps;
	delete[] a_outside;
}

void EnsemblePruning::Prepare(int flag) {
	size_t i, j;
	tpMetricU *src=nullptr, *target = nullptr;
	for (i = 0; i < nWeak; i++) {	//transpose
		src = mA + i*nSamp;	target = mB + i;
		for (j = 0; j < nSamp; j++, target+=nWeak, src++) {
			*target = *src;
		}
	}
	memcpy(mA, mB, sizeof(tpMetricU)*nWeak*nSamp);
}

void EnsemblePruning::Reset4Pick(int flag) {
	FREE_a(init_score);
	nWeak = 0;
	forest.clear();
}

bool EnsemblePruning::Pick(int nTree, int isToCSV,int flag){
	//assert(nWeak<= nTree);
	int nPick = nWeak,nLarge=nSamp/3,i,no,k, nZero, num_ones=0;	
	for (cc_0_sum = 0, i = 0; i < nWeak; i++) { cc_0_sum += fabs(cc_0[i]);	}
	assert(cc_0_sum>0 && cc_0_sum<nWeak*10);
	/*if (isToCSV) {
		ToCSV("E:\\EnsemblePruning_"+std::to_string(nSamp) + "__.csv",0x0);
	}*/	
	GST_TIC(tic);	
	Prepare();
	//nWeak = nWeak_;
	bool balanced = false;
	double *grad=new double[nWeak];
	for (i = 0; i < nWeak; i++) { 
		cc_0[i] /= cc_0_sum;
		scale_(nSamp,mA+i,nWeak, cc_0[i]);	//scaled_a = np.multiply(sub_a,sub_x)	discrepancy_minimize的输入
	}
	//Compare(0x0);

	memcpy(cc_1, cc_0, sizeof(tpMetricU)*nWeak);
	memset(wx, 0x0, sizeof(tpMetricU)*nWeak);
	while(!partial_infty_color(nWeak,false, 0x0))	;
	round_coloring(balanced);
	basic_local_search(plus_minus,balanced);
	greedy(grad,balanced);
	basic_local_search(grad, balanced);
	double nrm_g = 0, nrm_y = 0,a;
	for (i = 0; i < nSamp; i++) {
		a = dot_(nWeak, mA + i*nWeak, grad);
		nrm_g = max(nrm_g, fabs(a));
		a = dot_(nWeak, mA + i*nWeak, plus_minus);
		nrm_y = max(nrm_y, fabs(a));
	}
	if (nrm_g < nrm_y) {
		memcpy(plus_minus, grad, sizeof(double)*nWeak);
	}
	//local_improvements(balanced);
	basic_local_search(plus_minus, balanced);
	for (num_ones=0,i = 0; i < nWeak; i++) {
		if (plus_minus[i] != -1) {
			plus_minus[i] =  1;
			num_ones++;
		}
	}
	int t = num_ones <= nWeak / 2 ? 1 : -1;
	for ( i = 0; i < nWeak; i++) {
		if (plus_minus[i] == t)
			cc_1[i] *= 2;
		else
			cc_1[i] = 0;
	}	
	printf("");
	delete[] grad;
	for (i = 0; i < nWeak; i++) {
		scale_(nSamp, mA + i, nWeak, 1.0/cc_0[i]);	//Compare的输入
		cc_0[i] *= cc_0_sum;		cc_1[i] *= cc_0_sum;
	}
	printf("\n====== EnsemblePruning::Pick time=%.4g", GST_TOC(tic));
	return Compare(flag);
}

/*
nPick = nSparsified();
while (nPick > T) {
vector<tpSAMP_ID> idx;
sort_indexes(nWeak, wx, idx);
nZero = 0;
k = nWeak-nLarge-nZero;		//non-zero entries in w	that are not in R.
float omiga = cc_0[nLarge];
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
*/