#pragma once

#include "EDA.hpp"
#include "./DataFold.hpp"
#include "./FeatVec_2D.hpp"

using namespace Grusoft;


//参见bmpp_ii.code
struct IntegralTable {
	int *sumary, *sum2, M, N;
	IntegralTable(int m_,int n_, int flag = 0x0) : sumary(nullptr), sum2(nullptr) {
		M = m_, N = n_;
		Clear();
		int nz = M*N;
		sumary = new int[nz]();
		GST_VERIFY(sumary != nullptr, "IntegralImage::Init:sumary is 0");
	}
	IntegralTable() {
		//m=featX.bins.size		n=featY.bins.size
		//samp_float2d
		//Integral(float *Y_2d,int flag)
	}
	void Clear() {
		if (sumary != nullptr)		delete[] sumary;
		if (sum2 != nullptr)			delete[] sum2;
	}
	double Value(int c0, int r0, int n, int m, bool isMean) {
		assert(n>0 && m>0);
		int r1 = r0 + m, c1 = c0 + n, pos = G_RC2POS(r1 - 1, c1 - 1, N);
		assert(G_RC_VALID(r1 - 1, 0, M - 1) && G_RC_VALID(c1 - 1, 0, N - 1));
		double a = sumary[pos];
		if (r0>0 && c0>0)
			a += sumary[G_RC2POS(r0 - 1, c0 - 1, N)];
		if (r0>0 && c1>0)
			a -= sumary[G_RC2POS(r0 - 1, c1 - 1, N)];
		if (r1>0 && c0>0)
			a -= sumary[G_RC2POS(r1 - 1, c0 - 1, N)];
		if (isMean)
			a /= (m*n);
		return a;
	}
	~IntegralTable() { Clear(); }

	void Integral(float *Y_2d,int flag) {	
		int i, r, c, pos, isVeify = 0;
		int *s_i = new int[M*N], *X_i=nullptr;
		//BIT_8 *X_0 = this->Gray(), a;
		float a;
		double x_limit = 256.0*M*N;

		X_i = sumary;		//s_i2=D_buffer+M*N;		
		s_i[0] = 0;			//s_i2[0] = 0;
		pos = 0;
		for (c = 0; c < N; c++) {		//for r==0
			a = Y_2d[pos];
			//s(r,c)=s(r-1,c)+i(r,c)
			s_i[pos] = a;
			//ii(r,c)=ii(r,c-1)+s(r,c)			
			X_i[pos] = c == 0 ? s_i[pos] : X_i[(pos - 1)] + s_i[pos];
			assert(X_i[pos] >= -x_limit && X_i[pos] <= x_limit);
			pos++;
		}
		for (r = 1; r < M; r++) {
			a = Y_2d[pos];
			s_i[pos] = s_i[pos - N] + a;
			X_i[pos] = s_i[pos];
			assert(X_i[pos] >= -x_limit && X_i[pos] <= x_limit);
			pos++;
			for (c = 1; c < N; c++) {
				a = Y_2d[pos];
				//s(r,c)=s(r-1,c)+i(r,c)
				s_i[pos] = s_i[pos - N] + a;
				//ii(r,c)=ii(r,c-1)+s(r,c)			
				X_i[pos] = X_i[(pos - 1)] + s_i[pos];
				assert(X_i[pos] >= -x_limit && X_i[pos] <= x_limit);
				pos++;
			}
		}
		assert(pos == M*N);
		assert(X_i[M*N - 1]<INT_MAX);
		delete s_i;
	}
};
//IntegralTable IIT;


void HistoGRAM_2D::GreedySplit_X(const FeatsOnFold *hData_, const SAMP_SET& samp_set, int flag) {
	if (fruit == nullptr)
		throw "HistoGRAM_2D::GreedySplit_X fruit is 0!!!";

	size_t minSet = hData_->config.min_data_in_leaf;
	double sum = samp_set.Y_sum_1, a = a0, errL = 0, g, gL = 0, g1 = 0, lft, rgt;
	//size_t nLeft = 0, nRight = nSamp,minSet=0;
	nRight = nSamp;		assert(nRight >= 0);
	for (auto item : bins) {
	//for each(HISTO_BIN item in bins) {
		double errR = sum - errL;
		if (nLeft<minSet || nRight<minSet) {
			goto LOOP;
		}
		if (item.tic == a || item.nz == 0) {
			goto LOOP;
		}
		assert(item.tic>a);
		//Regression trees in CART have a constant numerical value in the leaves and use the variance as a measure of impurity
		g = errL*errL / nLeft + errR*errR / nRight;


		if (g>g1 || gL>g1) {
			g1 = MAX(g, gL);
			fruit->mxmxN = g1;			fruit->tic_left = item.tic;
			fruit->nLeft = nLeft;		fruit->nRight = nRight;
			//fruit->thrshold = item.tic;
			fruit->isNanaLeft = false;
			if (gL > g) {
				fruit->isNanaLeft = true;
				fruit->nLeft += binNA.nz;		fruit->nRight -= binNA.nz;
			}
			//fruit->lft_impuri = nLeft*lft*lft;		fruit->rgt_impuri = nRight*rgt*rgt;
		}
	LOOP:
		errL += item.G_sum;		a = item.tic;
		assert(nRight >= item.nz);
		nLeft += item.nz;			nRight -= item.nz;
	}
}

HistoGRAM::~HistoGRAM() {
	bins.clear();
}

void HistoGRAM::CompressBins(int flag) {
	vector<HISTO_BIN>::iterator iter = bins.begin();
	while (iter != bins.end()) {
		if (iter->nz == 0) {
			iter = bins.erase(iter);
		}
		else {
			++iter;
		}
	}
}

/*
	v0.2	cys
		1/28/2019
*/
void HistoGRAM::GreedySplit_X(const FeatsOnFold *hData_, const SAMP_SET& samp_set, int flag) {
	if (fruit == nullptr)
		throw "HistoGRAM::GreedySplit_X fruit is 0!!!";

	size_t minSet = hData_->config.min_data_in_leaf,nBin=bins.size(),i;
	string optimal = hData_->config.leaf_optimal, obj = hData_->config.objective;
	//double sum = samp_set.Y_sum_1, a = a0, errL = 0, g, gL = 0, g1 = 0, lft, rgt;
	double gL = 0, gR0, hL = 0, hR = 0, a = a0, g, g1 = 0;
	nLeft = 0;				nRight = nSamp;		assert(nRight >= 0);
	double gSum = binNA.G_sum, hSum = binNA.H_sum;
	double gainL = 0;		//对应于isNanaLeft
	for (auto item : bins) {
	//for each(HISTO_BIN item in bins) {
		gSum += item.G_sum, hSum += item.H_sum;
	}
	if (optimal == "lambda_0") {
		//assert(hSum == nSamp);
	}	else {
		assert(hSum == nSamp);
	}
	if(fabs(gSum+samp_set.Y_sum_1)>=1.e-7*fabs(gSum) && fabs(samp_set.Y_sum_1)>0.001)//不同的gSum计算确实有差异
		throw "HistoGRAM::GreedySplit_X gSum is different!!!";

	//for (auto item : bins) {
	for (i = 0; i < nBin;i++) {
		const HISTO_BIN &item = bins[i];
		//for each(HISTO_BIN item in bins) {
		double gR = gSum - gL, hR = hSum - hL;
		if (nLeft<minSet || nRight<minSet) {
			goto LOOP;
		}
		//4/9/2019	cys	注销item.nz == 0 这样可以使split thrshold更紧凑
		if (item.tic == a /*|| item.nz == 0*/) {
			goto LOOP;
		}
		assert(item.tic>a);
		//double errR = sum - errL;
		//Regression trees in CART have a constant numerical value in the leaves and use the variance as a measure of impurity
		//g = errL*errL / nLeft + errR*errR / nRight;
		g = gL*gL / hL + gR*gR / hR;
		if (false) {	//似乎并没有使结果更好 难以理解	 10/29/2018
			if (binNA.nz > 0 && nRight - binNA.nz >= minSet) {
				double eL = gL + binNA.G_sum, eR = gR - binNA.G_sum;
				gainL = eL*eL / (hL + binNA.H_sum) + eR*eR / (hR - binNA.H_sum);
			}
			else
				gainL = 0;
		}

		if (g>g1 || gainL>g1) {
			fruit->bin_S0 = bins[i-1];		fruit->bin_S1 = item;
			assert(fruit->bin_S0.nz>0);
			g1 = MAX(g, gainL);
			fruit->mxmxN = g1;			fruit->tic_left = item.tic;
			fruit->nLeft = nLeft;		fruit->nRight = nRight;
			//fruit->thrshold = item.tic;
			fruit->adaptive_thrsh = item.split_F;
			fruit->isNanaLeft = false;
			if (gainL > g) {
				fruit->isNanaLeft = true;
				fruit->nLeft += binNA.nz;		fruit->nRight -= binNA.nz;
				assert(bins[(int)(item.tic) - 1].nz>0);

			}
			//fruit->lft_impuri = nLeft*lft*lft;		fruit->rgt_impuri = nRight*rgt*rgt;
		}
	LOOP:
		//errL += item.Y_sum;		
		gL += item.G_sum;		hL += item.H_sum;
		a = item.tic;
		assert(nRight >= item.nz);
		nLeft += item.nz;			nRight -= item.nz;
	}
}

/*
	参见 "On Grouping for Maximum Homogeneity"
	"Fast Exact k-Means, k-Medians and Bregman Divergence Clustering in 1D"
	v0.1	cys
		10/10/2018
	v0.2	cys
		3/11/2019
*/
void HistoGRAM::GreedySplit_Y(const FeatsOnFold *hData_, const SAMP_SET& samp_set, bool tryX, int flag) {
	//FeatBlit flitX=flit;
	char temp[2000];
	FRUIT fruitX;
	if (tryX) {
		GreedySplit_X(hData_, samp_set, flag);
		fruitX = *fruit;
	}
	else {
		fruitX.mxmxN = -DBL_MAX;
	}
	//return;
	size_t minSet = hData_->config.min_data_in_leaf;
	//double sum = -samp_set.Y_sum_1, errL = 0, g, g1 = 0, a;
	double gL = 0, gR0, hL = 0, hR = 0, a = a0, g, g1 = 0;
	nLeft = 0;	nRight = nSamp;
	double gSum = binNA.G_sum, hSum = binNA.H_sum;
	double gainL = 0;		//对应于isNanaLeft
	for (auto item : bins) {
	//for each(HISTO_BIN item in bins) {
		gSum += item.G_sum, hSum += item.H_sum;
	}
	vector<tpSAMP_ID> idx;
	vector<double> Y_means;
	for (auto item : bins) {
	//for each(HISTO_BIN item in bins) {
		if (item.nz == 0) {
			Y_means.push_back(DBL_MAX);	continue;
		}
		//a = item.G_sum / item.nz;
		a = item.G_sum / item.H_sum;
		Y_means.push_back(a);
	}
	sort_indexes(Y_means, idx);
	size_t nBin = bins.size(), i, nY_l = 0, nY_r = 0;
	for (i = 0; i < nBin-1; i++) {		assert(Y_means[idx[i]]<= Y_means[idx[i+1]]);	}
	for (i = 0; i < nBin; i++) {
		const HISTO_BIN& item = bins[idx[i]];
		double gR = gSum - gL, hR = hSum - hL;
		if (nLeft<minSet || nRight<minSet) {
			goto LOOP;
		}
		if (item.nz == 0) {
			goto LOOP;
		}
		if(i>0 && Y_means[idx[i-1]]== Y_means[idx[i]]) {	//连续多个bin的Y_means可能一样
			goto LOOP;
		}
		//double errR = sum - errL;		
		//assert(!IS_NAN_INF(errR) && !IS_NAN_INF(errL));
		//Regression trees in CART have a constant numerical value in the leaves and use the variance as a measure of impurity
		//g = errL*errL / nLeft + errR*errR / nRight;
		g = gL*gL / hL + gR*gR / hR;
		if (g>g1) {
			g1 = g;
			fruit->mxmxN = g1;
			fruit->tic_left = Y_means[idx[i]];	// item.G_sum / item.nz;		assert(fruit->tic_left == Y_means[idx[i]]);
			fruit->nLeft = nLeft;		fruit->nRight = nRight;
			//fruit->thrshold = Y_means[idx[i]];
			fruit->adaptive_thrsh = Y_means[idx[i]];
			sprintf(temp, "L(%g/%g %d) R(%g/%g %d)", gL, hL, nLeft, gR, hR, nRight );
			this->sX = temp;
		}
	LOOP:
		//errL += item.G_sum;		a = item.tic;
		gL += item.G_sum;		hL += item.H_sum;
		assert(nRight >= item.nz);
		nLeft += item.nz;			nRight -= item.nz;
	}
	if (g1 == 0)	//Y的分布不一样，可能找不到split(例如minSet不符合)
	{		fruitX.histo = nullptr;	return;	}

	double aX = fruitX.mxmxN - fabs(fruitX.mxmxN)*1.0e-4;		//浮点误差真麻烦
	//assert(fruit->mxmxN>= aX);
	if (fruit->mxmxN>aX) {	//确实有可能
		fruit->isY = true;
		for (i = 0; i < nBin; i++) {
			HISTO_BIN& item = bins[idx[i]];
			if (Y_means[idx[i]] < fruit->adaptive_thrsh) {
			//if (item.G_sum / item.nz < fruit->thrshold) {
				item.fold = 0;
				nY_l += item.nz;
			}
			else {
				item.fold = 1;
				nY_r += item.nz;
			}
		}
		assert(nY_l== fruit->nLeft && nY_r== fruit->nRight);
	}
	else/**/ {
		if (fruit->mxmxN < fruitX.mxmxN)
			;// printf("!!!Split_X(%g)>Split_Y(%g)!!!", fruitX.mxmxN, fruit->mxmxN);
		*fruit = fruitX;
	}
	fruitX.histo = nullptr;
	return;
}

/*
	v0.1	cys
		10/10/2018
*/
void HistoGRAM::Regress(const FeatsOnFold *hData_, const SAMP_SET& samp_set, int flag) {
	if (fruit == nullptr)
		throw "HistoGRAM::GreedySplit_X fruit is 0!!!";

	size_t minSet = hData_->config.min_data_in_leaf;
	double sum = samp_set.Y_sum_1, g0 = sum*sum / nSamp, g1 = 0, mean_0 = sum / nSamp, a;
	//size_t nLeft = 0, nRight = nSamp,minSet=0;

	for (auto item : bins) {
	//for each(HISTO_BIN item in bins) {
		if (item.nz == 0) {
			continue;
		}
		a = item.G_sum*item.G_sum / item.nz;
		g1 += a;
	}
	assert(g1 >= g0);
	fruit->mxmxN = g1;
}

