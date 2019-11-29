#pragma once

#include "EDA.hpp"
#include "./DataFold.hpp"
#include "./FeatVec_2D.hpp"

using namespace Grusoft;

size_t HistoGRAM::nAlloc = 0;
double HistoGRAM::memAlloc = 0;
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
	//if (fruit == nullptr)
	//	throw "HistoGRAM_2D::GreedySplit_X fruit is 0!!!";

	size_t minSet = hData_->config.min_data_in_leaf,i;
	double sum = samp_set.Y_sum_1, a = -DBL_MAX, errL = 0, g, gL = 0, g1 = 0, lft, rgt;
	//size_t nLeft = 0, nRight = nSamp,minSet=0;
	nRight = nSamp;		assert(nRight >= 0);
	HISTO_BIN*binNA = this->hBinNA();
//	for (auto item : bins) {
	for (i = 0; i < nBins; i++) {
		const HISTO_BIN &item = bins[i];

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
			g1 = MAX2(g, gL);
			UpdateBestGain(item.tic,g1,nLeft,nRight);
			/*fruit->mxmxN = g1;			fruit->tic_left = item.tic;
			fruit->nLeft = nLeft;		fruit->nRight = nRight;
			//fruit->thrshold = item.tic;
			fruit->isNanaLeft = false;
			if (gL > g) {
				fruit->isNanaLeft = true;
				fruit->nLeft += binNA->nz;		fruit->nRight -= binNA->nz;
			}
			//fruit->lft_impuri = nLeft*lft*lft;		fruit->rgt_impuri = nRight*rgt*rgt;*/
		}
	LOOP:
		errL += item.G_sum;		a = item.tic;
		assert(nRight >= item.nz);
		nLeft += item.nz;			nRight -= item.nz;
	}
}

HistoGRAM::~HistoGRAM() {
	//bins.clear();
	if (buffer == nullptr) {
		if (bins != nullptr)		
			delete[] bins;
	}
}

void HistoGRAM::Dump(const vector<BIN_FEATA>&binFeatas, MAP_CATEGORY&mapCategory, int flag) {
	int i;
	MAP_CATEGORY::iterator it;
	for (it = mapCategory.begin(); it != mapCategory.end(); it++)	{
		int cat = it->first, i = it->second;
		printf("%d=>%d\tnz=%d,tic=%d,split=%g\n", cat,i, bins[i].nz, bins[i].tic, binFeatas[i].split_F);
	}
	/*for (i = 0; i < nBins; i++) {
		printf("%d\tnz=%d,tic=%d,split=%g\n",i,bins[i].nz, bins[i].tic, binFeatas[i].split_F);
	}*/
}

/*
	v0.1	cys
		8/27/2019
*/
HistoGRAM* HistoGRAM::FromDiff(const HistoGRAM*hP, const HistoGRAM*hBrother, bool isBuffer, int flag) {
	assert(hP->nBins >= hBrother->nBins || hP->nBins >= hBrother->nBins -1);		//hP会被压缩掉NA，需要继续核查
	//CopyBins(*hP, false, 0x0);
	//bins.reserve(hP->bins.size());
	int i = 0, nBin = hP->nBins,brother=0,nPass=0,nValid=0;
	double G_sum = 0, G_0 = 0, G_1 = 0;
	for (i = 0; i < nBin; i++) {
		//HISTO_BIN& cur = bins[i];
		HISTO_BIN cur = hP->bins[i];
		const HISTO_BIN* off = brother >= hBrother->nBins ? nullptr : &(hBrother->bins[brother]);
		if (off!=nullptr && cur.tic == off->tic) {
			assert(cur.nz >= off->nz);
			cur.nz -= off->nz;
			G_0 += cur.G_sum;		G_1 += off->G_sum;
			cur.G_sum -= off->G_sum;
			cur.H_sum -= off->H_sum;
			if (cur.nz == 0) {
				cur.G_sum = 0;	cur.H_sum = 0;
			}
			else {
				//assert();
			}
			G_sum += cur.G_sum;
			brother = brother + 1;
			//if (brother == hBrother->bins.size())		break;
		}
		if (cur.nz > 0) {
			bins[nValid++] = cur;
			/*if (isBuffer) {
			}else
				bins.push_back(cur);*/
		}	else {
			nPass = nPass + 1;
		}

	}
	if (isBuffer) {
		//assert(nValid <= nBins);
		//bins.resize(nValid);
		nBins = nValid;
	}
	//assert(brother == hBrother->bins.size());
	return this;
}
/*
void HistoGRAM::TicMap(tpQUANTI*map, int flag) {
	int i, tic;
	for (i = 0; i < nBins; i++) {
		tic = bins[i].tic;
		map[tic] = i;
	}
}*/

void HistoGRAM::CheckValid(const LiteBOM_Config&config, vector<BIN_FEATA>*featas,int flag) {
	int i,  nZ = 0, tic_0=-1;
	for (i = 0; i < nBins; i++) {		//这样最快
		//if (i<nBin - 1 && bins[i].nz == 0) { continue; }
		nZ += bins[i].nz;
		//assert(bins[i].nz>=config.min_data_in_bin || i>= nBin-2);
		assert(bins[i].tic > tic_0);
		if (i > 0 && featas!=nullptr) {
			assert((*featas)[i-1].split_F < (*featas)[i].split_F);
		}
		tic_0 = bins[i].tic;
	}
	assert(nZ==nSamp);
}

//似乎偏慢，需要提速
void HistoGRAM::CompressBins(int flag) {
	//GST_TIC(t1);
	/*vector<HISTO_BIN>::iterator iter = bins.begin();		//很慢
	while (iter != bins.end()) {
		if (iter->nz == 0) {
			iter = bins.erase(iter);
		}
		else {
			++iter;
		}
	}*/
	int i, nZ=0;
	/*vector<HISTO_BIN> binsN;
	for (i = 0; i < nBin; i++) {
		if (i<nBin -1 && bins[i].nz == 0) {
			continue;
		}	else {
			binsN.push_back(bins[i]);
		}
	}
	if (binsN.size() < bins.size())
		bins = binsN;
	else
		binsN.clear();*/
	for (i = 0; i < nBins; i++) {		//这样最快
		if (i<nBins - 1 && bins[i].nz == 0) {	continue;		}
		else {	nZ++;		}
		if (nZ <= i) {
			bins[nZ-1] = bins[i];		
		}
	}
	//assert(++nZ <= nBin);
	if (nZ < nBins) {
		nBins = nZ;
		//bins.resize(nZ);
	}
}


void HistoGRAM::MoreHisto(const FeatsOnFold *hData_, vector<HistoGRAM*>&more,  int flag) {
	return;
	/*
	assert(hFeat != nullptr);
	size_t minSet = hData_->config.min_data_in_leaf, nBin = bins.size(), i;
	HistoGRAM *H_1 = new HistoGRAM(hFeat, nSamp);
	H_1->bins = bins;
	vector<BIN_FEATA>& featas=hFeat->hDistri->binFeatas;
	assert( featas.size()==nBin );
	for (i = 0; i < nBin; i++) {
		//const HISTO_BIN &item = bins[i];
		const BIN_FEATA& feata = hFeat->hDistri->binFeatas[i];
		HISTO_BIN& bin_1 = H_1->bins[i];
		bin_1.split_F = feata.density;
	}
	sort(H_1->bins.begin(), H_1->bins.end(), HISTO_BIN::isSplitSmall);
	for (i = 0; i < nBin; i++) {
		H_1->bins[i].tic = i;
	}
	//sort_by_splitF;
	more.push_back(H_1);
	H_1->split_by = SPLIT_HISTOGRAM::BY_DENSITY;*/
}

void HistoGRAM::RandomCompress(FeatVector *hFV,bool isSwarm,int flag) {
	/*int nBins_0 = bins.size(), i,start=0;
	size_t nz = 0;
	double G_sum = 0, H_sum = 0;
	if (isSwarm) {
		vector<HISTO_BIN> binsN;
		assert(nBins_0== hFV->select_bins->DIM());
		vector<int> picks;
		hFV->select_bins->PickOnStep(nBins_0,picks,true);
		for (i = 0; i < nBins_0-1; i++) {
			if (picks[i]==0 && i<nBins_0-2) {
				nz+= bins[i].nz;
				G_sum += bins[i].G_sum;		H_sum += bins[i].H_sum;
				continue;
			}
			if(nz+bins[i].nz>0)	{
				bins[i].nz += nz;
				bins[i].G_sum += G_sum;				bins[i].H_sum += H_sum;

				binsN.push_back(bins[i]);
				nz = 0;		 G_sum = 0, H_sum = 0;
			}
		}
		binsN.push_back(bins[nBins_0 - 1]);
		if (binsN.size() < bins.size())
			bins = binsN;
		else
			binsN.clear();
	}
	else {
		if (true) {
			this->CompressBins();	//带来扰动，有意思
		}
		int nTo = max(bins.size()/4,4);	//
		while (bins.size() > nTo ) {
			int no = rand() % (bins.size() - 2);		//binNA总是放在最后
			HISTO_BIN&target = bins[no + 1],&src= bins[no];
			if (src.nz == 0)
				continue;
			target.nz += src.nz;
			target.G_sum += src.G_sum;
			target.H_sum += src.H_sum;
			bins.erase(bins.begin()+no);		
		}
	}
	if (hFV->select_bins != nullptr) {
		vector<double>& position = hFV->select_bins->cand.position;
		int nGate = position.size();
		for (i = 0; i < nGate; i++)	position[i] = 0;
		for (auto bin : bins) {
			assert(bin.tic < nGate);
			position[bin.tic] = 1;
		}
		i = 0;
	}*/
}

FRUIT::~FRUIT() {
	//if (histo != nullptr)
	//	delete histo;
	//if (bsfold != nullptr)
	//	delete bsfold;
	if (mapFolds != nullptr)
		delete[] mapFolds;
}

/**/
FRUIT::FRUIT(FeatsOnFold *hFold, const HistoGRAM *his_, int flag) : histo_refer(his_) {
	const FeatVector *hFeat = his_->hFeat;
	Distribution *hDistri = hFold->histoDistri(hFeat);
	int nMaxBin = hDistri->binFeatas.size(),i;
	//assert(nMaxBin == hFeat->hDistri->histo->nMostBins);
	//参见	FeatVec_T::SplitOn		if (fold <= 0)	left[nLeft++] = samp;
	mapFolds = new tpFOLD[nMaxBin]();
	//for (i = 0; i < nMaxBin; i++)		mapFolds[i] = 1;
	Set(hFold,his_);
}

/*
	v0.1	cys
		11/1/2019
*/
void FRUIT::Set(FeatsOnFold *hFold,const HistoGRAM*histo, int flag) {
	assert(histo != nullptr);
	const FeatVector *hFeat = histo->hFeat;
	Distribution *hDistri = hFold->histoDistri(hFeat);
	int nMaxBin = hDistri->binFeatas.size();
	best_feat_id = histo->hFeat->id;
	split_by = histo->split_by;
	//assert(fruit->bin_S0.nz>0);
	mxmxN = histo->fruit_info.mxmxN;			
	nLeft = histo->fruit_info.nLeft;		nRight = histo->fruit_info.nRight;
	//fruit->thrshold = item.tic;
	isY = histo->fruit_info.isY;
	if (isY) {	//参见GreedySplit_Y，不存在bin_S0,bin_S1
		assert(histo->hFeat->isCategory());
		histo_refer = histo;		
		//需要fold信息
		if (hFeat->isCategory()) {
			memset(mapFolds,0x0,sizeof(tpFOLD)*nMaxBin);
			for (int i = 0; i < histo->nBins; i++) {
				int pos = histo->bins[i].tic, fold = histo->bins[i].fold;
				mapFolds[pos] = fold;
			}
		}
		adaptive_thrsh = histo->fruit_info.adaptive_thrsh;
	}	else {
		int pos = histo->fruit_info.tic;
		assert(pos > 0 && pos < histo->nBins);
		tic_left = pos;
		bin_S0 = histo->bins[pos - 1];			bin_S1 = histo->bins[pos];
		//adaptive_thrsh = histo->split_F(bin_S1.tic);	// bin_S1.split_F;
		adaptive_thrsh = hDistri->split_F(bin_S1.tic);
	}
	//assert(fruit->adaptive_thrsh!= DBL_MAX);
	isNanaLeft = false;
}

void HistoGRAM::UpdateBestGain(int item_tic,double g1,size_t nLef,size_t nRight,int flag) {
	fruit_info.mxmxN = g1;
	fruit_info.nLeft = nLeft;		fruit_info.nRight = nRight;
	fruit_info.tic = item_tic;
	/*fruit->best_feat_id = hFeat->id;
	fruit->split_by = this->split_by;
	fruit->bin_S0 = bins[i-1];		fruit->bin_S1 = item;
	//assert(fruit->bin_S0.nz>0);
	g1 = MAX2(g, gainL);
	fruit->mxmxN = g1;			fruit->tic_left = item.tic;
	fruit->nLeft = nLeft;		fruit->nRight = nRight;
	//fruit->thrshold = item.tic;
	fruit->adaptive_thrsh = item.split_F;
	//assert(fruit->adaptive_thrsh!= DBL_MAX);
	fruit->isNanaLeft = false;
	if (gainL > g) {
	fruit->isNanaLeft = true;
	fruit->nLeft += binNA->nz;		fruit->nRight -= binNA->nz;
	assert(bins[(int)(item.tic) - 1].nz>0);

	}
	//fruit->lft_impuri = nLeft*lft*lft;		fruit->rgt_impuri = nRight*rgt*rgt;*/
}

/*
	v0.2	cys
		1/28/2019
*/
void HistoGRAM::GreedySplit_X(FeatsOnFold *hData_, const SAMP_SET& samp_set, int flag) {
	//GST_TIC(t1);
	//if (fruit == nullptr)
	//	throw "HistoGRAM::GreedySplit_X fruit is 0!!!";
	fruit_info.Clear();
	//if (nSamp == 139)		//仅用于调试
	//	nSamp = 139;
	size_t minSet = hData_->config.min_data_in_leaf,i;
	string optimal = hData_->config.leaf_optimal, obj = hData_->config.objective;
	//double sum = samp_set.Y_sum_1, a = a0, errL = 0, g, gL = 0, g1 = 0, lft, rgt;
	double gL = 0, gR0, hL = 0, hR = 0, a = -DBL_MAX, g, g1 = 0, split_0= -DBL_MAX;	//g1 = fruit->mxmxN
	nLeft = 0;				nRight = nSamp;		assert(nRight >= 0);
	HISTO_BIN*binNA = this->hBinNA();
	double gSum = 0, hSum = 0, lambda_l2= hData_->config.lambda_l2;
	double gainL = 0;		//对应于isNanaLeft
	//for (auto item : bins) {
	for (i = 0; i < nBins;i++) {
		gSum += bins[i].G_sum, hSum += bins[i].H_sum;
	}
	if (optimal == "lambda_0") {
		//assert(hSum == nSamp);
	}	else {
		assert(hSum == nSamp);
	}
	if(fabs(gSum+samp_set.Y_sum_1)>=1.e-6*fabs(gSum) && fabs(samp_set.Y_sum_1)>0.001)//不同的gSum计算确实有差异
		printf( "\tHistoGRAM::gSum is mismatch(%g-%g)", gSum,samp_set.Y_sum_1);

	//for (auto item : bins) {
	for (i = 0; i < nBins;i++) {
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
		//if (item.split_F == split_0 /*|| item.nz == 0*/) {			goto LOOP;		}
		//double errR = sum - errL;
		//Regression trees in CART have a constant numerical value in the leaves and use the variance as a measure of impurity
		//g = errL*errL / nLeft + errR*errR / nRight;
		g = gL*gL / (hL+lambda_l2) + gR*gR / (hR+ lambda_l2);
		if (false) {	//似乎并没有使结果更好 难以理解	 10/29/2018
			if (binNA->nz > 0 && nRight - binNA->nz >= minSet) {
				double eL = gL + binNA->G_sum, eR = gR - binNA->G_sum;
				gainL = eL*eL / (hL + lambda_l2 + binNA->H_sum) + eR*eR / (hR + lambda_l2 - binNA->H_sum);
			}
			else
				gainL = 0;
		}
		
		//double bin_w = hData_->rander_bins.Uniform_(0, 1);
		if (g>g1 || gainL>g1) {
			if (i == nBins - 1)		//仅用于测试
				i = nBins - 1;
			g1 = MAX2(g, gainL);
			UpdateBestGain(i, g1, nLeft, nRight);			
		}
	LOOP:
		//errL += item.Y_sum;		
		gL += item.G_sum;		hL += item.H_sum;
		a = item.tic;			//split_0 = item.split_F;
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
void HistoGRAM::GreedySplit_Y(FeatsOnFold *hData_, const SAMP_SET& samp_set, bool tryX, int flag) {
	char temp[2000];
	//FRUIT fruitX;
	FRUIT_INFO fruitX;
	HISTO_BIN*binNA = this->hBinNA();
	if (tryX) {
		GreedySplit_X(hData_, samp_set, flag);
		fruitX = fruit_info;	// *fruit;
	}
	else {
		fruitX.mxmxN = -DBL_MAX;
	}
	//return;
	size_t minSet = hData_->config.min_data_in_leaf;
	//double sum = -samp_set.Y_sum_1, errL = 0, g, g1 = 0, a;
	double gL = 0, gR0, hL = 0, hR = 0, a = -DBL_MAX, g, g1 = 0;
	nLeft = 0;	nRight = nSamp;
	double gSum = binNA->G_sum, hSum = binNA->H_sum;
	double gainL = 0;		//对应于isNanaLeft
	size_t i, nY_l = 0, nY_r = 0;
	for (i = 0; i < nBins - 1; i++){
	//for (auto item : bins) {
		gSum += bins[i].G_sum, hSum += bins[i].H_sum;
	}
	vector<tpSAMP_ID> idx;
	vector<double> Y_means;
	for (i = 0; i < nBins; i++) {
	//for (auto item : bins) {
		if (bins[i].nz == 0) {
			assert(bins[i].H_sum == 0 && bins[i].G_sum == 0);
			Y_means.push_back(DBL_MAX);	continue;
		}
		//a = item.G_sum / item.nz;
		a = bins[i].G_sum / bins[i].H_sum;
		Y_means.push_back(a);
	}
	//Y_means.push_back(DBL_MAX);		//always last bin for NA
	sort_indexes(Y_means, idx);
	for (i = 0; i < nBins-1; i++) {		assert(Y_means[idx[i]]<= Y_means[idx[i+1]]);	}
	for (i = 0; i < nBins; i++) {
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
			fruit_info.mxmxN = g1;
			//fruit_info.tic_left = Y_means[idx[i]];	// item.G_sum / item.nz;		assert(fruit->tic_left == Y_means[idx[i]]);
			fruit_info.nLeft = nLeft;		fruit_info.nRight = nRight;
			//fruit->thrshold = Y_means[idx[i]];
			fruit_info.adaptive_thrsh = Y_means[idx[i]];
			//sprintf(temp, "L(%g/%g %d) R(%g/%g %d)", gL, hL, nLeft, gR, hR, nRight );
			//fruit->sX = temp;
		}
	LOOP:
		//errL += item.G_sum;		a = item.tic;
		gL += item.G_sum;		hL += item.H_sum;
		assert(nRight >= item.nz);
		nLeft += item.nz;			nRight -= item.nz;
	}
	if (g1 == 0){	//Y的分布不一样，可能找不到split(例如minSet不符合)
			//fruitX.histo = nullptr;	
		return;	
	}

	double aX = fruitX.mxmxN - fabs(fruitX.mxmxN)*1.0e-4;		//浮点误差真麻烦
	//assert(fruit->mxmxN>= aX);
	if (fruit_info.mxmxN>aX) {	//确实有可能
	//	if (fruit->mxmxN>aX) {	//确实有可能
		fruit_info.isY = true;
		assert(hFeat->isCategory());
		for (i = 0; i < nBins; i++) {
			HISTO_BIN& item = bins[idx[i]];
			if (Y_means[idx[i]] < fruit_info.adaptive_thrsh) {
				item.fold = 0;
				nY_l += item.nz;
			}
			else {
				item.fold = 1;
				nY_r += item.nz;
			}
		}
		assert(nY_l== fruit_info.nLeft && nY_r== fruit_info.nRight);
	}
	else {
		if (fruit_info.mxmxN < fruitX.mxmxN)
			;// printf("!!!Split_X(%g)>Split_Y(%g)!!!", fruitX.mxmxN, fruit->mxmxN);
		fruit_info = fruitX;
	}
	//fruitX.histo = nullptr;/**/
	return;
}

/*
	v0.1	cys
		10/10/2018
*/
void HistoGRAM::Regress(const FeatsOnFold *hData_, const SAMP_SET& samp_set, int flag) {
	//if (fruit == nullptr)
	//	throw "HistoGRAM::GreedySplit_X fruit is 0!!!";

	size_t minSet = hData_->config.min_data_in_leaf,i;
	double sum = samp_set.Y_sum_1, g0 = sum*sum / nSamp, g1 = 0, mean_0 = sum / nSamp, a;
	//size_t nLeft = 0, nRight = nSamp,minSet=0;

//	for (auto item : bins) {
	for (i = 0; i < nBins; i++) {
		const HISTO_BIN &item = bins[i];
		if (item.nz == 0) {
			continue;
		}
		a = item.G_sum*item.G_sum / item.nz;
		g1 += a;
	}
	assert(g1 >= g0);
	//fruit->mxmxN = g1;
}


int HistoGRAM_BUFFER::NodeFeat2NO(int node, int feat)	const {
	int feat_pos = feat;
	if (!mapFeats.empty()) {
		feat_pos = mapFeats.at(feat);
		assert(feat_pos >= 0 && feat_pos < ldFeat_);
	}	
	
	int no = node*ldFeat_ + feat_pos;
	assert(no >= 0 && no < nzMost);

	return no;
}

/*
	v0.1	cys
		9/10/2019
*/
size_t HistoGRAM_BUFFER::SetBinsAtBuffer(const FeatsOnFold *hData_, vector<int>& pick_feats, int flag = 0x0) {
	size_t pos = 0, node,no;
	for (node = 0; node < nMostNode; node++) {
		for (auto feat: pick_feats) {
			no = NodeFeat2NO(node, feat);
			FeatVector *hFV = hData_->feats[feat];
			Distribution *hDistri = hData_->histoDistri(hFV);
			HistoGRAM *H_src = hDistri->histo,*histo = buffers[no];
			if (H_src == nullptr) {
				buffers[no] = nullptr;	 	continue;
			}
			if (histo == nullptr) {
				histo = new HistoGRAM(hFV, 0);			
				histo->buffer = this;		
			}	else {
				//assert(histo->nMostBins >= H_src->nBins);
				histo->hFeat = hFV;
			}
			histo->nMostBins = H_src->nBins;
			//histo->CopyBins(*(hFV->hDistri->histo), true, 0x0);
			histo->nBins = H_src->nBins;	
			histo->bins = bins_buffer + pos;		
			//histo->CopyBins(*(H_src), true, 0x0);
			pos += histo->nBins;

			buffers[no] = histo;
		}
	}
	return pos;
}

HistoGRAM_BUFFER::HistoGRAM_BUFFER(const FeatsOnFold *hData_0, int flag):hData_(hData_0){
	int nMostLeaf = hData_->config.num_leaves,node,feat,no,nzZ=0;
	nzMEM = 0;
	nMostNode = nMostLeaf * 2;
	nMostFeat = hData_->nFeat();	
	for (nMostBin = 0,feat = 0; feat < nMostFeat; feat++) {
		FeatVector *hFV = hData_->feats[feat];
		Distribution *hDistr = hData_->histoDistri(hFV);
		if (hDistr->histo == nullptr)		{
			nzZ++;		 continue;
		}
		nMostBin += hDistr->histo->nBins;
	}
	nMostBin *= nMostNode;
	bins_buffer = new HISTO_BIN[nMostBin];	nzMEM +=sizeof(HISTO_BIN)*nMostBin;

	ldFeat_ = nMostFeat;
	nzMost = nMostNode*nMostFeat;
	buffers.resize(nzMost);
	nzMEM += sizeof(HistoGRAM)*nzMost;
	vector<int> feats;
	feats.resize(nMostFeat);
	iota(feats.begin(), feats.end(), 0);
	size_t pos = SetBinsAtBuffer(hData_,feats);
	assert(pos==nMostBin);

	printf("\n********* HistoGRAM_BUFFER MEM=%.6g(M) nMostBin=%lld\n********* \tnMostFeat=%d,nMostNode=%d zero=%d\n", 
		nzMEM/1.0e6, nMostBin, nMostNode, nMostFeat, nzZ);
}

void HistoGRAM_BUFFER::BeforeTrainTree(vector<int>& pick_feats, size_t nPickSamp, int flag) {
	GST_TIC(t1);
	assert(pick_feats.size() <= nMostFeat);
	mapFeats.clear();
	int no = 0,node;
	if (pick_feats.size() < nMostFeat) {
		for (auto feat : pick_feats) {
			mapFeats.insert(pair<int, int>(feat, no++));
		}
		ldFeat_ = pick_feats.size();
		SetBinsAtBuffer(hData_, pick_feats, 0x0);
	}
	else
		ldFeat_ = nMostFeat;
	
	for (node = 0; node < nMostNode; node++) {
		for (auto feat : pick_feats) {
			no = NodeFeat2NO(node, feat);
			HistoGRAM *histo = buffers[no];
			assert(histo != nullptr);
			histo->nBins = 0;
			histo->nSamp = 0;
			histo->fruit_info.Clear();
		}
	}
		/*if (histo->bins.size() < histo->nMostBins) {
			histo->bins.resize(histo->nMostBins);
		}*/
		/*for (auto bin : histo->bins) {
			bin.nz = 0;
		}*/
}

HistoGRAM_BUFFER::~HistoGRAM_BUFFER() {
	Clear();
}

HistoGRAM*HistoGRAM_BUFFER::Get(int node, int feat, int flag)	const {
	assert(node >= 0 && node < nMostNode);
	assert(feat >= 0 && feat < nMostFeat);
	int no = NodeFeat2NO(node, feat);
	if (buffers[no] == nullptr)
		throw "HistoGRAM_BUFFER::Get is 0 !!!";
	return buffers[no];
}

/*void HistoGRAM_BUFFER::Set(int feat, HistoGRAM*histo) {

}*/

void HistoGRAM_BUFFER::Clear(int flag) {
	if (bins_buffer != nullptr)
		delete[] bins_buffer;
	for (auto histo : buffers) {
		if (histo == nullptr)
			continue;
		delete histo;
	}
	buffers.clear();
}
/*
void ManifoldTree::ClearHisto() {
for (auto node : nodes) {
for (auto pa : node->H_HISTO) {
HistoGRAM* histo = pa;
if (histo != nullptr)
delete histo;
}
node->H_HISTO.clear();
}
}
*/

//Sturge’s Rule	http://www.statisticshowto.com/choose-bin-sizes-statistics/
//Histogram Binwidth Optimization Method	http://176.32.89.45/~hideaki/res/histogram.html
//Optimal Data-Based Binning for Histograms
void HistoGRAM::OptimalBins(size_t nMost, size_t nSamp, double a0_, double a1_, int flag) {
	/*bins.clear();
	if (a0_ == a1_)
		return;
	if (nMost == 0) {
		//nMost = 3.49*sigma/pow(nSamp, 0.333);	//Rice’s Rule
		nMost = 1 + 2 * pow(nSamp, 0.333);	//Rice’s Rule
		nMost = 1 + 3.322*log(nSamp);	//Sturge’s Rule
	}
	assert(nMost >= 2);
	bins.resize(nMost);*/
	//a1 = a1_,		a0 = a0_;
}

/*需要合并*/
void HistoGRAM::ReSet(size_t nMost, int flag) {
	nSamp = nMost;
	delete[] bins;
	bins = new HISTO_BIN[nMost];
	nBins = 0;	 nMostBins = nMost;
	//bins.clear();
	//bins.resize(nMost);
	//a1 = -DBL_MAX, a0 = DBL_MAX;
}

/*
double  HistoGRAM::split_F(int no, int flag) const { 
	assert(hFeat != nullptr );
	Distribution *hDistr = hFeat->histoDistri();
	double a = hDistr->split_F(no,flag);
	return a; 
}
*/

void  HistoGRAM::CopyBins(const HistoGRAM &src, bool isReset, int flag) {
	//nSamp = src.nSamp;
	if (nMostBins >= src.nBins) {
		nBins = src.nBins;
	}	else {
		nBins = src.nBins;
		if (bins != nullptr)
		{		delete[] bins;	}
		bins = new HISTO_BIN[nBins];
		nMostBins = nBins;		
	}
	memcpy(bins, src.bins, sizeof(HISTO_BIN)*nBins);
	if (isReset) {
		for (int i = 0; i<nBins; i++) {
			HISTO_BIN &item = bins[i];
			item.nz = 0;
			//item.Y_sum = 0;		
			item.H_sum = 0;		item.G_sum = 0;
		}
	}
	//a1 = -DBL_MAX, a0 = DBL_MAX;
}