#pragma once

#include "EDA.hpp"
#include "./DataFold.hpp"
#include <limits>

using namespace Grusoft;

/*
	- If "mean", then replace missing values using the mean along
          the axis.
    - If "median", then replace missing values using the median along
        the axis.
    - If "most_frequent", then replace missing using the most frequent
        value along the axis.
*/
ExploreDA::ExploreDA(LiteBOM_Config&config,int nFeat_, int flag):nFeat(nFeat_){
	
	arrDistri.resize(nFeat);
	
}

Distribution::~Distribution() {
	//vThrsh.clear();
	vUnique.clear( );
	if(histo != nullptr)
		delete histo;
}

//
void Distribution::UpdateHistoByW(const LiteBOM_Config&config, float *wBins, int flag) {
	size_t nBin_0 = histo->nBins, i,nMaxSplit=int((nBin_0-1) /10.0),nSplit=0,id,nDrop=0;
	if (nMaxSplit == 0)
		return;
	double*comp_=new double[2* nBin_0](),w_avg=0;
	for (i = 0; i < nBin_0-1; i++) {	//最后一个无法拆分诶
		comp_[2 * i] = wBins[i];
		comp_[2 * i+1] = histo->bins[i].nz;
		w_avg += wBins[i] * wBins[i];
	}
	w_avg = sqrt(w_avg / (nBin_0 - 1));
	int *mask = new int[nBin_0]();
	vector<tpSAMP_ID> idx;
	double split_1=0;
	//sort_indexes(nBin_0, wBins, idx);
	idx.resize(nBin_0 - 1);	iota(idx.begin(), idx.end(), 0);
	// sort indexes based on comparing values in v
	std::sort(idx.begin(), idx.end(), [&comp_](size_t i1, size_t i2)
		{return (comp_[2*i1] < comp_[2 * i2]) || (comp_[2 * i1] == comp_[2 * i2] && comp_[2 * i1+ 1] < comp_[2 * i2+ 1]); }
	);
	for (i = 0; i < nBin_0-2; i++) {
		int k_1 = idx[i], k_2 = idx[i+1];
		assert(wBins[k_1]<wBins[k_2] || (wBins[k_1]==wBins[k_2] && histo->bins[k_1].nz<=histo->bins[k_2].nz));
	}
	for (i = 0; i < nMaxSplit; i++) {
		id = idx[nBin_0-2-i];
		bool isSplit = false;
		if (wBins[id] > w_avg ) {
			isSplit = true;
		}	else if (rand() % 10 == 0) {
			isSplit = true;
		}
		if(isSplit){
			assert(id != nBin_0 - 1);
			mask[id] = -1;
			nSplit++;
		}
	}

	assert(nSplit > 0);
	//if(nBin_0 + nSplit>config.feat_quanti)
	if (nBin_0 + nSplit > histo->nMostBins)
		nDrop = nBin_0 + nSplit - histo->nMostBins;	// config.feat_quanti;
	for (i = 0; i < nDrop; i++) {
		id = idx[i];
		assert(id != nBin_0 - 1);
		mask[id] = -2;
	}

	vector<HISTO_BIN> binX;
	for (i = 0; i < nBin_0-1; i++) {
		HISTO_BIN bin0 = histo->bins[i];
		assert(bin0.split_F < vMax);
		split_1 = i==nBin_0-1 ? DBL_MAX : histo->bins[i + 1].split_F;
		if (split_1 == DBL_MAX )
			split_1 = this->vMax;
		if(IS_NAN_INF(split_1))
			split_1 = this->vMax;
		if (mask[i] == -2) {
			continue;
		}
		binX.push_back(bin0);
		if (mask[i] == -1) {
			assert(i < nBin_0 - 1);
			HISTO_BIN bin1 = bin0;
			bin1.split_F = (bin0.split_F + split_1) / 2;
			bin1.split_F = bin0.split_F+(split_1- bin0.split_F)*0.618;
			binX.push_back(bin1);
		}
	}
	binX.push_back(histo->bins[nBin_0 - 1]);
	delete[] histo->bins;
	histo->nBins = binX.size();
	histo->bins = new HISTO_BIN[histo->nBins];
	for(i=0;i<histo->nBins;i++)
		histo->bins[i] = binX[i];

	size_t nBin = histo->nBins;
	//assert(nBin <= config.feat_quanti);
	assert(nBin <= histo->nMostBins);
	for (i = 0; i < nBin; i++) {
		histo->bins[i].tic = i;
		if (i < nBin - 1)
			assert(histo->bins[i].split_F<histo->bins[i+1].split_F);
	}
	delete[] mask;
	delete[] comp_;
	//printf("[+%d,-%d,nBin=%d=>%d]\t", nSplit, nDrop, nBin_0, nBin);
}

bool Distribution::isValidFeatas() {
	if (histo == nullptr)
		return false;
	if (binFeatas.size() != histo->nBins)
		return false;

	for (auto feata : binFeatas) {
		if (IS_NAN_INF(feata.density))
			return false;
	}
	return true;
}


void Distribution::Dump(int feat, bool isQuanti, int flag) {
	char tmp[2000] = "";
	if (rSparse>0 || rNA>0) {
		sprintf(tmp, "\tsparse=%.3g,nana=%.2g",rSparse, rNA);
	}
	char typ = BIT_TEST(type, Distribution::CATEGORY) ? '#' : ' ';
	if (isQuanti && histo!=nullptr) {
		size_t nBin=histo->nBins,n1 = ceil(nBin / 4.0), n2 = ceil(nBin / 2.0), n3 = ceil(nBin *3.0 / 4);
		HISTO_BIN&b0 = histo->bins[0], &b1 = histo->bins[n1], &b2 = histo->bins[n2], &b3 = histo->bins[n3], &b4 = histo->bins[nBin-1];
		if(b4.nz==0)	//最后一个BIN是冗余的
			b4= histo->bins[nBin - 2];
		printf("%4d %c%12s nBin=%d[%.3g(%d),%.3g(%ld),%.3g(%ld),%.3g(%ld),%.3g(%ld)]%s \n", feat, typ, nam.c_str(), 
			histo == nullptr ? 0 : histo->nBins,
			b0.tic,b0.nz, b1.tic, b1.nz, b2.tic, b2.nz, b3.tic, b3.nz, b4.tic, b4.nz,tmp);
	}	else {
		//printf("%4d %c%12s [%.3g,%.3g,%.3g,%.3g,%.3g]\tnBin=%d[%.3g,%.3g,%.3g,%.3g,%.3g]%s \n", feat, typ, nam.c_str(), 
		//	vMin, q1, q2, q3, vMax,
		//需要输出中位数
		printf("%4d %c%12s [%.4g-%.4g]\tBIG=%d\tnBin=%d[%.4g,%.4g,%.4g,%.4g,%.4g]%s \n", feat, typ, nam.c_str(),
			vMin, vMax, histo == nullptr ? 0 : histo->nBigBins /*corr.D_sum*/,
			 histo == nullptr ? 0 : histo->nBins,
			H_q0, H_q1, H_q2, H_q3, H_q4, tmp);
	}
}

void Distribution::HistoOnUnique_1(const LiteBOM_Config&config, vector<vDISTINCT>&uniques, size_t nA0, bool isMap, int flag) {
	isUnique = true;
	size_t nMostBin = uniques.size();
	assert(histo != nullptr);
	int noBin = 0;
	size_t i, i_0 = 0, nUnique = vUnique.size(), nz;
	double a0 = vUnique[0].val, a1 = vUnique[vUnique.size() - 1].val, v0;
	//mapCategory.clear();
	while (i_0 < nUnique) {
		v0 = vUnique[i_0].val;		
		HISTO_BIN& bin = histo->bins[noBin];
		bin.tic = noBin;
		bin.split_F = std::nextafter(v0, INFINITY);
		//mapCategory.insert(pair<int, int>((int)(v0), noBin));
		bin.nz = vUnique[i_0].nz;
		++i_0; ++noBin;
	}
	double delta = double(fabs(a1 - a0)) / nMostBin / 100.0;
	//histo->bins.resize(noBin + 1);		//always last bin for NA
	histo->nBins = noBin + 1;
	histo->bins[noBin].split_F = a1+delta;
	histo->bins[noBin].tic = noBin;
	histo->bins[noBin].nz = nSamp - nA0;
	histo->CheckValid(config);
}

/*
	always last bin for NA
	v0.1	cys
		8/30/2019
*/
void Distribution::HistoOnFrequncy_1(const LiteBOM_Config&config, vector<vDISTINCT>& vUnique,size_t nA0, size_t nMostBin, int flag) {
	assert(histo != nullptr);
	size_t nA = 0, T_avg = nA0*1.0 / nMostBin,SMALL_na=0, BIG_bins=0,nUnique= vUnique.size(),nz;
	for (int i = 0; i < nUnique;i++) {
		nA += vUnique[i].nz;
		if (vUnique[i].nz <= T_avg*2) {
			SMALL_na += vUnique[i].nz;
		}	else {
			vUnique[i].type = vDISTINCT::LARGE;	 BIG_bins++;
		}
	}
	//if (BIG_bins > 0)	while会自动更新
	//	T_avg = SMALL_na*1.0 / (nMostBin - BIG_bins);
	size_t T_avg_small = max(config.min_data_in_bin, SMALL_na / (nMostBin - BIG_bins)/ 10);
	T_avg_small = config.min_data_in_bin;

	histo->nBigBins = BIG_bins;

	size_t i_0 = -1,  noBin = 0, pos, nDistinc=0;
	double a0 = vUnique[0].val, a1 = vUnique[vUnique.size()-1].val, v0;
	double T_min_decrimi = 0, crimi = 0;
	bool isDcrimi = corr.dcrimi != nullptr;
	if (isDcrimi) {
		T_avg = max(1, T_avg / 2);  T_min_decrimi = corr.D_sum / nMostBin;
	}
	while (++i_0 < nUnique) {
		v0 = vUnique[i_0].val;	nz = 0;		nDistinc = 0;
		HISTO_BIN& bin = histo->bins[noBin];
		BIN_FEATA& feata = binFeatas[noBin];
		bin.tic = noBin;	//tic split_F必须一致
		bin.split_F = i_0 > 0 ? (v0 + vUnique[i_0 - 1].val) / 2 : v0;
		//bin.split_F =  v0;
		T_avg = nMostBin - noBin > BIG_bins ? max(config.min_data_in_bin, SMALL_na / (nMostBin- noBin- BIG_bins)) : config.min_data_in_bin;
		T_avg = max(T_avg, T_avg_small) ;	//T_avg会越来越小
		if (isDcrimi) {
			crimi = corr.dcrimi[i_0];
		}
		do	{
			if (vUnique[i_0].type == vDISTINCT::LARGE) {
				nz += vUnique[i_0].nz;
				BIG_bins--;		break;
			}			
			SMALL_na -= vUnique[i_0].nz;
			nz += vUnique[i_0].nz;			nDistinc++;
			if (isDcrimi) {
				if (nz >= T_avg && crimi > T_min_decrimi) {
					break;
				}
				crimi += corr.dcrimi[i_0];
			}
			else	if (nz >= T_avg )
				break;
			if (isUnique && nz>= T_avg_small)
				break;
			if (i_0+1<nUnique && vUnique[i_0+1].type == vDISTINCT::LARGE && nz>T_avg / 2)
			{		break;			}
		} while (++i_0 < nUnique);
		
		//assert(i_1 == nUnique );
		//、assert(nz >= config.min_data_in_bin || i_0 == nUnique);
		bin.nz = nz;		
		noBin = noBin + 1;
		feata.density = nz*1.0 / nDistinc;
	}
	assert(SMALL_na>=0 && BIG_bins==0);
	assert(i_0 == nUnique+1 || i_0 == nUnique);
	double delta = double(fabs(a1 - a0)) / nMostBin / 100.0;
	double d_max = DBL_MAX;	// std::numeric_limits<double>::max();
	if (nUnique < nMostBin && nA>nMostBin*10) {
		d_max = a1 + delta;
		isUnique = true;
		//assert(histo->bins.size()== nUnique);
	} else {
	}

	//histo->bins.resize(noBin + 1);		//always last bin for NA
	histo->nBins = noBin + 1;
	histo->bins[noBin].split_F = d_max;	
	histo->bins[noBin].tic = noBin;
	histo->bins[noBin].nz = nSamp-nA;
	histo->CheckValid(config);
	/*nz = histo->bins.size();
	if (nz >= 2) {
		size_t n1 = ceil(nz / 4.0), n2 = ceil(nz / 2.0), n3 = ceil(nz *3.0 / 4)-1;
		HISTO_BIN&b0 = histo->bins[0], &b1 = histo->bins[n1], &b2 = histo->bins[n2], &b3 = histo->bins[n3], &b4 = histo->bins[nz-1];
		H_q0 = b0.split_F,				H_q4 = b4.split_F;
		H_q1 = q1 = b1.split_F,			H_q2 = q2 = b2.split_F;		H_q3 = q3 = b3.split_F;
	}*/

}

#define IS_INT(dtype) (true)
#define CAST_ON_STR(x, dtype)	IS_INT(dtype) ? (int*)(x):(float*)(x)

