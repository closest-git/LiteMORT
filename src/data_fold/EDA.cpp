#pragma once

#include "EDA.hpp"
#include "./DataFold.hpp"
#include <limits>

using namespace Grusoft;
GRander Distribution::rander_(42);
/*
	- If "mean", then replace missing values using the mean along
          the axis.
    - If "median", then replace missing values using the median along
        the axis.
    - If "most_frequent", then replace missing using the most frequent
        value along the axis.
*/
ExploreDA::ExploreDA(LiteBOM_Config&config, int flag){

	//arrDistri.resize(nFeat);
	
}
ExploreDA::~ExploreDA() {
	for (auto distri : mapDistri) {
		assert(distri.second!=nullptr);
		delete distri.second;
	}
	mapDistri.clear();
}

void ExploreDA::AddDistri(const string&name, int id, int flag) {
	assert(mapDistri.find(id) == mapDistri.end());

	Distribution *distri = new Distribution();
	distri->nam = name;
	mapDistri.insert(pair<int, Distribution*>(id, distri));
}

Distribution* ExploreDA::GetDistri(int id)	{
	if (mapDistri.find(id) == mapDistri.end()) {
		printf("\nExploreDA::GetDistri id=%d is XXX\n", id);
		throw "!!!!!! ExploreDA::GetDistri id is XXX	!!!!!!";
	}
	return mapDistri[id];
}


Distribution::~Distribution() {
	//vThrsh.clear();
	binFeatas.clear();
	sortedA.clear();
	vUnique.clear( );
	mapCategory.clear();
	if(histo != nullptr)
		delete histo;
}

double Distribution::split_F(int no, int flag) const {
	assert(no >= 0 && no < binFeatas.size());
	double a = binFeatas[no].split_F;
	return a;
}
//
void Distribution::UpdateHistoByW(const LiteBOM_Config&config, int nTree, float *wBins, int flag) {
	//histo->nMostBins = config.feat_quanti;
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
		}
		else if (rander_.Uniform_(0,1)<=0.1) {
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
	vector<BIN_FEATA> feataX;
	assert(binFeatas.size() == histo->nBins);
	for (i = 0; i < nBin_0-1; i++) {
		HISTO_BIN bin0 = histo->bins[i];
		//assert(bin0.split_F < vMax);
		assert(binFeatas[i].split_F < vMax);
		split_1 = i == nBin_0 - 1 ? DBL_MAX : binFeatas[i+1].split_F;	// histo->bins[i + 1].split_F;
		if (split_1 == DBL_MAX )
			split_1 = this->vMax;
		if(IS_NAN_INF(split_1))
			split_1 = this->vMax;
		if (mask[i] == -2) {
			continue;
		}
		feataX.push_back(binFeatas[i]);
		binX.push_back(bin0);
		if (mask[i] == -1) {
			assert(i < nBin_0 - 1);
			HISTO_BIN bin1 = bin0;
			//bin1.split_F = (bin0.split_F + split_1) / 2;
			//bin1.split_F = bin0.split_F+(split_1- bin0.split_F)*0.618;
			BIN_FEATA f_;
			f_.split_F = binFeatas[i].split_F + (split_1 - binFeatas[i].split_F) * 0.618;
			feataX.push_back(f_);
			binX.push_back(bin1);
		}
	}
	binX.push_back(histo->bins[nBin_0 - 1]);
	BIN_FEATA f_;
	f_.split_F = binFeatas[nBin_0 - 1].split_F;
	feataX.push_back(f_);
	delete[] histo->bins;
	binFeatas = feataX;
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
			assert(binFeatas[i].split_F<binFeatas[i + 1].split_F);
			//assert(histo->bins[i].split_F<histo->bins[i+1].split_F);
	}
	delete[] mask;
	delete[] comp_;
	if (nTree % 50 == 0)
		 printf("[+%d,-%d,nBin=%d=>%d]\t", nSplit, nDrop, nBin_0, nBin);
	assert(binFeatas.size() == histo->nBins);

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

bool Distribution::VerifySame(const Distribution& rhs) const {
	if (nSamp != rhs.nSamp)
		return false;
	if (type != rhs.type)
		return false;
	//if (histo != rhs.histo)
	//	return false;
	
	if(mapCategory != rhs.mapCategory)
		return false;
	if (binFeatas != rhs.binFeatas)
		return false;/**/
	if (nam != rhs.nam)
		return false;
	if (desc != rhs.desc)
		return false;
	return false;
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
		printf("%4d %c%12s nBin=%d[%d(%ld),%d(%ld),%d(%ld),%d(%ld),%d(%ld)]%s \n", feat, typ, nam.c_str(), 
			histo == nullptr ? 0 : histo->nBins,
			b0.tic,b0.nz, b1.tic, b1.nz, b2.tic, b2.nz, b3.tic, b3.nz, b4.tic, b4.nz,tmp);
	}	else {
		//printf("%4d %c%12s [%.3g,%.3g,%.3g,%.3g,%.3g]\tnBin=%d[%.3g,%.3g,%.3g,%.3g,%.3g]%s \n", feat, typ, nam.c_str(), 
		//	vMin, q1, q2, q3, vMax,
		//需要输出中位数
		printf("%4d %c%12s [%.4g-%.8g]\tBIG=%d\tnBin=%d[%.4g,%.4g,%.4g,%.4g,%.4g]%s \n", feat, typ, nam.c_str(),
			vMin, vMax, histo == nullptr ? 0 : histo->nBigBins /*corr.D_sum*/,
			 histo == nullptr ? 0 : histo->nBins,
			H_q0, H_q1, H_q2, H_q3, H_q4, tmp);
	}
}

void Distribution::HistoOnUnique_1(const LiteBOM_Config&config, vector<vDISTINCT>&uniques, size_t nA0, bool isMap, int flag) {
	isUnique = true;
	size_t nMostBin = uniques.size();
	assert(histo != nullptr);
	//int noBin = 0;
	size_t i, i_0 = 0, nUnique = vUnique.size(), nz;
	double a0 = vUnique[0].val, a1 = vUnique[vUnique.size() - 1].val, v0, vLeftOuter;
	if(isMap)
		mapCategory.clear();
	while (i_0 < nUnique) {
		v0 = vUnique[i_0].val;	
		vLeftOuter = i_0 > 0 ? vUnique[i_0 - 1].val : v0;
		AddBin(config, vUnique[i_0].nz, vLeftOuter, v0, flag);
		/*HISTO_BIN& bin = histo->bins[noBin];
		bin.tic = noBin;
		//bin.split_F = std::nextafter(v0, INFINITY);
		binFeatas[noBin].split_F = std::nextafter(v0, INFINITY);
		bin.nz = vUnique[i_0].nz;*/
		if (isMap)	mapCategory.insert(pair<int, int>((int)(v0), histo->nBins-1));
		++i_0; //++noBin;
	}
	double delta = double(fabs(a1 - a0)) / nMostBin / 100.0;
	AddBin(config, nSamp - nA0, a1, a1 + delta, nSamp==nA0?-1:flag);			//always last bin for NA
	/*//histo->bins.resize(noBin + 1);		
	histo->nBins = noBin + 1;
	//histo->bins[noBin].split_F = a1+delta;
	binFeatas[noBin].split_F = a1 + delta;
	histo->bins[noBin].tic = noBin;
	histo->bins[noBin].nz = nSamp - nA0;*/
	int nBin = histo->nBins;	// bins.size();		//always last bin for NA
	histo->nMostBins = histo->nBins;	// bins.size();
	assert(binFeatas.size() >= nBin);
	binFeatas.resize(nBin);
	histo->CheckValid(config);
}

BIN_FEATA& Distribution::AddBin(const LiteBOM_Config&config, size_t nz,double left_outer,double left_inner, int flag) {
	assert(left_inner >= left_outer);
	if (nz > 0) {
	}	else {
		assert(flag==-1);		//always last bin for NA
	}
	int noBin = histo->nBins;
	HISTO_BIN& bin = histo->bins[noBin];
	BIN_FEATA& feata = binFeatas[noBin];
	bin.tic = noBin;	//tic split_F必须一致
						//bin.split_F = i_0 > 0 ? (v0 + vUnique[i_0 - 1].val) / 2 : v0;
	feata.split_F = (left_outer + left_inner) / 2 ;
	bin.nz = nz;
	//feata.density = nz*1.0 / nDistinc;

	histo->nBins++;
	assert(histo->nBins<=binFeatas.size());
	return feata;
}

int Distribution::HistoOnFrequncy_small(const LiteBOM_Config&config, vector<vDISTINCT>& vUnique, int i_0,int i_1, size_t T_bin, int flag) {
	size_t nz = 0,mimimum=config.min_data_in_bin,nBin0 = histo->nBins;
	double v0;
	int i;
	while (i_0 <= i_1) {
		nz = 0;
		v0 = vUnique[i_0].val;
		if (i_0 == 228) {	//仅用于调试
			i_0 = 228;
		}
		for (i = i_0; i <= i_1; i++) {
			assert(vUnique[i].type != vDISTINCT::LARGE);
			nz += vUnique[i].nz;
			if (nz >= T_bin)
				break;
			if (isUnique && nz >= mimimum)
				break;
		}
		AddBin(config,nz, i_0>0 ? vUnique[i_0 - 1].val : v0, v0, 0x0);
		i_0 = i + 1;
	}
	return histo->nBins - nBin0;
}

/*
	always last bin for NA
	v0.1	cys
		 8/30/2019
	v0.2	cys
		10/31/2019
*/
void Distribution::HistoOnFrequncy_1(const LiteBOM_Config&config, vector<vDISTINCT>& vUnique, size_t nA0, size_t nMostBin, int flag) {
	nMostBin = MIN2(vUnique.size(), nMostBin);
	assert(histo != nullptr);
	size_t nA = 0, avg = nA0*1.0 / nMostBin, SMALL_na_0 = 0, BIG_bins_0 = 0, nUnique = vUnique.size(), nz, minimum=config.min_data_in_bin, T_222;
	//size_t
	double a0 = vUnique[0].val, a1 = vUnique[vUnique.size() - 1].val;
	for (int i = 0; i < nUnique; i++) {
		nA += vUnique[i].nz;
		if (vUnique[i].nz <= avg) {
			SMALL_na_0 += vUnique[i].nz;
		}	else {
			vUnique[i].type = vDISTINCT::LARGE;	 BIG_bins_0++;
			//printf("%.4g=%d\t", vUnique[i].val, vUnique[i].nz);
		}
	}
	assert(BIG_bins_0<nMostBin);
	histo->nBigBins = BIG_bins_0;
	size_t SMALL_nRight = SMALL_na_0, BIG_nRight = BIG_bins_0,T_next;
	double T_base,vLeftInner,vLeftOuter;
	int nBin = 0, i_0=0,i_1;
	
	if(SMALL_nRight==0)
		T_base = nA*1.0 / BIG_nRight;
	else
		T_base = SMALL_nRight*1.0 / (nMostBin-histo->nBins - BIG_nRight);
	assert(T_base > 1);
	//T_base = MAX2(T_base,minimum);
	while (i_0 < nUnique) {
		vLeftInner = vUnique[i_0].val;
		vLeftOuter = i_0 > 0 ? vUnique[i_0 - 1].val : vLeftInner;
		nz = 0;		i_1 = i_0;
		if (histo->nBins == nMostBin)
			T_next = nA *10;
		else {
			if (SMALL_nRight == 0)
				T_base = nA*1.0 / BIG_nRight;
			else
				T_base = SMALL_nRight*1.0 / (nMostBin - histo->nBins - BIG_nRight);
			T_next = size_t(T_base+0.5);
		}
		T_next = MAX2(T_next, minimum);
		while (nz < T_next && i_1<nUnique) {
			if (vUnique[i_1].type == vDISTINCT::LARGE) {
				BIG_nRight--;
			}	else {
				SMALL_nRight -= vUnique[i_1].nz;
			}
			nz += vUnique[i_1++].nz;
		}
		if (T_next>nA) {
			printf("\tHisto  undesirable BIN(nz=%lld)@[%d-%d] nUnique=%d\n", nz, i_0, i_1, nUnique);
		}
		vDISTINCT& last = vUnique[i_1 - 1];
		if (i_1==i_0+1 || last.type != vDISTINCT::LARGE) {
			AddBin(config, nz, vLeftOuter, vLeftInner, 0x0);
		}	else {	//maybe split
			size_t nLeft = nz - last.nz;
			if (nLeft > T_base / 2) {
				AddBin(config, nLeft, vLeftOuter, vLeftInner, 0x0);
				vLeftOuter= vUnique[i_1-2].val; vLeftInner = last.val;
				AddBin(config, last.nz, vLeftOuter, vLeftInner, 0x0);
			}
			else {
				AddBin(config, nz, vLeftOuter, vLeftInner, 0x0);
			}
		}
		i_0 = i_1;		
	}
	
	assert(SMALL_nRight ==0 && BIG_nRight ==0);
	double delta = double(fabs(a1 - a0)) / nMostBin / 100.0;
	double d_max = DBL_MAX;	// std::numeric_limits<double>::max();
	if (nUnique < nMostBin && nA>nMostBin * 10) {
		d_max = a1 + delta;
		isUnique = true;
		//assert(histo->bins.size()== nUnique);
	}
	else {
	}
	AddBin(config,nSamp - nA, a1,d_max, -1);		//always last bin for NA
	histo->CheckValid(config,&binFeatas);
}

/*
void Distribution::HistoOnFrequncy_1(const LiteBOM_Config&config, vector<vDISTINCT>& vUnique, size_t nA0, size_t nMostBin, int flag) {
	assert(histo != nullptr);
	size_t nA = 0, T_avg = nA0*1.0 / nMostBin, SMALL_na = 0, BIG_bins = 0, nUnique = vUnique.size(), nz;
	for (int i = 0; i < nUnique; i++) {
		nA += vUnique[i].nz;
		if (vUnique[i].nz <= T_avg * 2) {
			SMALL_na += vUnique[i].nz;
		}
		else {
			vUnique[i].type = vDISTINCT::LARGE;	 BIG_bins++;
		}
	}
	//if (BIG_bins > 0)	while会自动更新
	//	T_avg = SMALL_na*1.0 / (nMostBin - BIG_bins);
	size_t T_avg_small = MAX2(config.min_data_in_bin, SMALL_na / (nMostBin - BIG_bins) / 10);
	T_avg_small = config.min_data_in_bin;

	histo->nBigBins = BIG_bins;

	size_t i_0 = -1, noBin = 0, pos, nDistinc = 0;
	double a0 = vUnique[0].val, a1 = vUnique[vUnique.size() - 1].val, v0;
	double T_min_decrimi = 0, crimi = 0;
	bool isDcrimi = corr.dcrimi != nullptr;
	if (isDcrimi) {
		T_avg = MAX2(1, T_avg / 2);  T_min_decrimi = corr.D_sum / nMostBin;
	}
	while (++i_0 < nUnique) {
		v0 = vUnique[i_0].val;	nz = 0;		nDistinc = 0;
		HISTO_BIN& bin = histo->bins[noBin];
		BIN_FEATA& feata = binFeatas[noBin];
		bin.tic = noBin;	//tic split_F必须一致
							//bin.split_F = i_0 > 0 ? (v0 + vUnique[i_0 - 1].val) / 2 : v0;
		feata.split_F = i_0 > 0 ? (v0 + vUnique[i_0 - 1].val) / 2 : v0;
		//bin.split_F =  v0;
		T_avg = nMostBin - noBin > BIG_bins ? MAX2(config.min_data_in_bin, SMALL_na / (nMostBin - noBin - BIG_bins)) : config.min_data_in_bin;
		T_avg = MAX2(T_avg, T_avg_small);	//T_avg会越来越小
		if (isDcrimi) {
			crimi = corr.dcrimi[i_0];
		}
		do {
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
			else	if (nz >= T_avg)
				break;
			if (isUnique && nz >= T_avg_small)
				break;
			if (i_0 + 1<nUnique && vUnique[i_0 + 1].type == vDISTINCT::LARGE && nz>T_avg / 2)
			{
				break;
			}
		} while (++i_0 < nUnique);

		//assert(i_1 == nUnique );
		//、assert(nz >= config.min_data_in_bin || i_0 == nUnique);
		bin.nz = nz;
		noBin = noBin + 1;
		feata.density = nz*1.0 / nDistinc;
	}
	assert(SMALL_na >= 0 && BIG_bins == 0);
	assert(i_0 == nUnique + 1 || i_0 == nUnique);
	double delta = double(fabs(a1 - a0)) / nMostBin / 100.0;
	double d_max = DBL_MAX;	// std::numeric_limits<double>::max();
	if (nUnique < nMostBin && nA>nMostBin * 10) {
		d_max = a1 + delta;
		isUnique = true;
		//assert(histo->bins.size()== nUnique);
	}
	else {
	}

	//histo->bins.resize(noBin + 1);		//always last bin for NA
	histo->nBins = noBin + 1;
	//histo->bins[noBin].split_F = d_max;	
	binFeatas[noBin].split_F = d_max;
	histo->bins[noBin].tic = noBin;
	histo->bins[noBin].nz = nSamp - nA;
	histo->CheckValid(config);

}*/

#define IS_INT(dtype) (true)
#define CAST_ON_STR(x, dtype)	IS_INT(dtype) ? (int*)(x):(float*)(x)

