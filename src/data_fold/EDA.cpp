#pragma once

#include "EDA.hpp"
#include "./DataFold.hpp"
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

void Distribution::Dump(int feat, bool isQuanti, int flag) {
	char tmp[2000] = "";
	if (rSparse>0 || rNA>0) {
		sprintf(tmp, "\tsparse=%.3g,nana=%.2g",rSparse, rNA);
	}
	char typ = BIT_TEST(type, Distribution::CATEGORY) ? '#' : ' ';
	if (isQuanti && histo!=nullptr) {
		size_t nBin=histo->bins.size(),n1 = ceil(nBin / 4.0), n2 = ceil(nBin / 2.0), n3 = ceil(nBin *3.0 / 4);
		HISTO_BIN&b0 = histo->bins[0], &b1 = histo->bins[n1], &b2 = histo->bins[n2], &b3 = histo->bins[n3], &b4 = histo->bins[nBin-1];
		if(b4.nz==0)	//最后一个BIN是冗余的
			b4= histo->bins[nBin - 2];
		printf("%4d %c%12s nBin=%d[%.3g(%d),%.3g(%ld),%.3g(%ld),%.3g(%ld),%.3g(%ld)]%s \n", feat, typ, nam.c_str(), 
			histo == nullptr ? 0 : histo->bins.size(),
			b0.tic,b0.nz, b1.tic, b1.nz, b2.tic, b2.nz, b3.tic, b3.nz, b4.tic, b4.nz,tmp);
	}	else {
		//printf("%4d %c%12s [%.3g,%.3g,%.3g,%.3g,%.3g]\tnBin=%d[%.3g,%.3g,%.3g,%.3g,%.3g]%s \n", feat, typ, nam.c_str(), 
		//	vMin, q1, q2, q3, vMax,
		printf("%4d %c%12s [%.3g,%.3g,%.3g]\tD=%.3g\tnBin=%d[%.3g,%.3g,%.3g,%.3g,%.3g]%s \n", feat, typ, nam.c_str(),
			vMin, q2, vMax,corr.D_sum,
			 histo == nullptr ? 0 : histo->bins.size(),
			H_q0, H_q1, H_q2, H_q3, H_q4, tmp);
	}
}

void Distribution::HistoOnFrequncy_1(const LiteBOM_Config&config, vector<_BIN_>& vUnique,size_t nA0, size_t nMostBin, int flag) {
	assert(histo != nullptr);
	size_t nA = 0, T_avg = nA0*1.0 / nMostBin,SMALL_na=0, BIG_bins=0,nUnique= vUnique.size(),nz;
	for (int i = 0; i < nUnique;i++) {
		nA += vUnique[i].nz;
		if (vUnique[i].nz <= T_avg) {
			SMALL_na += vUnique[i].nz;
		}	else {
			vUnique[i].type = _BIN_::LARGE;	 BIG_bins++;
		}
	}

	size_t i_0 = -1,  noBin = 0, pos;
	double a0 = vUnique[0].val, a1 = vUnique[vUnique.size()-1].val, v0;
	double T_min_decrimi = 0, crimi = 0;
	bool isDcrimi = corr.dcrimi != nullptr;
	if (isDcrimi) {
		T_avg = max(1, T_avg / 2);  T_min_decrimi = corr.D_sum / nMostBin;
	}
	while (++i_0 < nUnique) {
		v0 = vUnique[i_0].val;	nz = 0;
		HISTO_BIN& bin = histo->bins[noBin];
		bin.tic = noBin;	//tic split_F必须一致
		bin.split_F = i_0 > 0 ? (v0 + vUnique[i_0 - 1].val) / 2 : v0;
		T_avg = nMostBin - noBin > BIG_bins ? max(1, SMALL_na / (nMostBin- noBin- BIG_bins)) : 1;
		if (isDcrimi) {
			crimi = corr.dcrimi[i_0];
		}
		do	{
			if (vUnique[i_0].type == _BIN_::LARGE) {
				BIG_bins--;		break;
			}			
			SMALL_na -= vUnique[i_0].nz;
			nz += vUnique[i_0].nz;
			if (isDcrimi) {
				if (nz >= T_avg && crimi > T_min_decrimi) {
					break;
				}
				crimi += corr.dcrimi[i_0];
			}
			else	if (nz >= T_avg)
				break;
		} while (++i_0 < nUnique);
		
		//assert(i_1 == nUnique );
		bin.nz = nz;		
		noBin = noBin + 1;
	}
	assert(SMALL_na>=0 && BIG_bins==0);
	histo->bins.resize(noBin + 1);
	assert(i_0 == nUnique+1 || i_0 == nUnique);
	double delta = double(fabs(a1 - a0)) / nMostBin / 100.0;
	histo->bins[noBin].split_F = a1 + delta;		//上界,为了QuantiAtEDA等
													//assert(histo->bins[histo->bins.size()-1].split_F>a1);
													//vThrsh.push_back(a1 + delta);
}