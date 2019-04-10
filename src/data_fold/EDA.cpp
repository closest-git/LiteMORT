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
