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

EnsemblePruning::EnsemblePruning(int n, int m, int flag) : nSamp(n), nWeak(n) {
	U = new float[nSamp*nWeak];
	w_0 = new float[nWeak];
	w = new float[nWeak];
}

void EnsemblePruning::Pick(int T,int flag){
	int nPick = nWeak,nLarge=nSamp/3,i,no,k, nZero;
	double sum = 0;
	short sigma = 0;
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


};