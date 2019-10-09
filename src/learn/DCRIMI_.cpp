#include <memory>
#include <iostream>
#include <algorithm>
//#include <tchar.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <fstream>
#include "DCRIMI_.hpp"
#include "../util/GST_def.h"

using namespace Grusoft;
using namespace std;

//double DCRIMI_::tX = 0;
double DCRIMI_2::tX = 0;

/*
	Copyright 2008-present, Grusoft.
	v0.1	cys
		6/13/2015
*/
//int DCRIMI_::nSPAN = 1000;
DCRIMI_::DCRIMI_(void *hB, int span, int flag) : D_span(span), hBase(hB), dump(1), isSaveFalse(false), isBidui(false) {
	D_span = span;			hBase = hB;			dump = 1;
	assert(span >= 100 && hBase != nullptr);
	D_inter = new float[D_span + 1];		D_intra = new float[D_span + 1];
	for (int i = 0; i <= D_span; i++) {
		D_inter[i] = 0.0;		D_intra[i] = 0.0;
	}
}
DCRIMI_::DCRIMI_(const DCRIMI_& dcri) :isSaveFalse(dcri.isSaveFalse) {
	memset(this, 0x0, sizeof(DCRIMI_));

	rFAR = dcri.rFAR, rFRR = dcri.rFRR, rEER = dcri.rEER,
		D_ = dcri.D_, sep = dcri.sep, eer_sep = dcri.eer_sep;
	mean_a = dcri.mean_a, mean_r = dcri.mean_r;
	devia_a = dcri.devia_a, devia_r = dcri.devia_r;
	dump = dcri.dump;
}

void DCRIMI_::Init(int flag) {
	for (int i = 0; i <= D_span; i++) {
		D_inter[i] = 0.0;		D_intra[i] = 0.0;
	}
	rFAR = 0.0, rFRR = 0.0, rEER = 0.0, D_ = 0.0, sep = 0.0, eer_sep = 0.0;
	rTop_1 = 0.0;			rTop_5 = 0.0;
}


void DCRIMI_::Insert_1(float dis, bool isIntra, int flag) {
	assert(dis>-0.001 && dis<1.001);
	int pos = dis*D_span;
	pos = max(pos, 0);		pos = min(pos, D_span);
	if (isIntra) {
		D_intra[pos]++;
	}
	else {
		D_inter[pos]++;
	}

}
void DCRIMI_::Analyze(const string &sTitle, int flag) {
	assert(D_inter != nullptr && D_intra != nullptr);
	int i, grid = 0;
	double s, D_s, f_ar, f_rr, w_a, w_r, f_ar_g = 1.0e-7;

	for (i = 0; i<8; i++) {
		f_ar_8[i] = -1.0;			f_rr_8[i] = -1.0;				hd_8[i] = -1.0;
	}
	D_s = 1.0 / D_span;		//only for hamming distance

	mean_a = 0.0;		mean_r = 0;
	nz_a = 0.0;			nz_r = 0.0;
	max_a = max_r = 0.0;		min_a = min_r = 1.0;
	for (i = 0; i <= D_span; i++) {
		mean_a += i*D_intra[i];		nz_a += D_intra[i];
		if (D_intra[i]>0) {
			max_a = MAX2(max_a, i*D_s);		min_a = MIN2(min_a, i*D_s);
		}
		mean_r += i*D_inter[i];		nz_r += D_inter[i];
		if (D_inter[i]>0) {
			max_r = MAX2(max_r, i*D_s);		min_r = MIN2(min_r, i*D_s);
		}
	}
	//	nz_a = nz_a;				nz_r = nz_r;
	mean_a = nz_a == 0 ? 0.0 : mean_a / nz_a*D_s;
	mean_r = nz_r == 0 ? 0.0 : mean_r / nz_r*D_s;
	//	mean_a = mean_a;				mean_r = mean_r;

	devia_a = 0.0;		devia_r = 0;
	w_a = 0.0;			w_r = 0.0;
	for (i = 0; i <= D_span; i++) {
		w_a += D_intra[i];		w_r += D_inter[i];
		f_ar = w_r*1.0 / nz_r;
		f_rr = (nz_a - w_a)*1.0 / nz_a;
		while (f_ar >= f_ar_g && f_ar_8[grid] == -1.0) {
			f_ar_8[grid] = f_ar;
			f_rr_8[grid] = f_rr;				hd_8[grid] = i*1.0 / D_span;
			f_ar_g *= 10;		grid++;
			if (f_ar < f_ar_g)
				break;
		}
		if (f_ar >= f_rr && rEER == 0.0) {
			rEER = (f_ar + f_rr) / 2.0;
			eer_sep = i*1.0 / D_span;//sep;
		}
		if (f_ar>1.0e-3 && sep == 0.0) {
			//if( f_ar>1.0e-2 && sep==0.0 )	{
			sep = i*1.0 / D_span;
			rFAR = f_ar;
			rFRR = f_rr;
		}
		if (D_intra[i] != 0) {
			s = (i*D_s - mean_a);
			devia_a += s*s*D_intra[i];
		}
		s = (i*D_s - mean_r);
		devia_r += s*s*D_inter[i];
	}
	devia_a = nz_a == 0 ? 0.0 : sqrt(devia_a / nz_a);
	devia_r = nz_r == 0 ? 0.0 : sqrt(devia_r / nz_r);
	//	devia_a = devian_a;				devia_r = devian_r;

	s = sqrt((devia_a*devia_a) + (devia_r*devia_r)) / 2.0;
	D_ = s == 0 ? 0.0 : fabs(mean_a - mean_r) / s;
	double accu = (1.0 - rFRR)*100.0;
	if (rFAR>2 * 1.0e-2) { accu /= (rFAR / 1.0e-2); }		//rFAR=1%
	if (dump != 0) {
		printf("\n@@@\"%s\" nz=(%g,%g) intra=(%.3g,%.3g,%.3g,%.3g),inter=(%.3g,%.3g,%.3g,%.3g)"
			"\n@@@\taccu=%.3g%%(T=%g,frr=%.3g far=%.2g%%)    EER=%.3g(%.3g) _DCRIMI_\n"
			, sTitle.c_str(), nz_a, nz_r, mean_a, devia_a, max_a, min_a, mean_r, devia_r, max_r, min_r,
			accu, sep, rFRR, rFAR * 100, rEER, eer_sep);
		for (i = 0; i<8; i++) {
			printf("(%.1e,%.3g)", f_ar_8[i], f_rr_8[i]);
		}
		printf("\n");
	}

}

void DCRIMI_::GetRoc(float *roc, int flag) {
	int i;
	double s, D_s, f_ar, f_rr, w_a = 0.0, w_r = 0.0, f_ar_g = 1.0e-7;
	for (i = 0; i < D_span; i++) {
		w_a += D_intra[i];		w_r += D_inter[i];
		roc[2 * i] = w_r*1.0 / nz_r;
		roc[2 * i + 1] = (nz_a - w_a)*1.0 / nz_a;
	}
}

double DCRIMI_::T_intra(int flag) {
	if (nz_a == 0)
		return 1.0;
	double t = mean_a + 7 * devia_a;
	t = min(t, 1.0);
	assert(t>0 && t <= 1.0);
	return t;
}